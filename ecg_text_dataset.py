import warnings
from transformers import logging as transformers_logging

# Add these lines at the beginning of your script, after your imports
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.hub")
transformers_logging.set_verbosity_error()

import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3,4,5,6,7'
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
import scipy.io
from transformers import PreTrainedModel, AutoConfig, AutoTokenizer
import argparse
import wandb
import pickle
from tqdm import tqdm

torch.set_num_threads(32)

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ECGCLIPDataset(Dataset):
    def __init__(self, ptbxl_dir, mimic_dir, tokenizer, cache_file=None):
        self.ptbxl_dir = ptbxl_dir
        self.mimic_dir = mimic_dir
        self.tokenizer = tokenizer
        self.cache_file = cache_file

        if cache_file and os.path.exists(cache_file):
            print(f"Loading file paths from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.files = pickle.load(f)
        else:
            print("Scanning directories for .mat files...")
            self.files = self._scan_directories()
            if cache_file:
                print(f"Saving file paths to cache: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.files, f)

        print(f"Total number of files: {len(self.files)}")

    def _scan_directories(self):
        ptbxl_files = [os.path.join(self.ptbxl_dir, f) for f in os.listdir(self.ptbxl_dir) if f.endswith('.mat')]
        mimic_files = [os.path.join(self.mimic_dir, f) for f in os.listdir(self.mimic_dir) if f.endswith('.mat')]
        return ptbxl_files + mimic_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        while True:
            try:
                path = self.files[idx]
                data = scipy.io.loadmat(path)
                
                if 'feats' not in data or 'text' not in data or len(data['text']) == 0 or not isinstance(data['text'][0], str):
                    raise ValueError("Invalid data format")
                
                ecg = torch.from_numpy(data['feats']).float()
                text = data['text'][0]
                
                tokenized = self.tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
                
                return {
                    'ecg': ecg,
                    'input_ids': tokenized['input_ids'].squeeze(),
                    'attention_mask': tokenized['attention_mask'].squeeze()
                }
            except Exception as e:
                print(f"Error processing file at index {idx}: {str(e)}. Skipping to next file.")
                idx = (idx + 1) % len(self.files)  # Move to the next file, wrapping around if necessary

class ECGEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=12
        )
        self.projection = nn.Linear(256, 768)

    def forward(self, x):
        x = self.conv_layers(x)  # Shape: [batch, 256, 312]
        x = x.permute(2, 0, 1)  # Shape: [312, batch, 256]
        x = self.transformer(x)
        x = x.permute(1, 0, 2) 
        x = self.projection(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=8),
            num_layers=12
        )
        self.projection = nn.Linear(config.hidden_size, 768)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = x.permute(1, 0, 2)  # Shape: [seq_len, batch, hidden_size]
        mask = attention_mask.bool().logical_not()
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.mean(dim=0)  # Global average pooling
        x = self.projection(x)
        return x

class ECGCLIPModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.ecg_encoder = ECGEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, ecg, input_ids, attention_mask):
        ecg_features = self.ecg_encoder(ecg)  # Shape: [batch, 312, 768]
        text_features = self.text_encoder(input_ids, attention_mask)  # Shape: [batch, 768]

        # Average ECG features across time dimension
        ecg_features_avg = ecg_features.mean(dim=1)  # Shape: [batch, 768]

        # Normalize features
        ecg_features_avg = ecg_features_avg / ecg_features_avg.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        logit_scale = self.logit_scale.exp()
        logits_per_ecg = logit_scale * ecg_features_avg @ text_features.t()
        logits_per_text = logits_per_ecg.t()

        return logits_per_ecg, logits_per_text, ecg_features

def train(args, model, train_loader, optimizer, scheduler, epoch, rank):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=rank != 0)
    for batch_idx, batch in enumerate(pbar):
        ecg = batch['ecg'].to(rank)
        input_ids = batch['input_ids'].to(rank)
        attention_mask = batch['attention_mask'].to(rank)

        optimizer.zero_grad()
        logits_per_ecg, logits_per_text, _ = model(ecg, input_ids, attention_mask)
        
        labels = torch.arange(len(logits_per_ecg)).to(rank)
        loss_ecg = F.cross_entropy(logits_per_ecg, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_ecg + loss_text) / 2

        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        if rank == 0:
            pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})

            if batch_idx % args.log_interval == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "epoch": epoch,
                    "step": batch_idx,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "ecg_loss": loss_ecg.item(),
                    "text_loss": loss_text.item(),
                    "logit_scale": model.module.logit_scale.exp().item()
                })

    avg_loss = total_loss / len(train_loader)
    if rank == 0:
        wandb.log({"train_epoch_loss": avg_loss, "epoch": epoch})
    return avg_loss

def validate(args, model, val_loader, rank):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", disable=rank != 0)
        for batch in pbar:
            ecg = batch['ecg'].to(rank)
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)

            logits_per_ecg, logits_per_text, _ = model(ecg, input_ids, attention_mask)
            
            labels = torch.arange(len(logits_per_ecg)).to(rank)
            loss_ecg = F.cross_entropy(logits_per_ecg, labels)
            loss_text = F.cross_entropy(logits_per_text, labels)
            loss = (loss_ecg + loss_text) / 2

            total_loss += loss.item()
            if rank == 0:
                pbar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(val_loader)
    if rank == 0:
        print(f'Validation Loss: {avg_loss:.6f}')
        wandb.log({"val_loss": avg_loss})
    return avg_loss

def run(rank, world_size, args):
    setup(rank, world_size)

    set_seed(args.seed + rank)  # Different seed for each process

    if rank == 0:
        wandb.init(project="ecg-clip", config=args)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    full_dataset = ECGCLIPDataset(args.ptbxl_dir, args.mimic_dir, tokenizer, cache_file=args.cache_file)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    if args.resume:
        print(f"Loading pre-trained model from {args.model_path}")
        model = ECGCLIPModel.from_pretrained(args.model_path).to(rank)
    else:
        config = AutoConfig.from_pretrained('bert-base-uncased')
        config.vocab_size = tokenizer.vocab_size
        model = ECGCLIPModel(config).to(rank)
    
    # Initialize the model weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    model.apply(init_weights)
    
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Implement a warmup period
    warmup_steps = 1000
    total_steps = args.epochs * len(train_loader)
    def warmup_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    #Load optimizer and scheduler states if resuming
    if args.resume:
        checkpoint = torch.load(os.path.join(args.model_path, 'training_state.pt'), map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
    else:
        start_epoch = 1
        best_val_loss = float('inf')
        
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_loss = train(args, model, train_loader, optimizer, scheduler, epoch, rank)
        val_loss = validate(args, model, val_loader, rank)

        if rank == 0:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": scheduler.get_last_lr()[0]
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save the model using save_pretrained
                model.module.save_pretrained(args.save_dir)
                tokenizer.save_pretrained(args.save_dir)
                
                # Save optimizer and scheduler states separately
                torch.save({
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_loss': best_val_loss
                }, os.path.join(args.save_dir, 'optimizer_scheduler.pt'))
                
                print(f"New best model saved at epoch {epoch}")

    if rank == 0:
        wandb.finish()

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptbxl_dir', type=str, default='/nfs_edlab/hschung/m3ae/ptbxl_translated/')
    parser.add_argument('--mimic_dir', type=str, default='/nfs_edlab/hschung/m3ae/mimic/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)  # Reduced learning rate
    parser.add_argument('--epochs', type=int, default=300)  # Increased number of epochs
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./ecg_clip_model')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache_file', type=str, default='dataset_cache.pkl', help='File to cache dataset file paths')
    parser.add_argument('--resume', action='store_true', help='Resume training from a saved model')
    parser.add_argument('--model_path', type=str, default='/home/hschung/ecg_clip/ecg_clip_model_epoch50', help='Path to the saved model for resuming training')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(run, args=(world_size, args), nprocs=world_size, join=True)