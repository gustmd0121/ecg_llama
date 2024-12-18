"""
Main entrypoint for training the language-vision model.
"""

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
import pathlib
import sys
import logging
from transformers import TrainingArguments
import torch 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset.data_handling import create_dataset
from model.model_utils import build_model
from utils.utils import get_available_device, set_seed, make_save_folder
from trainer_llama import MultimodalLlamaTrainer
import json

torch.set_num_threads(32)


logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


def run_inference(model, tokenizer, data_module, device, model_path):
    """
    Perform inference with a trained model.

    Args:
        model: Placeholder for the model (not needed for loading from disk).
        tokenizer: The tokenizer to decode outputs.
        data_module: Data module containing the test dataset.
        device: The device to run the inference on.
        model_path: Path to the saved Hugging Face model.

    Returns:
        List of inference results.
    """
    # Load the model from the saved directory
    model = model.from_pretrained(args.model_path).to(device)
    model.eval()

    # Load the tokenizer from the saved directory
    tokenizer = tokenizer.from_pretrained(args.model_path)

    results = []
    for batch in data_module['eval_dataset']:
        inputs = {
            "input_ids": batch[0]["input_ids"].unsqueeze(0).to(device),
            "labels": batch[0]["labels"].to(device)
        }
        
        if batch[0].get("image") is not None:
            inputs["pixel_values"] = batch[0]["image"].unsqueeze(0).to(device)
        if batch[0].get("image2") is not None:
            inputs["images2"] = batch[0]["image2"].unsqueeze(0).to(device)
        else:
            inputs["images2"] = torch.zeros((1, 3, 224, 224)).to(device)
            
        if batch[0].get("ecg") is not None:
            is_zero_sample1 = (batch[0]['ecg'].unsqueeze(0).sum(dim=(1, 2)) == 0)
            inputs["ecg"] = batch[0]["ecg"].unsqueeze(0).to(device)
            inputs["ecg_padding_mask"] = is_zero_sample1.unsqueeze(1).unsqueeze(2).expand(batch[0]['ecg'].unsqueeze(0).size()).clone()
        if batch[0].get("ecg2") is not None:
            inputs["ecg2"] = batch[0]["ecg2"].unsqueeze(0).to(device)
            inputs["ecg_padding_mask2"] = inputs["ecg_padding_mask"]
        else:
            inputs["ecg2"] = torch.zeros((1, 12, 5000)).to(device)
            inputs["ecg_padding_mask2"] = inputs["ecg_padding_mask"]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=1,
                max_new_tokens=50,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        if batch[0].get('conversation') is not None:
            conversation = batch[0]['conversation']
            question = next((item['value'] for item in conversation if item['from'] == 'human'), None)
            answer = next((item['value'] for item in conversation if item['from'] == 'gpt'), None)
            results.append({
            'question': question,
            'answer': answer,
            'response': response
            })

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_model_id',
                        default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                        help='Huggingface model path to the language model. Examples:'
                             '- meta-llama/Llama-2-13b-chat-hf'
                             '- lmsys/vicuna-7b-v1.5'
                             '- meta-llama/Meta-Llama-3-8B-Instruct')

    parser.add_argument('--vision_model_id',
                        default='openai/clip-vit-large-patch14',
                        help='Huggingface model path to the vision model. Examples:'
                             '- openai/clip-vit-large-patch14')

    parser.add_argument('--data_paths',
                        nargs='+',
                        default=['/nfs_edlab/hschung/ptbxl_ecg_mapping/paraphrased_ecg_signals/', '/nfs_edlab/hschung/mimic_ecg_mapping/paraphrased_ecg_signals/'],
                        choices=[['/nfs_edlab/hschung/ptbxl_ecg_mapping/paraphrased_ecg_signals/', '/nfs_edlab/hschung/mimic_ecg_mapping/paraphrased_ecg_signals/'],["/nfs_edlab/hschung/ptbxl_ecg_mapping/paraphrased_hf_ecg_images/", "/nfs_edlab/hschung/mimic_ecg_mapping/paraphrased_hf_ecg_images/"],["/nfs_edlab/hschung/ptbxl_ecg_mapping/paraphrased_hf_ecg_spectrogram_images/", "/nfs_edlab/hschung/mimic_ecg_mapping/paraphrased_hf_ecg_spectrogram_images/"]],
                        help='List of paths to directories containing train and valid subdirectories with JSON files')

    parser.add_argument('--checkpoint_save_path',
                        default='./model_checkpoints/',
                        help='Path to the folder where to save the model checkpoints during training')

    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="The batch size to use for training. Default: 16")

    parser.add_argument('--report_to',
                        default='none',
                        choices=['wandb', 'none'],
                        help='Which reporting tool to use. Options: wandb, none. Default: none')

    parser.add_argument('--max_eval_samples',
                        type=int,
                        default=100,
                        help='Maximum number of samples to use for evaluation. Using 384 samples '
                             'provides 95% confidence level with 5% margin of error')

    # Add new arguments for distributed training and quantization
    parser.add_argument('--load_in_8bit', 
                       action='store_true',
                       help='Load model in 8-bit quantization')
    
    parser.add_argument('--num_gpus',
                       type=int,
                       default=-1,
                       help='Number of GPUs to use (-1 for all available)')
    
    parser.add_argument('--data_type', default='signal', choices=['image', 'signal'])
    
    parser.add_argument('--inference', default=True)
    
    parser.add_argument('--model_path',
                        default="/home/hschung/ecg-llm/llama-multimodal-vqa/src/model_checkpoints/w2v_cmsc_rlm_unfrozen_with_presence_mask/final_model",
                        help='Path to the pretrained model weights')
    
    args = parser.parse_args(sys.argv[1:])
    logging.info(f"Parameters received: {args}")

    if args.num_gpus > 1:
        # Initialize process group
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = get_available_device()

    # Reproducibility
    set_seed()

    logging.info("Building model stack...")
    model_stack = build_model(text_model_id=args.text_model_id,
                              vision_model_id=args.vision_model_id,
                              freeze_vision_model=False,
                              freeze_language_model=False,
                              freeze_multimodal_projector=False,
                              device=device,
                              use_bfloat16=not args.load_in_8bit,  # Don't use bf16 with 8-bit
                              load_in_8bit=args.load_in_8bit,
                              data_type=args.data_type,
                              )

    if args.num_gpus > 1:
        model_stack['model'] = DDP(
            model_stack['model'],
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # Enable finding unused parameters
            broadcast_buffers=False  # Disable broadcasting buffers for better performance
        )

    logging.info("Building data module...")
    data_module = create_dataset(tokenizer=model_stack['tokenizer'],
                                 image_processor=model_stack['processor'].image_processor,
                                 is_multimodal=True,
                                 data_paths=args.data_paths,  # Changed from data_path
                                 image_folder="data/images",
                                 image_aspect_ratio="pad",
                                 config=model_stack['config'],
                                 max_eval_samples=args.max_eval_samples,
                                 inference=args.inference)  # Add this parameter

    output_dir = make_save_folder(args.checkpoint_save_path)
    logging.info(f"Output dir is: {output_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=2,  # Increased for multi-GPU
        num_train_epochs=1,
        gradient_accumulation_steps=2,
        save_total_limit=2,
        learning_rate=2e-4,
        weight_decay=0.,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=500,  # Increased to reduce checkpoint frequency
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        do_eval=True,
        bf16=not args.load_in_8bit,  # Don't use bf16 with 8-bit
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=args.report_to,
        max_grad_norm=1.0,
        local_rank=local_rank if args.num_gpus > 1 else -1,
        deepspeed=None,  # Can add DeepSpeed config if needed
        ddp_find_unused_parameters=True,  # Add this line
    )

    if training_args.gradient_checkpointing:
        model_stack['model'].gradient_checkpointing_enable()

    logging.info("Building trainer...")
    trainer = MultimodalLlamaTrainer(model=model_stack['model'],
                                     tokenizer=model_stack['tokenizer'],
                                     group_by_modality_length=True,
                                     args=training_args,
                                     **data_module)

    if not args.inference:
        logging.info("Starting training...")
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()

        logging.info("Saving final model...")
        trainer.save_model(os.path.join(output_dir, "final_model"))
    else:
        logging.info("Running inference...")
        results = run_inference(model_stack['model'], model_stack['tokenizer'], data_module, device, args.model_path)
        logging.info(f"Inference results: {results}")

        # Save the inference results to a file
        results_file = os.path.join(output_dir, "inference_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Inference results saved to {results_file}")
