import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import pathlib
import sys
import logging
from transformers import TrainingArguments
import torch 
from dataset.data_handling import create_dataset
from model.model_utils import build_model
from utils.utils import get_available_device, set_seed, make_save_folder
from trainer_llama import MultimodalLlamaTrainer


logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')

torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_model_id',
                        default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                        help='Huggingface model path to the language model. Examples:'
                             '- meta-llama/Llama-2-13b-chat-hf'
                             '- lmsys/vicuna-7b-v1.5'
                             '- meta-llama/Meta-Llama-3-8B-Instruct'
                             '- meta-llama/Meta-Llama-3.1-8B-Instruct'
                             '- google/gemma-2-9b-it')
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument('--vision_model_id',
                        default='openai/clip-vit-large-patch14',
                        help='Huggingface model path to the vision model. Examples:'
                             '- openai/clip-vit-large-patch14')
    parser.add_argument('--data_path',
                        default='/nfs_edlab/hschung/llava_instruct_80k.json',
                        help='Path to the visual-instruction dataset for finetuning')
    parser.add_argument('--checkpoint_save_path',
                        default='/nfs_edlab/hschung/model_checkpoints/',
                        help='Path to the folder where to save the model checkpoints during training')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs to use for training')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.set_defaults(load_in_4bit=False)
    parser.add_argument("--batch_size",
                        type=int,
                        default=2
                        ,
                        help="The batch size to use for training. Default: 16")
    parser.add_argument('--report_to',
                        default='wandb',
                        choices=['wandb', 'none'],
                        help='Which reporting tool to use. Options: wandb, none. Default: none')
    parser.add_argument('--ecg_dataset',
                        default='all',
                        choices=['all', 'ptbxl', 'mimic'],
                        help='ECG-QA Dataset used')

    #PatchTST or ECG Encoder
    parser.add_argument('--encoding',
                        default='ecg_clip',
                        choices=['patchtst', 'ecg_encoder', 'ecg_clip', 'vision_encoder'],
                        help='ECG-QA Dataset used')

    #ecg-text multimodal
    parser.add_argument('--ecg_text', default=False)
    
    #vision_model
    parser.add_argument('--freeze_vision', default=False)
    parser.add_argument('--data_type', default='raw_signal', choices=['signal_image', 'spectrogram_image', 'wavelet_image', 'raw_signal'])

    args = parser.parse_args(sys.argv[1:])
    logging.info(f"Parameters received: {args}")

    # Reproducibility
    set_seed()

    # Add this code snippet here
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

    not_ecg_text = not args.ecg_text
    
    logging.info("Building model stack...")
    model_stack = build_model(text_model_id=args.text_model_id,
                              vision_model_id=args.vision_model_id,
                              freeze_vision_model=args.freeze_vision,
                              freeze_language_model=False,
                              freeze_multimodal_projector=False,
                              device="cuda",
                              use_bfloat16=True,
                              load_in_4bit=args.load_in_4bit,
                              ecg_text=args.ecg_text,
                              encoding = args.encoding)

    logging.info("Building data module...")
    data_module = create_dataset(tokenizer=model_stack['tokenizer'],
                                 tokenizer2=model_stack['tokenizer2'],
                                 image_processor=model_stack['processor'].image_processor,
                                 is_multimodal=True,
                                 data_path=args.data_path,
                                 image_folder="/nfs_edlab/hschung/train2017",
                                 image_aspect_ratio="pad",
                                 config=model_stack['config'],
                                 ecg_dataset=args.ecg_dataset,
                                 ecg_text=args.ecg_text,
                                 encoding = args.encoding,
                                 data_type=args.data_type)

    output_dir = make_save_folder(args.checkpoint_save_path)
    logging.info(f"Output dir is: {output_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        dataloader_num_workers=4,
        num_train_epochs=3,
        gradient_accumulation_steps=4,
        save_total_limit=1,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=300,
        logging_steps=10,
        evaluation_strategy="steps",
        do_eval=True,
        eval_steps=300,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=args.report_to,
        deepspeed="/home/hschung/llama-multimodal-vqa/ds_config.json",
        per_device_train_batch_size=4,  # Updated to match deepspeed_config.json
        max_grad_norm=0.3,
        local_rank=args.local_rank,
    )

    if training_args.gradient_checkpointing:
        model_stack['model'].gradient_checkpointing_enable()
    
    logging.info("Building trainer...")
    trainer = MultimodalLlamaTrainer(model=model_stack['model'],
                                     tokenizer=model_stack['tokenizer'],
                                     group_by_modality_length=False,
                                     args=training_args,
                                     **data_module)

    logging.info("Starting training...")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    logging.info("Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
