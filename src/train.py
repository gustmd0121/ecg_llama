"""
Main entrypoint for training the language-vision model.
"""

import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES']='7'
import pathlib
import sys
import logging
from transformers import TrainingArguments
import torch 
from dataset.data_handling import create_dataset
from model.model_utils import build_model
from utils.utils import get_available_device, set_seed, make_save_folder
from trainer_llama import MultimodalLlamaTrainer

torch.set_num_threads(32)


logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


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
                        choices=[['/nfs_edlab/hschung/ptbxl_ecg_mapping/paraphrased_ecg_signals/', '/nfs_edlab/hschung/mimic_ecg_mapping/paraphrased_ecg_signals/'],["/nfs_edlab/hschung/ptbxl_ecg_mapping/paraphrased_hf_ecg_images/", "/nfs_edlab/hschung/mimic_ecg_mapping/paraphrased_hf_ecg_images/"]],
                        help='List of paths to directories containing train and valid subdirectories with JSON files')

    parser.add_argument('--checkpoint_save_path',
                        default='./model_checkpoints/',
                        help='Path to the folder where to save the model checkpoints during training')

    parser.add_argument("--batch_size",
                        type=int,
                        default=26,
                        help="The batch size to use for training. Default: 16")

    parser.add_argument('--report_to',
                        default='wandb',
                        choices=['wandb', 'none'],
                        help='Which reporting tool to use. Options: wandb, none. Default: none')

    parser.add_argument('--max_eval_samples',
                        type=int,
                        default=4000,
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

    args = parser.parse_args(sys.argv[1:])
    logging.info(f"Parameters received: {args}")

    # Reproducibility
    set_seed()

    logging.info("Building model stack...")
    model_stack = build_model(text_model_id=args.text_model_id,
                              vision_model_id=args.vision_model_id,
                              freeze_vision_model=True,
                              freeze_language_model=False,
                              freeze_multimodal_projector=False,
                              device=get_available_device(),
                              use_bfloat16=not args.load_in_8bit,  # Don't use bf16 with 8-bit
                              load_in_8bit=args.load_in_8bit,
                              data_type=args.data_type)

    logging.info("Building data module...")
    data_module = create_dataset(tokenizer=model_stack['tokenizer'],
                                 image_processor=model_stack['processor'].image_processor,
                                 is_multimodal=True,
                                 data_paths=args.data_paths,  # Changed from data_path
                                 image_folder="data/images",
                                 image_aspect_ratio="pad",
                                 config=model_stack['config'],
                                 max_eval_samples=args.max_eval_samples)  # Add this parameter

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
        # Add distributed training args
        ddp_find_unused_parameters=False,
        max_grad_norm=1.0,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        deepspeed=None  # Can add DeepSpeed config if needed
    )

    if training_args.gradient_checkpointing:
        model_stack['model'].gradient_checkpointing_enable()

    logging.info("Building trainer...")
    trainer = MultimodalLlamaTrainer(model=model_stack['model'],
                                     tokenizer=model_stack['tokenizer'],
                                     group_by_modality_length=True,
                                     args=training_args,
                                     **data_module)

    logging.info("Starting training...")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    logging.info("Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
