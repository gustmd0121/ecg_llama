"""
Main entrypoint for Visual-Question Answering with a pretrained model
Example run: python src/vqa.py --model_path="./model_checkpoints/04-23_18-53-28/checkpoint-1000" --testset_path="./data/testset.json"
"""
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import argparse
import sys
import logging
import json

import torch
from PIL import Image
from transformers import AutoTokenizer, BertTokenizerFast, AutoConfig
from model.configuration_llama import MultimodalLlamaConfig
from model.modeling_llama import MultimodalLlamaForConditionalGeneration
from processing_llama import MultiModalLlamaProcessor
from utils.utils import get_available_device
from utils.constants import IMAGE_TOKEN, PAD_TOKEN


torch.set_num_threads(32)

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        default="/home/hschung/ecg-llm/llama-multimodal-vqa/src/model_checkpoints/11-06_08-26-24/final_model",
                        help='Path to the pretrained model weights')

    parser.add_argument('--testset_path',
                        default="/nfs_edlab/hschung/ptbxl_ecg_mapping/paraphrased_hf_ecg_images/test/00000.json",
                        help='Path to the test set JSON file')

    args = parser.parse_args(sys.argv[1:])
    logging.info(f"Parameters received: {args}")

    logging.info("Loading pretrained model...")
    device = get_available_device()
    
    logging.info(f"Parameters received: {args}")

    logging.info("Loading pretrained model...")
    # Register your custom config if not already done
    AutoConfig.register("multimodal_llama", MultimodalLlamaConfig)
    
    # Load the config from the model path
    config = AutoConfig.from_pretrained(args.model_path)
    
    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    if tokenizer.pad_token is None or tokenizer.pad_token != PAD_TOKEN:
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
        tokenizer.pad_token = PAD_TOKEN

    # Update config with new tokenizer length
    config.tokenizer_len = len(tokenizer)

    # Load the model with token resizing disabled initially
    multimodal_llama_model = MultimodalLlamaForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        # Add these parameters to prevent token resizing issues
    ).eval()

    # Manually resize token embeddings after loading
    if len(tokenizer) != multimodal_llama_model.config.text_config.vocab_size:
        multimodal_llama_model.resize_token_embeddings(len(tokenizer))

    processor = MultiModalLlamaProcessor.from_pretrained(args.model_path)

    logging.info("Running model for VQA...")

    with open(args.testset_path, 'r') as f:
        testset = json.load(f)

    results = []
    for sample in testset:
        image_path = sample['image']
        user_question = sample['question']
        prompt = (f"<|start_header_id|>user<|end_header_id|> <image>\n{user_question} <|eot_id|>\n"
                  f"<|start_header_id|>assistant<|end_header_id|>:")

        raw_image = Image.open(image_path)
        inputs = processor(prompt, raw_image, return_tensors='pt').to(device, torch_dtype=torch.bfloat16)

        output = multimodal_llama_model.generate(**inputs,
                                                 max_new_tokens=100,
                                                 do_sample=False)
        answer = processor.decode(output[0][2:], skip_special_tokens=True)
        results.append({'question': user_question, 'answer': answer, 'image': image_path})

    with open('vqa_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    logging.info("Inference completed. Results saved to vqa_results.json")
