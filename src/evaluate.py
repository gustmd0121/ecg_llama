import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import math
import torch
from model.configuration_llama import MultimodalLlamaConfig
from model.modeling_llama import MultimodalLlamaForConditionalGeneration
from PIL import Image
from utils.llava_conversation_lib import conv_templates
from dataset.data_utils import tokenizer_image_token, preprocess_multimodal
from processing_llama import MultiModalLlamaProcessor
from transformers import AutoTokenizer, BertTokenizerFast, AutoConfig, AutoImageProcessor
from utils.constants import IMAGE_TOKEN, PAD_TOKEN, LORA_CONFIG
import logging
import argparse
import wfdb
import orjson
from dataset.data_utils import preprocess_multimodal
import copy
import random
import csv

torch.set_num_threads(32)

device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',
                    default="/nfs_edlab/hschung/ecg_llama/clip_frozen_image_with_presence_mask/final_model/",
                    help='Path to the pretrained model weights')
parser.add_argument('--ecg_path',
                    default=["/home/edlab/hschung/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/02000/02485_hr", "/home/edlab/hschung/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/04000/04849_hr"],
                    help='Path to the prompt image file')
parser.add_argument('--text_model_id',
                    default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                    help='Huggingface model path to the language model used during training')
parser.add_argument('--user_question',
                    default="Which numeric features in the second ECG are now considered abnormal in relation to the first ECG?",
                    help='The question to ask about the ecg to the model. Example: "Does this ECG have AV Block?"')
parser.add_argument('--ecg_dataset',
                    default="ptbxl")
parser.add_argument('--output_path',
                    default="/home/hschung/ecg-llm/ecg_llama/evaluation/clip_frozen_image_with_presence_mask_llama3.1_with_template_ids.pt")
parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
parser.add_argument('--num_questions', type=int, default=100)
parser.add_argument('--ecg_text', default=False, help='Use ECG text mode')
parser.add_argument('--encoding', default='vision_encoder', choices=['patchtst', 'ecg_encoder', 'vision_encoder'], help='Encoding method')
parser.add_argument('--data_type', default='signal_image', choices=['signal_image', 'spectrogram_image', 'raw_signal'], help='Type of ECG data')
parser.add_argument('--vision_model_id', default='openai/clip-vit-large-patch14', help='Vision model ID for image encoding')

args = parser.parse_args()
logging.info(f"Parameters received: {args}")

logging.info("Loading pretrained model...")
# Register your custom config if not already done
AutoConfig.register("multimodal_llama", MultimodalLlamaConfig)

# Load the config from the model path
config = AutoConfig.from_pretrained(args.model_path)

model = MultimodalLlamaForConditionalGeneration.from_pretrained(
    args.model_path,
    config=config,
    torch_dtype=torch.bfloat16,
    ignore_mismatched_sizes=True
).to(device).eval()

logging.info("Running model for ECG-QA...")

random.seed(args.seed)

# Load answers.csv and store template_id and classes in a dictionary
template_classes = {}
if args.ecg_dataset == 'mimic':
    with open('/home/hschung/llama-multimodal-vqa/answers_for_each_template_mimic.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            template_id = int(row[0])
            classes = eval(row[1])  # Evaluate the string representation of the list
            template_classes[template_id] = classes
elif args.ecg_dataset == 'ptbxl':
    with open('/home/hschung/llama-multimodal-vqa/answers_for_each_template_ptbxl.csv', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            template_id = int(row[0])
            classes = eval(row[1])  # Evaluate the string representation of the list
            template_classes[template_id] = classes

#load ecg dataset 
if args.ecg_dataset == 'all':
    base_path = ["/nfs_edlab/hschung/output/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/template/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/paraphrased/"]
elif args.ecg_dataset == 'ptbxl':
    base_path = ["/nfs_edlab/hschung/output/"]
else:
    base_path = ["/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/template/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/paraphrased/"]  

# Load the dataset from the specified path
dataset2 = torch.load('/home/hschung/ecg-llm/ecg_llama/ptbxl_test_dataset_1018/ptbxl_test_questions_by_types.pt')
# dataset3 = torch.load('/home/hschung/llama-multimodal-vqa/ptbxl_test_dataset_1018/ptbxl_test_questions_by_template.pt')

#tokenizer 
tokenizer = AutoTokenizer.from_pretrained(args.text_model_id, use_fast=False)
tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
if tokenizer.pad_token is None or tokenizer.pad_token != PAD_TOKEN:
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
        tokenizer.pad_token = PAD_TOKEN
tokenizer_len = len(tokenizer)
image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
processor = MultiModalLlamaProcessor(
    image_processor=image_processor,
    tokenizer=tokenizer
    )

if args.ecg_text:
    tokenizer2 = BertTokenizerFast.from_pretrained("bert-base-uncased")

#set the pad tokens 
model.config.image_token_index = tokenizer.additional_special_tokens_ids[0]
model.config.pad_token_index = tokenizer.pad_token_id
                     
generated_answers = {'comparison_irrelevant-query':[], 'single-verify':[], 'single-choose':[], 'single-query':[], 'comparison_consecutive-verify':[], 'comparison_irrelevant-verify':[], 'comparison_consecutive-query':[]}         
ground_truths = {'comparison_irrelevant-query':[], 'single-verify':[], 'single-choose':[], 'single-query':[], 'comparison_consecutive-verify':[], 'comparison_irrelevant-verify':[], 'comparison_consecutive-query':[]} 
questions_total = {'comparison_irrelevant-query':[], 'single-verify':[], 'single-choose':[], 'single-query':[], 'comparison_consecutive-verify':[], 'comparison_irrelevant-verify':[], 'comparison_consecutive-query':[]}               
template_ids = {'comparison_irrelevant-query':[], 'single-verify':[], 'single-choose':[], 'single-query':[], 'comparison_consecutive-verify':[], 'comparison_irrelevant-verify':[], 'comparison_consecutive-query':[]} 
                       
for question_type, questions in dataset2.items():
    for item in questions:
        if args.ecg_text:
            question = item['question']
            question_tokens = tokenizer2.encode(question, add_special_tokens=True)
            input_ids_m3ae = question_tokens[:tokenizer.model_max_length]
            input_ids_m3ae = torch.tensor(input_ids_m3ae, dtype=torch.long).unsqueeze(0).to(device)

            text = '<image>' + '\n' + item['question']
            conv = conv_templates["llama_3"].copy()
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0)

            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
            text_padding_mask = (input_ids_m3ae == tokenizer2.pad_token_id)

            ecg1, _ = wfdb.rdsamp(item['ecg_path'][0])
            ecg1 = torch.from_numpy(ecg1.T).unsqueeze(0)

            if len(item['ecg_path']) == 2:
                ecg2, _ = wfdb.rdsamp(item['ecg_path'][1])
                ecg2 = torch.from_numpy(ecg2.T).unsqueeze(0)
            else:
                ecg2 = torch.zeros(1, 12, 5000)

            ecg_padding_mask = torch.zeros_like(ecg1, dtype=torch.bool)
            ecg2_padding_mask = (ecg2 == 0).all(dim=1).all(dim=1).unsqueeze(1).unsqueeze(2).expand_as(ecg2)

            #inputs for m3ae
            inputs = {
                "input_ids": input_ids.to(device),
                "input_ids_m3ae": input_ids_m3ae.to(device),
                "pixel_values": ecg1.to(device),
                "ecg_values2": ecg2.to(device),
                "attention_mask": attention_mask.to(device),
                "text_padding_mask": text_padding_mask.to(device),
                "ecg_padding_mask": ecg_padding_mask.to(device),
                "ecg2_padding_mask": ecg2_padding_mask.to(device)
            }
        elif args.data_type == 'raw_signal':
            # Load ECG data
            ecg1, _ = wfdb.rdsamp(item['ecg_path'][0])
            ecg1 = torch.from_numpy(ecg1.T).unsqueeze(0)

            if len(item['ecg_path']) == 2:
                ecg2, _ = wfdb.rdsamp(item['ecg_path'][1])
                ecg2 = torch.from_numpy(ecg2.T).unsqueeze(0)
            else:
                ecg2 = torch.zeros(1, 12, 5000)

            ecg_padding_mask = torch.zeros_like(ecg1, dtype=torch.bool)
            ecg2_padding_mask = (ecg2 == 0).all(dim=1).all(dim=1).unsqueeze(1).unsqueeze(2).expand_as(ecg2)

            #Retrieve template id 
            template_id = item['template_id']
            options = ', '.join(template_classes[template_id])
            
            #prepare inputs for the model
            text = '<image>' + '\n' + item['question']
            conv = conv_templates["llama_3"].copy()
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0)
            
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
            # Prepare input tensors
            inputs = {
                "input_ids": input_ids.to(device),
                "pixel_values": torch.stack([ecg1, ecg2], dim=0).to(device, dtype=torch.bfloat16),
                "attention_mask": attention_mask.to(device),
                "ecg_padding_mask": ecg_padding_mask.to(device),
                "ecg2_padding_mask": ecg2_padding_mask.to(device)
            }
        
        elif args.data_type == 'signal_image':
            # Load and process first ECG image
            image_path1 = f"/nfs_edlab/hschung/ecg_plots/records500/10000/{item['ecg_id'][0]}_hr/{item['ecg_id'][0]}_hr.jpg"
            image1 = Image.open(image_path1).convert('RGB')
            image_tensor1 = processor.image_processor.preprocess(image1, return_tensors='pt')['pixel_values'][0]

            # Load and process second ECG image if it exists
            if len(item['ecg_path']) == 2:
                image_path2 = f"/nfs_edlab/hschung/ecg_plots/records500/02000/0{item['ecg_id'][1]}_hr/0{item['ecg_id'][1]}_hr.jpg"
                image2 = Image.open(image_path2).convert('RGB')
                image_tensor2 = processor.image_processor.preprocess(image2, return_tensors='pt')['pixel_values'][0]
            else:
                # If no second image, create zero tensor with same dimensions
                image_tensor2 = torch.zeros_like(image_tensor1)

            # Stack images
            stacked_images = torch.stack([image_tensor1, image_tensor2], dim=0)
            # Convert to model's dtype after stacking
            stacked_images = stacked_images.to(dtype=model.model_dtype)

            # Add appropriate number of image tokens to the question
            num_tokens = 1 if len(item['ecg_path']) == 2 else 1
            text = (IMAGE_TOKEN * num_tokens) + '\n' + item['question']
            
            # Prepare conversation and process text inputs
            conv = conv_templates["llama_3"].copy()
            conv.append_message(conv.roles[0], text)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors='pt').unsqueeze(0)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            inputs = {
                "input_ids": input_ids.to(device),
                "pixel_values": stacked_images.to(device),
                "attention_mask": attention_mask
            }
         
        # Generate the response
        with torch.no_grad():
            outputs = model.generate(**inputs, num_beams=1, max_new_tokens=50, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

        # Decode the generated tokens
        response = tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
        
        ground_truths[question_type].append(', '.join(item['answer']))
        generated_answers[question_type].append(response)
        questions_total[question_type].append(item["question"])
        template_ids[question_type].append(item["template_id"])
        
        print("Question type:", question_type)
        print("Ground truth:", ', '.join(item['answer']))
        print("Generated response:", response)
        print()

torch.save({"generated_answers":generated_answers, "ground_truths":ground_truths, "questions": questions_total, "template_ids":template_ids}, args.output_path)