import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import json
import torch
from model.configuration_llama import MultimodalLlamaConfig
from model.modeling_llama import MultimodalLlamaForConditionalGeneration

from utils.llava_conversation_lib import conv_templates
from dataset.data_utils import tokenizer_image_token, preprocess_multimodal

from transformers import AutoTokenizer, BertTokenizerFast, AutoConfig
from utils.constants import IMAGE_TOKEN, PAD_TOKEN, LORA_CONFIG
import logging
import argparse
import wfdb
import orjson
from dataset.data_utils import preprocess_multimodal
import copy
import random
import csv

device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',
                    default="/home/hschung/llama-multimodal-vqa/src/model_checkpoints/07-19_17-35-26/checkpoint-20100",
                    help='Path to the pretrained model weights')
parser.add_argument('--ecg_path',
                    default=["/home/edlab/hschung/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/02000/02485_hr", "/home/edlab/hschung/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/04000/04849_hr"],
                    help='Path to the prompt image file')
parser.add_argument('--text_model_id',
                    default='meta-llama/Meta-Llama-3-8B-Instruct',
                    help='Huggingface model path to the language model used during training')
parser.add_argument('--user_question',
                    default="Which numeric features in the second ECG are now considered abnormal in relation to the first ECG?",
                    help='The question to ask about the ecg to the model. Example: "Does this ECG have AV Block?"')
parser.add_argument('--ecg_dataset',
                    default="ptbxl")
parser.add_argument('--output_path',
                    default="/home/hschung/llama-multimodal-vqa/ptbxl_test_samples/test_samples.pt")
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--num_questions', type=int, default=100)
parser.add_argument('--ecg_text', default=False, help='Use ECG text mode')
parser.add_argument('--encoding', default='patchtst', choices=['patchtst', 'ecg_encoder', 'vision_encoder'], help='Encoding method')
parser.add_argument('--data_type', default='raw_signal', choices=['signal_image', 'spectrogram_image', 'raw_signal'], help='Type of ECG data')
parser.add_argument('--vision_model_id', default='openai/clip-vit-large-patch14', help='Vision model ID for image encoding')

args = parser.parse_args()
logging.info(f"Parameters received: {args}")

logging.info("Loading pretrained model...")
# Register your custom config if not already done
AutoConfig.register("multimodal_llama", MultimodalLlamaConfig)

# Load the config from the model path
# config = AutoConfig.from_pretrained(args.model_path)

# model = MultimodalLlamaForConditionalGeneration.from_pretrained(
#     args.model_path,
#     config=config,
#     torch_dtype=torch.bfloat16
# ).to(device).eval()

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

dataset2 = []

for base in base_path:
    for dataset_type in ['test']:
        dataset_path = os.path.join(base, dataset_type)
        for file_name in os.listdir(dataset_path):
            if file_name.endswith('.json'):
                full_path = os.path.join(dataset_path, file_name)
                with open(full_path, 'r') as file:
                    data = orjson.loads(file.read())
                    # Assuming each item in data has an 'ecg_id' key
                    for item in data:
                        dataset2.append(item)  

#random shuffle
random.shuffle(dataset2)

#tokenizer 
tokenizer = AutoTokenizer.from_pretrained(args.text_model_id, use_fast=False)
tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
if tokenizer.pad_token is None or tokenizer.pad_token != PAD_TOKEN:
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
        tokenizer.pad_token = PAD_TOKEN
tokenizer_len = len(tokenizer)

# if args.ecg_text:
#     tokenizer2 = BertTokenizerFast.from_pretrained("bert-base-uncased")

# #set the pad tokens 
# model.config.image_token_index = tokenizer.additional_special_tokens_ids[0]
# model.config.pad_token_index = tokenizer.pad_token_id

                     
question_type_samples = {'comparison_irrelevant-query':[], 'single-verify':[], 'single-choose':[], 'single-query':[], 'comparison_consecutive-verify':[], 'comparison_irrelevant-verify':[], 'comparison_consecutive-query':[]}                      
                       
for item in dataset2: 
    # Check if all categories have reached 100 items
    if all(len(category) >= args.num_questions for category in question_type_samples.values()):
        break

    # Check if the current category has reached 100 items before loading ECG data
    if not len(question_type_samples[item['question_type']]) < args.num_questions:
        continue
    
    question_type_samples[item['question_type']].append(item['sample_id'])
    
    print(item['question_type'] + ":", len(question_type_samples[item['question_type']]))
    
torch.save({"sampled_ids": question_type_samples}, args.output_path)    
print("Finished Sampling")
     
