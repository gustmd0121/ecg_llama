import json
import os
from pathlib import Path

def convert_sample(sample):
    converted = {
        "template_id": sample["template_id"],
        "question_id": sample["question_id"],
        "sample_id": sample["sample_id"],
        "question_type": sample["question_type"],
        "attribute_type": sample["attribute_type"],
        "ecg_id": sample["ecg_id"],
        "image": sample["ecg_path"],
        "conversations": [
            {
                "from": "human",
                "value": sample["question"]
            },
            {
                "from": "gpt",
                "value": ", ".join(sample["answer"])
            }
        ]
    }
    return converted

def process_file(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    converted_data = [convert_sample(sample) for sample in data]
    
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=4)

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir in ['train', 'valid', 'test']:
        input_subdir = os.path.join(input_dir, subdir)
        output_subdir = os.path.join(output_dir, subdir)
        
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        for file in os.listdir(input_subdir):
            if file.endswith('.json'):
                input_file = os.path.join(input_subdir, file)
                output_file = os.path.join(output_subdir, file)
                process_file(input_file, output_file)
                print(f"Processed {input_file} -> {output_file}")
input_dir = "/nfs_edlab/hschung/mimic_ecg_mapping/template/"
output_dir = "/nfs_edlab/hschung/mimic_ecg_mapping/template_hf_ecg_images/"


# Process each directory
process_directory(input_dir, output_dir)

print("Conversion completed.")