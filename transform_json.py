import os
import json

def clean_path(path):
    # Split the path into components
    parts = path.split('/')
    # Remove the redundant directory (second to last component)
    if len(parts) > 1:
        parts.pop(-2)
    return '/'.join(parts)

def transform_json(input_data):
    transformed_data = []
    for item in input_data:
        new_item = {
            "template_id": item["template_id"],
            "question_id": item["question_id"],
            "sample_id": item["sample_id"],
            "question_type": item["question_type"],
            "attribute_type": item["attribute_type"],
            "ecg_id": item["ecg_id"],
            "image": [clean_path(path) for path in item["image"]],
            "conversations": item["conversations"]
        }
        transformed_data.append(new_item)
    return transformed_data

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                transformed_data = transform_json(data)
                
                output_file_path = os.path.join(output_dir, os.path.relpath(file_path, input_dir))
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                with open(output_file_path, 'w') as f:
                    json.dump(transformed_data, f, indent=4)

input_dir = "/nfs_edlab/hschung/ptbxl_ecg_mapping/paraphrased_hf_ecg_spectrogram_images"
output_dir = "/nfs_edlab/hschung/ptbxl_ecg_mapping/paraphrased_hf_ecg_spectrogram_images"
process_directory(input_dir, output_dir)