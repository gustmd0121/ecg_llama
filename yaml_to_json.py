import yaml
import json

def convert_yaml_to_json(yaml_file_path, json_file_path):
    # Read the YAML file
    with open(yaml_file_path, 'r') as yaml_file:
        yaml_content = yaml.safe_load(yaml_file)

    # Convert the dictionary to a JSON string
    json_content = json.dumps(yaml_content, indent=2)

    # Write the JSON string to a file
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_content)

# Example usage
yaml_file_path = '/home/hschung/ecg-llm/llama-multimodal-vqa/ckpts/mimic_iv_ecg_physionet_pretrained.yaml'
json_file_path = '/home/hschung/ecg-llm/llama-multimodal-vqa/ckpts/mimic_iv_ecg_physionet_pretrained.json'
convert_yaml_to_json(yaml_file_path, json_file_path)