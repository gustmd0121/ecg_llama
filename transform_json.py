import os
import json

def transform_json(input_data):
    transformed_data = []
    for item in input_data:
        new_item = {
            "template_id": item["template_id"],
            "question_id": item["question_id"],
            "sample_id": item["sample_id"],  # Adjust sample_id as needed
            "question_type": item["question_type"],
            "attribute_type": item["attribute_type"],
            "ecg_id": item["ecg_id"],
            "ecg": [path for path in item["ecg_path"]],
            "conversations": [
                {
                    "from": "human",
                    "value": item["question"]
                },
                {
                    "from": "gpt",
                    "value": ", ".join(item["answer"])
                }
            ]
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
                
                output_file_path = os.path.join(output_dir, file)
                with open(output_file_path, 'w') as f:
                    json.dump(transformed_data, f, indent=4)

if __name__ == "__main__":
    base_directory = "/nfs_edlab/hschung/mimic_ecg_mapping/paraphrased/"
    transformed_base = os.path.join(base_directory, "transformed")
    
    # Create directories for each split in the transformed directory
    for split in ["train", "valid", "test"]:
        input_dir = os.path.join(base_directory, split)
        output_dir = os.path.join(transformed_base, split)
        process_directory(input_dir, output_dir)