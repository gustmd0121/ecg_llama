import json
import os

def convert_sample(sample):
    # Convert the image paths
    sample["image"] = [
        img_path.replace(
            "/nfs_data_storage/mimic-iv-ecg/files/",
            "/nfs_edlab/hschung/ecg_plots/files/"
        )
        for img_path in sample["image"]
    ]

    # Convert the image paths
    sample["image"] = [
        os.path.join(img_path, f"{os.path.basename(img_path)}.jpg")
        for img_path in sample["image"]
    ]    

    return sample

def process_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    converted_data = [convert_sample(sample) for sample in data]
    
    with open(file_path, 'w') as f:
        json.dump(converted_data, f, indent=4)

def process_directory(input_dir):
    for subdir in ['train', 'valid', 'test']:
        input_subdir = os.path.join(input_dir, subdir)
        
        for file in os.listdir(input_subdir):
            if file.endswith('.json'):
                file_path = os.path.join(input_subdir, file)
                process_file(file_path)
                print(f"Processed {file_path}")

input_dir = "/nfs_edlab/hschung/mimic_ecg_mapping/template_hf_ecg_images/"

# Process each directory
process_directory(input_dir)

print("Conversion completed.")