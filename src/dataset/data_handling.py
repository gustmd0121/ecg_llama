import glob
import os
from dataset.data_classes import SupervisedDataset, DataCollatorForSupervisedDataset
from dataset.data_utils import get_preprocess_func, preprocess, preprocess_llama_2, preprocess_llama_3
import torch

def get_json_files(directory):
    """Get all JSON files from a directory."""
    return sorted(glob.glob(os.path.join(directory, "*.json")))

def create_dataset(tokenizer, image_processor, data_paths, image_folder, image_aspect_ratio, 
                  is_multimodal, config, max_eval_samples=None, inference=False):
    """Make dataset and collator for supervised fine-tuning."""
    
    preprocess_func = get_preprocess_func(config.text_model_id)
    
    train_datasets = []
    valid_datasets = []
    
    # Iterate through each data directory
    for data_path in data_paths:
        # Handle train and validation directories for each data path
        train_jsons = get_json_files(os.path.join(data_path, "train"))
        valid_jsons = get_json_files(os.path.join(data_path, "valid"))
        test_jsons = get_json_files(os.path.join(data_path, "test"))
        
        # Create train datasets
        for json_file in train_jsons:
            dataset = SupervisedDataset(
                tokenizer=tokenizer,
                image_processor=image_processor,
                data_path=json_file,
                image_folder=image_folder,
                image_aspect_ratio=image_aspect_ratio,
                is_multimodal=is_multimodal,
                preprocess_func=preprocess_func
            )
            train_datasets.append(dataset)
        
        if not inference: 
            # Create validation datasets
            for json_file in valid_jsons:
                dataset = SupervisedDataset(
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    data_path=json_file,
                    image_folder=image_folder,
                    image_aspect_ratio=image_aspect_ratio,
                    is_multimodal=is_multimodal,
                    preprocess_func=preprocess_func
                )
                valid_datasets.append(dataset)
        else:
            # Create test datasets
            for json_file in test_jsons:
                dataset = SupervisedDataset(
                    tokenizer=tokenizer,
                    image_processor=image_processor,
                    data_path=json_file,
                    image_folder=image_folder,
                    image_aspect_ratio=image_aspect_ratio,
                    is_multimodal=is_multimodal,
                    preprocess_func=preprocess_func
                )
                valid_datasets.append(dataset)
    
    # Combine datasets from all directories into single datasets
    train_dataset = SupervisedDataset.merge_datasets(train_datasets) if train_datasets else None
    valid_dataset = SupervisedDataset.merge_datasets(valid_datasets) if valid_datasets else None
    
    # If max_eval_samples is set, randomly sample from validation dataset
    if valid_dataset and max_eval_samples and max_eval_samples < len(valid_dataset):
        # Use torch.randperm to randomly select indices
        indices = torch.randperm(len(valid_dataset))[:max_eval_samples].tolist()
        # Create new dataset with subset of samples
        valid_dataset.list_data_dict = [valid_dataset.list_data_dict[i] for i in indices]
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )
