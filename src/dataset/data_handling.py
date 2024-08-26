from dataset.data_classes import SupervisedDataset_train, SupervisedDatasetimage_valid, SupervisedDatasetimage_train, SupervisedDataset_valid, SupervisedDatasetimage_valid, DataCollatorForSupervisedDataset, DataCollatorForSupervisedDataset_ecg_text, DataCollatorForSupervisedDatasetimage, SupervisedDataset_test, SupervisedDatasetimage_test
from dataset.data_utils import get_preprocess_func, preprocess, preprocess_llama_2, preprocess_llama_3


def create_dataset(tokenizer, tokenizer2, image_processor, data_path, image_folder, image_aspect_ratio, is_multimodal, config, ecg_dataset, ecg_text, encoding, data_type):
    """Make dataset and collator for supervised fine-tuning."""
    preprocess_func = get_preprocess_func(config.text_model_id)
    if encoding != "vision_encoder":
        train_dataset = SupervisedDataset_train(tokenizer=tokenizer,
                                                tokenizer2=tokenizer2,
                                                image_processor=image_processor,
                                                data_path=data_path,
                                                image_folder=image_folder,
                                                image_aspect_ratio=image_aspect_ratio,
                                                is_multimodal=is_multimodal,
                                                preprocess_func=preprocess_func,
                                                ecg_dataset=ecg_dataset,
                                                ecg_text=ecg_text,
                                                encoding=encoding,
                                                data_type=data_type)

        eval_dataset = SupervisedDataset_valid(tokenizer=tokenizer,
                                                tokenizer2=tokenizer2,
                                                image_processor=image_processor,
                                                data_path=data_path,
                                                image_folder=image_folder,
                                                image_aspect_ratio=image_aspect_ratio,
                                                is_multimodal=is_multimodal,
                                                preprocess_func=preprocess_func,
                                                ecg_dataset=ecg_dataset,
                                                ecg_text=ecg_text,
                                                encoding=encoding,
                                                data_type=data_type)

        test_dataset = SupervisedDataset_test(tokenizer=tokenizer,
                                                tokenizer2=tokenizer2,
                                                image_processor=image_processor,
                                                data_path=data_path,
                                                image_folder=image_folder,
                                                image_aspect_ratio=image_aspect_ratio,
                                                is_multimodal=is_multimodal,
                                                preprocess_func=preprocess_func,
                                                ecg_dataset=ecg_dataset,
                                                ecg_text=ecg_text,
                                                encoding=encoding,
                                                data_type=data_type)

        if not ecg_text: 
            data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        else: 
            data_collator = DataCollatorForSupervisedDataset_ecg_text(tokenizer=tokenizer, tokenizer2=tokenizer2) 
    else:
        train_dataset = SupervisedDatasetimage_train(tokenizer=tokenizer,
                                                image_processor=image_processor,
                                                data_path=data_path,
                                                image_folder=image_folder,
                                                image_aspect_ratio=image_aspect_ratio,
                                                is_multimodal=is_multimodal,
                                                preprocess_func=preprocess_func,
                                                ecg_dataset=ecg_dataset,
                                                ecg_text=ecg_text,
                                                encoding=encoding,
                                                data_type=data_type)    
        
        eval_dataset = SupervisedDatasetimage_valid(tokenizer=tokenizer,
                                                image_processor=image_processor,
                                                data_path=data_path,
                                                image_folder=image_folder,
                                                image_aspect_ratio=image_aspect_ratio,
                                                is_multimodal=is_multimodal,
                                                preprocess_func=preprocess_func,
                                                ecg_dataset=ecg_dataset,
                                                ecg_text=ecg_text,
                                                encoding=encoding,
                                                data_type=data_type) 
        
        test_dataset = SupervisedDatasetimage_test(tokenizer=tokenizer,
                                                image_processor=image_processor,
                                                data_path=data_path,
                                                image_folder=image_folder,
                                                image_aspect_ratio=image_aspect_ratio,
                                                is_multimodal=is_multimodal,
                                                preprocess_func=preprocess_func,
                                                ecg_dataset=ecg_dataset,
                                                ecg_text=ecg_text,
                                                encoding=encoding,
                                                data_type=data_type) 
           
        data_collator = DataCollatorForSupervisedDatasetimage(tokenizer=tokenizer) 

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
