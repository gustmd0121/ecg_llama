import copy
import json
import os
from dataclasses import dataclass
import torch
from torchvision import transforms 
import transformers
from PIL import Image
from torch.utils.data import Dataset
from utils.constants import IMAGE_TOKEN, PAD_TOKEN
from utils.constants import IGNORE_INDEX
from dataset.data_utils import preprocess_multimodal
import wfdb

def get_actual_path(ecg_path):
    base_dir = '/nfs_edlab/hschung/ecg_plots'
    if 'records500' in ecg_path:
        sub_dir = 'records500'
    elif 'files' in ecg_path:
        sub_dir = 'files'
    else:
        raise ValueError(f"Unexpected path format: {ecg_path}")
    
    sub_path = f'/{sub_dir}' + ecg_path.split(f'/{sub_dir}', 1)[1]
    return f"{base_dir}{sub_path}/{ecg_path.split('/')[-1]}.jpg"

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 image_folder: str,
                 image_processor,
                 image_aspect_ratio,
                 tokenizer,
                 preprocess_func,
                 is_multimodal: bool = True):
        super(SupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.preprocess_func = preprocess_func
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
        self.is_multimodal = is_multimodal
        self.data_path = data_path
        self.image_folder = image_folder
        self.list_data_dict = list_data_dict

    @classmethod
    def merge_datasets(cls, datasets):
        """Merge multiple SupervisedDataset instances into one."""
        if not datasets:
            return None
        
        # Use the first dataset as base
        merged = copy.deepcopy(datasets[0])
        
        # Extend the list_data_dict with data from other datasets
        for dataset in datasets[1:]:
            merged.list_data_dict.extend(dataset.list_data_dict)
        
        return merged

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"
        
        data_dict = {}
        
        if 'image' in sources[0]:
            image_files = self.list_data_dict[i]['image']
            processor = self.image_processor
            
            # Process first image
            image = Image.open(image_files[0]).convert('RGB')
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            data_dict['image'] = image
            
            if len(image_files) == 2:
                # Process second image
                image2 = Image.open(image_files[1]).convert('RGB')
                image2 = processor.preprocess(image2, return_tensors='pt')['pixel_values'][0]
                data_dict['image2'] = image2
            else:
                image2 = None
                data_dict['image2'] = image2

            # Modify conversation data based on number of images
            conversations = copy.deepcopy([e["conversations"] for e in sources])
            for conv in conversations:
                # Find first human message
                for turn in conv:
                    if turn['from'] == 'human':
                        # Add appropriate number of image tokens
                        num_tokens = 2 if len(image_files) == 2 else 1
                        turn['value'] = (IMAGE_TOKEN * num_tokens) + '\n' + turn['value'].strip()
                        break  # Only modify first human message

            sources = preprocess_multimodal(conversations,
                                            is_multimodal=self.is_multimodal)
            
        elif 'ecg' in sources[0]:
            #ecg
            ecg_files = self.list_data_dict[i]['ecg']
            ecg, _ = wfdb.rdsamp(ecg_files[0])
            ecg = torch.from_numpy(ecg.T) 
            data_dict["ecg"] = ecg 
              
                
            if len(ecg_files) == 2: 
                ecg2, _ = wfdb.rdsamp(ecg_files[1])
                ecg2 = torch.from_numpy(ecg2.T)   
                data_dict["ecg2"] = ecg2          
            else:
                ecg2 = None
                data_dict["ecg2"] = ecg2
            
            conversations = copy.deepcopy([e["conversations"] for e in sources])
            for conv in conversations:
                # Find first human message
                for turn in conv:
                    if turn['from'] == 'human':
                        # Add appropriate number of image tokens
                        num_tokens = 2 if len(ecg_files) == 2 else 1
                        turn['value'] = (IMAGE_TOKEN * num_tokens) + '\n' + turn['value'].strip()    
                        break
            sources = preprocess_multimodal(conversations,
                                            is_multimodal=self.is_multimodal)

        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # Get processed text data
        text_data = self.preprocess_func(
            sources,
            self.tokenizer,
            has_image=('image' or 'ecg' in self.list_data_dict[i]))

        if isinstance(i, int):
            # Update data_dict with processed text data while preserving image information
            if 'image' in data_dict:
                data_dict.update({
                    'input_ids': text_data["input_ids"][0],
                    'labels': text_data["labels"][0],
                    'ecg': None,
                    'ecg2': None
                })
            elif 'ecg' in data_dict:
                data_dict.update({
                    'input_ids': text_data["input_ids"][0],
                    'labels': text_data["labels"][0],
                    'image': None,
                    'image2': None
                })
        else:
            data_dict.update(text_data)

        # Handle case where no image exists but model is multimodal
        if 'image' and 'ecg' not in data_dict and self.is_multimodal:
            crop_size = self.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['image2'] = None

        data_dict["conversation"] = sources[0]

        return data_dict, data_dict['image2']


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels, image, image2, ecg, ecg2 = tuple([instance[0][key] for instance in instances]
                                  for key in ("input_ids", "labels", "image", "image2", "ecg", "ecg2"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if ecg is None or any(e is None for e in ecg): 
            batch['pixel_values'] = torch.stack(image)
            processed_image2 = [i if i is not None else torch.zeros((3, 224, 224)) for i in image2]
            batch['images2'] = torch.stack(processed_image2)
            batch['ecg'] = None
            batch['ecg2'] = None
            
        else:  
            batch['ecg'] = torch.stack(ecg)      
            processed_ecg2 = [i if i is not None else torch.zeros((12, 5000)) for i in ecg2]
            batch['ecg2'] = torch.stack(processed_ecg2)
            batch['pixel_values'] = None
            batch['images2'] = None
            
            #ecg padding mask 
            # Check if each lead is all zeros [batch_size, 12]
            is_zero_sample1 = (batch['ecg'].sum(dim=(1, 2)) == 0)
            is_zero_sample2 = (batch['ecg2'].sum(dim=(1, 2)) == 0)

            # Expand lead masks to full shape [batch_size, 12, 5000]
            ecg_padding_mask = is_zero_sample1.unsqueeze(1).unsqueeze(2).expand(batch['ecg'].size()).clone()
            ecg_padding_mask2 = is_zero_sample2.unsqueeze(1).unsqueeze(2).expand(batch['ecg2'].size()).clone()
            
            batch["ecg_padding_mask"] = ecg_padding_mask
            batch["ecg_padding_mask2"] = ecg_padding_mask2
            
        
            
        return batch