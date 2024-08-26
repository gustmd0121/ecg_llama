import copy
import json
import os
from dataclasses import dataclass
import torch
import transformers
from transformers import BertTokenizerFast
from torchvision import transforms 
from PIL import Image
from torch.utils.data import Dataset
import orjson
from utils.constants import IGNORE_INDEX
from dataset.data_utils import preprocess_multimodal
import matplotlib.pyplot as plt 
from scipy import signal
import scipy.io
import wfdb 
import numpy as np 
import pdb
import pywt 


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

#image-text inputs 
class SupervisedDatasetimage_train(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 image_folder: str,
                 image_processor,
                 image_aspect_ratio,
                 tokenizer,
                 preprocess_func,
                 ecg_dataset,
                 ecg_text,
                 encoding,
                 data_type,
                 is_multimodal: bool = True):
        super(SupervisedDatasetimage_train, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.preprocess_func = preprocess_func
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
        self.is_multimodal = is_multimodal
        self.data_path = data_path
        self.image_folder = image_folder
        self.list_data_dict = list_data_dict
        self.ecg_dataset = ecg_dataset
        self.ecg_text = ecg_text
        self.encoding = encoding
        self.data_type = data_type

        self.patch_resize_transform = transforms.Compose([
        transforms.Resize(256),  # Resize the shorter side to 256 pixels
        transforms.CenterCrop(224),  # Crop the center 224x224 pixels
        transforms.ToTensor(),  # Convert to tensor
        ])

        #ecg dataset 
        if self.ecg_dataset == 'all':
            self.base_path = ["/nfs_edlab/hschung/output/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/template/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/paraphrased/"]
        elif self.ecg_dataset == 'ptbxl':
            self.base_path = ["/nfs_edlab/hschung/output/"]
        else:
            self.base_path = ["/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/template/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/paraphrased/"]  

        self.dataset2 = {'train': []}

        for base in self.base_path:
            for dataset_type in ['train']:
                dataset_path = os.path.join(base, dataset_type)
                for file_name in os.listdir(dataset_path):
                    if file_name.endswith('.json'):
                        full_path = os.path.join(dataset_path, file_name)
                        with open(full_path, 'r') as file:
                            data = orjson.loads(file.read())
                            # Assuming each item in data has an 'ecg_id' key
                            for item in data:
                                self.dataset2[dataset_type].append(item)  

    def __len__(self):
        return len(self.dataset2['train'])

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
        #input_ids, labels, ecg 
        source2_original = self.dataset2['train'][i]
        ecg_paths = source2_original['ecg_path']               
        
        #process QA
        source2 = [
            {
                'from': 'human',
                'value': f'<image>\n{source2_original["question"]}'
            },
            {
                'from': 'gpt',
                'value': ', '.join(source2_original['answer'])
            }
        ]
        
        source2 = preprocess_multimodal(copy.deepcopy([source2]),
                                        is_multimodal=self.is_multimodal)           

        data_dict2 = self.preprocess_func(
            source2,
            self.tokenizer,
            has_image=True
            )

        data_dict2 = dict(input_ids=data_dict2["input_ids"][0],
                            labels=data_dict2["labels"][0])  
        data_dict2["ecg2"] = None 
        if self.data_type == 'raw_signal':
            #ecg
            ecg, _ = wfdb.rdsamp(ecg_paths[0])
            ecg = torch.from_numpy(ecg.T) 
            data_dict2["ecg"] = ecg 
              
                
            if len(ecg_paths) == 2: 
                ecg2, _ = wfdb.rdsamp(ecg_paths[1])
                ecg2 = torch.from_numpy(ecg2.T)   
                data_dict2["ecg2"] = ecg2
                
        elif self.data_type == 'signal_image':
            actual_path = get_actual_path(ecg_paths[0])
            with Image.open(actual_path) as img:
                image_spectrogram = img.convert('RGB')
                data_dict2['ecg'] = self.patch_resize_transform(image_spectrogram)
            if len(ecg_paths) == 2:
                actual_path2 = get_actual_path(ecg_paths[1])
                with Image.open(actual_path2) as img:
                    image_spectrogram2 = img.convert('RGB')
                    data_dict2['ecg2'] = self.patch_resize_transform(image_spectrogram2)
        elif self.data_type == 'spectrogram_image':
                # Compute spectrogram for each channel
                ecg, _ = wfdb.rdsamp(ecg_paths[0])
                ecg = torch.from_numpy(ecg.T) 
                spectrograms = []
                for channel in ecg:
                    f, t, Sxx = signal.spectrogram(channel, fs=500, nperseg=256, noverlap=128)
                    Sxx = np.log1p(Sxx) #logscale
                    Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min()) #normalize the spectrogram
                    spectrograms.append(Sxx)

                # Concatenate spectrograms along the time axis
                Sxx_concatenated = np.concatenate(spectrograms, axis=-1)

                cm = plt.get_cmap('viridis')
                spectrogram_colormap = cm(Sxx_concatenated)  # Apply colormap
                spectrogram_image = (spectrogram_colormap[..., :3] * 255).astype(np.uint8)  # Convert to RGB image
            
                image_spectrogram = self.patch_resize_transform(Image.fromarray(spectrogram_image))
                data_dict2['ecg'] = image_spectrogram
                if len(ecg_paths) == 2:
                    ecg2, _ = wfdb.rdsamp(ecg_paths[1])
                    ecg2 = torch.from_numpy(ecg2.T)       
                    spectrograms2 = []
                    for channel in ecg:
                        f, t, Sxx2 = signal.spectrogram(channel, fs=500)
                        Sxx2 = np.log1p(Sxx2) #logscale
                        Sxx2 = (Sxx2 - Sxx2.min()) / (Sxx2.max() - Sxx2.min()) #normalize the spectrogram
                        spectrograms2.append(Sxx2)

                    # Concatenate spectrograms along the time axis
                    Sxx2_concatenated = np.concatenate(spectrograms2, axis=-1)

                    cm = plt.get_cmap('viridis')
                    spectrogram2_colormap = cm(Sxx2_concatenated)  # Apply colormap
                    spectrogram2_image = (spectrogram2_colormap[..., :3] * 255).astype(np.uint8)  # Convert to RGB image
                
                    image_spectrogram2 = self.patch_resize_transform(Image.fromarray(spectrogram2_image))   
                    data_dict2['ecg'] = image_spectrogram2             
        
        elif self.data_type == 'wavelet_image':
            def process_ecg(ecg_path):
                ecg, fields = wfdb.rdsamp(ecg_path)
                fs = fields['fs']
                ecg = ecg.T
                
                scales = np.arange(1, 128)
                wavelet = 'morl'
                
                lead_scalograms = []
                for lead in ecg:
                    coefficients, _ = pywt.cwt(lead, scales, wavelet, 1.0 / fs)
                    scalogram = np.abs(coefficients)
                    lead_scalograms.append(scalogram)
                
                # Normalize across all scalograms
                all_scalograms = np.array(lead_scalograms)
                min_val = np.min(all_scalograms)
                max_val = np.max(all_scalograms)
                epsilon = 1e-8
                if max_val - min_val > epsilon:
                    normalized_scalograms = (all_scalograms - min_val) / (max_val - min_val)
                else:
                    normalized_scalograms = np.zeros_like(all_scalograms)
                
                # Create a composite image of all lead scalograms
                num_leads = len(normalized_scalograms)
                grid_size = int(np.ceil(np.sqrt(num_leads)))
                composite_image = np.zeros((grid_size * 128, grid_size * normalized_scalograms[0].shape[1], 3))
                
                for idx, scalogram in enumerate(normalized_scalograms):
                    i, j = divmod(idx, grid_size)
                    rgb_scalogram = plt.get_cmap('viridis')(scalogram)[:, :, :3]
                    
                    # Ensure the scalogram fits into the 128xN patch
                    padded_scalogram = np.zeros((128, scalogram.shape[1], 3))
                    padded_scalogram[:scalogram.shape[0], :scalogram.shape[1], :3] = rgb_scalogram
                    
                    composite_image[i*128:(i+1)*128, j*scalogram.shape[1]:(j+1)*scalogram.shape[1]] = padded_scalogram
                
                img_pil = Image.fromarray(np.uint8(composite_image * 255))
                img_tensor = self.patch_resize_transform(img_pil)
                return img_tensor

            data_dict2['ecg'] = process_ecg(ecg_paths[0])

            if len(ecg_paths) == 2:
                data_dict2['ecg2'] = process_ecg(ecg_paths[1])
            else:
                data_dict2['ecg2'] = torch.zeros(3, 224, 224)
        
        return data_dict2, data_dict2['ecg']


@dataclass
class DataCollatorForSupervisedDatasetimage(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels, ecg, ecg2 = tuple([instance[0][key] for instance in instances]
                                        for key in ("input_ids", "labels", "ecg", "ecg2"))
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

        ecgs = torch.stack([instance[0]['ecg'] for instance in instances])
        ecgs2 = torch.stack([instance[0]['ecg2'] if 'ecg2' in instance[0] and instance[0]['ecg2'] is not None else torch.zeros(3, 224, 224) for instance in instances])
        ecg_padding_mask = torch.zeros_like(ecgs, dtype=torch.bool)
        ecg2_padding_mask = torch.tensor([instance[0]['ecg2'] is None for instance in instances], dtype=torch.bool).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 3, 224, 224).clone()
        batch['pixel_values'] = ecgs
        batch['ecg_values2'] = ecgs2
        batch['ecg_padding_mask'] = ecg_padding_mask
        batch['ecg2_padding_mask'] = ecg2_padding_mask
        return batch

#ecg-text 
class SupervisedDataset_train(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 image_folder: str,
                 image_processor,
                 image_aspect_ratio,
                 tokenizer,
                 tokenizer2,
                 preprocess_func,
                 ecg_dataset,
                 ecg_text,
                 encoding,
                 data_type,
                 is_multimodal: bool = True):
        super(SupervisedDataset_train, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2
        self.preprocess_func = preprocess_func
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
        self.is_multimodal = is_multimodal
        self.data_path = data_path
        self.image_folder = image_folder
        self.list_data_dict = list_data_dict
        self.ecg_dataset = ecg_dataset
        self.ecg_text = ecg_text
        self.encoding = encoding
        self.data_type = data_type
        
        #ecg dataset 
        if self.ecg_dataset == 'all':
            self.base_path = ["/nfs_edlab/hschung/output/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/template/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/paraphrased/"]
        elif self.ecg_dataset == 'ptbxl':
            self.base_path = ["/nfs_edlab/hschung/output/"]
        else:
            self.base_path = ["/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/template/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/paraphrased/"]  

        self.dataset2 = {'train': []}

        for base in self.base_path:
            for dataset_type in ['train']:
                dataset_path = os.path.join(base, dataset_type)
                for file_name in os.listdir(dataset_path):
                    if file_name.endswith('.json'):
                        full_path = os.path.join(dataset_path, file_name)
                        with open(full_path, 'r') as file:
                            data = orjson.loads(file.read())
                            # Assuming each item in data has an 'ecg_id' key
                            for item in data:
                                self.dataset2[dataset_type].append(item)  

    def __len__(self):
        return len(self.dataset2['train'])


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
        #input_ids, labels, ecg 
        source2_original = self.dataset2['train'][i]
        ecg_paths = source2_original['ecg_path']               
        
        #process QA
        source2 = [
            {
                'from': 'human',
                'value': f'<image>\n{source2_original["question"]}'
            },
            {
                'from': 'gpt',
                'value': ', '.join(source2_original['answer'])
            }
        ]
        
        source2 = preprocess_multimodal(copy.deepcopy([source2]),
                                        is_multimodal=self.is_multimodal)           

        data_dict2 = self.preprocess_func(
            source2,
            self.tokenizer,
            has_image=True
            )

        data_dict2 = dict(input_ids=data_dict2["input_ids"][0],
                            labels=data_dict2["labels"][0])  

        if self.ecg_text:
            question = source2_original["question"]

            # Tokenize the question and answer using the BERT tokenizer
            question_tokens = self.tokenizer2.encode(question, add_special_tokens=True)

            # Concatenate question and answer tokens
            input_ids_m3ae = question_tokens
            input_ids_m3ae = input_ids_m3ae[:self.tokenizer.model_max_length]

            data_dict2["input_ids_m3ae"] = torch.tensor(input_ids_m3ae, dtype=torch.long)

        #ecg
        ecg, _ = wfdb.rdsamp(ecg_paths[0])
        ecg = torch.from_numpy(ecg.T) 
        data_dict2["ecg"] = ecg 
        data_dict2["ecg2"] = None   
            
        if len(ecg_paths) == 2: 
            ecg2, _ = wfdb.rdsamp(ecg_paths[1])
            ecg2 = torch.from_numpy(ecg2.T)   
            data_dict2["ecg2"] = ecg2           
        
        return data_dict2, data_dict2['ecg']

class SupervisedDataset_valid(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 image_folder: str,
                 image_processor,
                 image_aspect_ratio,
                 tokenizer,
                 tokenizer2,
                 preprocess_func,
                 ecg_dataset,
                 ecg_text,
                 encoding,
                 data_type,
                 is_multimodal: bool = True):
        super(SupervisedDataset_valid, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2
        self.preprocess_func = preprocess_func
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
        self.is_multimodal = is_multimodal
        self.data_path = data_path
        self.image_folder = image_folder
        self.list_data_dict = list_data_dict
        self.ecg_dataset = ecg_dataset
        self.ecg_text = ecg_text
        self.encoding = encoding
        self.data_type = data_type
        
        #ecg dataset 
        if self.ecg_dataset == 'all':
            self.base_path = ["/nfs_edlab/hschung/output/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/template/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/paraphrased/"]
        elif self.ecg_dataset == 'ptbxl':
            self.base_path = ["/nfs_edlab/hschung/output/"]
        else:
            self.base_path = ["/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/template/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/paraphrased/"]  

        self.dataset2 = {'valid': []}

        for base in self.base_path:
            for dataset_type in ['valid']:
                dataset_path = os.path.join(base, dataset_type)
                for file_name in os.listdir(dataset_path):
                    if file_name.endswith('.json'):
                        full_path = os.path.join(dataset_path, file_name)
                        with open(full_path, 'r') as file:
                            data = orjson.loads(file.read())
                            # Assuming each item in data has an 'ecg_id' key
                            for item in data:
                                self.dataset2[dataset_type].append(item)  
                                if len(self.dataset2[dataset_type]) >= 1000:
                                    break  # Break out of the innermost loop
                            else:
                                continue  # Continue if the inner loop wasn't broken
                            break  # Break out of the file_name loop
                    else:
                        continue  # Continue if the file doesn't end with .json
                    break  # Break out of the dataset_type loop
                else:
                    continue  # Continue if the dataset_type loop wasn't broken
                break  # Break out of the base loop
    def __len__(self):
        return len(self.dataset2['valid'])


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
        #input_ids, labels, ecg 
        source2_original = self.dataset2['valid'][i]
        ecg_paths = source2_original['ecg_path']               
        
        #process QA
        source2 = [
            {
                'from': 'human',
                'value': f'<image>\n{source2_original["question"]}'
            },
            {
                'from': 'gpt',
                'value': ', '.join(source2_original['answer'])
            }
        ]
        
        source2 = preprocess_multimodal(copy.deepcopy([source2]),
                                        is_multimodal=self.is_multimodal)           

        data_dict2 = self.preprocess_func(
            source2,
            self.tokenizer,
            has_image=True
            )

        data_dict2 = dict(input_ids=data_dict2["input_ids"][0],
                            labels=data_dict2["labels"][0])  

        if self.ecg_text:
            question = source2_original["question"]

            # Tokenize the question and answer using the BERT tokenizer
            question_tokens = self.tokenizer2.encode(question, add_special_tokens=True)

            # Concatenate question and answer tokens
            input_ids_m3ae = question_tokens
            input_ids_m3ae = input_ids_m3ae[:self.tokenizer.model_max_length]

            data_dict2["input_ids_m3ae"] = torch.tensor(input_ids_m3ae, dtype=torch.long)

        #ecg
        ecg, _ = wfdb.rdsamp(ecg_paths[0])
        ecg = torch.from_numpy(ecg.T) 
        data_dict2["ecg"] = ecg 
        data_dict2["ecg2"] = None   
            
        if len(ecg_paths) == 2: 
            ecg2, _ = wfdb.rdsamp(ecg_paths[1])
            ecg2 = torch.from_numpy(ecg2.T)   
            data_dict2["ecg2"] = ecg2           
        
        return data_dict2, data_dict2['ecg']

class SupervisedDataset_test(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 image_folder: str,
                 image_processor,
                 image_aspect_ratio,
                 tokenizer,
                 tokenizer2,
                 preprocess_func,
                 ecg_dataset,
                 ecg_text,
                 encoding,
                 data_type,
                 is_multimodal: bool = True):
        super(SupervisedDataset_test, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2
        self.preprocess_func = preprocess_func
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
        self.is_multimodal = is_multimodal
        self.data_path = data_path
        self.image_folder = image_folder
        self.list_data_dict = list_data_dict
        self.ecg_dataset = ecg_dataset
        self.ecg_text = ecg_text
        self.encoding = encoding
        self.data_type = data_type
        
        #ecg dataset 
        if self.ecg_dataset == 'all':
            self.base_path = ["/nfs_edlab/hschung/output/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/template/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/paraphrased/"]
        elif self.ecg_dataset == 'ptbxl':
            self.base_path = ["/nfs_edlab/hschung/output/"]
        else:
            self.base_path = ["/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/template/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/paraphrased/"]  

        self.dataset2 = {'test': []}

        for base in self.base_path:
            for dataset_type in ['test']:
                dataset_path = os.path.join(base, dataset_type)
                for file_name in os.listdir(dataset_path):
                    if file_name.endswith('.json'):
                        full_path = os.path.join(dataset_path, file_name)
                        with open(full_path, 'r') as file:
                            data = orjson.loads(file.read())
                            # Assuming each item in data has an 'ecg_id' key
                            for item in data:
                                self.dataset2[dataset_type].append(item)  
                                if len(self.dataset2[dataset_type]) >= 1000:
                                    break  # Break out of the innermost loop
                            else:
                                continue  # Continue if the inner loop wasn't broken
                            break  # Break out of the file_name loop
                    else:
                        continue  # Continue if the file doesn't end with .json
                    break  # Break out of the dataset_type loop
                else:
                    continue  # Continue if the dataset_type loop wasn't broken
                break  # Break out of the base loop
    def __len__(self):
        return len(self.dataset2['test'])


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
        #input_ids, labels, ecg 
        source2_original = self.dataset2['test'][i]
        ecg_paths = source2_original['ecg_path']               
        
        #process QA
        source2 = [
            {
                'from': 'human',
                'value': f'<image>\n{source2_original["question"]}'
            },
            {
                'from': 'gpt',
                'value': ', '.join(source2_original['answer'])
            }
        ]
        
        source2 = preprocess_multimodal(copy.deepcopy([source2]),
                                        is_multimodal=self.is_multimodal)           

        data_dict2 = self.preprocess_func(
            source2,
            self.tokenizer,
            has_image=True
            )

        data_dict2 = dict(input_ids=data_dict2["input_ids"][0],
                            labels=data_dict2["labels"][0])  

        if self.ecg_text:
            question = source2_original["question"]

            # Tokenize the question and answer using the BERT tokenizer
            question_tokens = self.tokenizer2.encode(question, add_special_tokens=True)

            # Concatenate question and answer tokens
            input_ids_m3ae = question_tokens
            input_ids_m3ae = input_ids_m3ae[:self.tokenizer.model_max_length]

            data_dict2["input_ids_m3ae"] = torch.tensor(input_ids_m3ae, dtype=torch.long)

        #ecg
        ecg, _ = wfdb.rdsamp(ecg_paths[0])
        ecg = torch.from_numpy(ecg.T) 
        data_dict2["ecg"] = ecg 
        data_dict2["ecg2"] = None   
            
        if len(ecg_paths) == 2: 
            ecg2, _ = wfdb.rdsamp(ecg_paths[1])
            ecg2 = torch.from_numpy(ecg2.T)   
            data_dict2["ecg2"] = ecg2           
        
        return data_dict2, data_dict2['ecg']

class SupervisedDatasetimage_valid(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 image_folder: str,
                 image_processor,
                 image_aspect_ratio,
                 tokenizer,
                 preprocess_func,
                 ecg_dataset,
                 ecg_text,
                 encoding,
                 data_type,
                 is_multimodal: bool = True):
        super(SupervisedDatasetimage_valid, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.preprocess_func = preprocess_func
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
        self.is_multimodal = is_multimodal
        self.data_path = data_path
        self.image_folder = image_folder
        self.list_data_dict = list_data_dict
        self.ecg_dataset = ecg_dataset
        self.ecg_text = ecg_text
        self.encoding = encoding
        self.data_type = data_type

        self.patch_resize_transform = transforms.Compose([
        transforms.Resize(256),  # Resize the shorter side to 256 pixels
        transforms.CenterCrop(224),  # Crop the center 224x224 pixels
        transforms.ToTensor()
        ])

        #ecg dataset 
        if self.ecg_dataset == 'all':
            self.base_path = ["/nfs_edlab/hschung/output/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/template/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/paraphrased/"]
        elif self.ecg_dataset == 'ptbxl':
            self.base_path = ["/nfs_edlab/hschung/output/"]
        else:
            self.base_path = ["/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/template/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/paraphrased/"]  

        self.dataset2 = {'valid': []}

        for base in self.base_path:
            for dataset_type in ['valid']:
                dataset_path = os.path.join(base, dataset_type)
                for file_name in os.listdir(dataset_path):
                    if file_name.endswith('.json'):
                        full_path = os.path.join(dataset_path, file_name)
                        with open(full_path, 'r') as file:
                            data = orjson.loads(file.read())
                            for item in data:
                                self.dataset2[dataset_type].append(item)
                                if len(self.dataset2[dataset_type]) >= 1000:
                                    break  # Break out of the innermost loop
                            else:
                                continue  # Continue if the inner loop wasn't broken
                            break  # Break out of the file_name loop
                    else:
                        continue  # Continue if the file doesn't end with .json
                    break  # Break out of the dataset_type loop
                else:
                    continue  # Continue if the dataset_type loop wasn't broken
                break  # Break out of the base loop

        print(f"Total items in valid dataset: {len(self.dataset2['valid'])}")

    def __len__(self):
        return len(self.dataset2['valid'])

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
        #input_ids, labels, ecg 
        source2_original = self.dataset2['valid'][i]
        ecg_paths = source2_original['ecg_path']               
        
        #process QA
        source2 = [
            {
                'from': 'human',
                'value': f'<image>\n{source2_original["question"]}'
            },
            {
                'from': 'gpt',
                'value': ', '.join(source2_original['answer'])
            }
        ]
        
        source2 = preprocess_multimodal(copy.deepcopy([source2]),
                                        is_multimodal=self.is_multimodal)           

        data_dict2 = self.preprocess_func(
            source2,
            self.tokenizer,
            has_image=True
            )

        data_dict2 = dict(input_ids=data_dict2["input_ids"][0],
                            labels=data_dict2["labels"][0])  
        data_dict2["ecg2"] = None 
        if self.data_type == 'raw_signal':
            #ecg
            ecg, _ = wfdb.rdsamp(ecg_paths[0])
            ecg = torch.from_numpy(ecg.T) 
            data_dict2["ecg"] = ecg 
              
                
            if len(ecg_paths) == 2: 
                ecg2, _ = wfdb.rdsamp(ecg_paths[1])
                ecg2 = torch.from_numpy(ecg2.T)   
                data_dict2["ecg2"] = ecg2
                
        elif self.data_type == 'signal_image':
            actual_path = get_actual_path(ecg_paths[0])
            with Image.open(actual_path) as img:
                image_spectrogram = img.convert('RGB')
                data_dict2['ecg'] = self.patch_resize_transform(image_spectrogram)
            if len(ecg_paths) == 2:
                actual_path2 = get_actual_path(ecg_paths[1])
                with Image.open(actual_path2) as img:
                    image_spectrogram2 = img.convert('RGB')
                    data_dict2['ecg2'] = self.patch_resize_transform(image_spectrogram2)

        elif self.data_type == 'spectrogram_image':
                # Compute spectrogram for each channel
                ecg, _ = wfdb.rdsamp(ecg_paths[0])
                ecg = torch.from_numpy(ecg.T) 
                spectrograms = []
                for channel in ecg:
                    f, t, Sxx = signal.spectrogram(channel, fs=500, nperseg=256, noverlap=128)
                    Sxx = np.log1p(Sxx) #logscale
                    Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min()) #normalize the spectrogram
                    spectrograms.append(Sxx)

                # Concatenate spectrograms along the time axis
                Sxx_concatenated = np.concatenate(spectrograms, axis=-1)

                cm = plt.get_cmap('viridis')
                spectrogram_colormap = cm(Sxx_concatenated)  # Apply colormap
                spectrogram_image = (spectrogram_colormap[..., :3] * 255).astype(np.uint8)  # Convert to RGB image
            
                image_spectrogram = self.patch_resize_transform(Image.fromarray(spectrogram_image))
                data_dict2['ecg'] = image_spectrogram
                if len(ecg_paths) == 2:
                    ecg2, _ = wfdb.rdsamp(ecg_paths[1])
                    ecg2 = torch.from_numpy(ecg2.T)       
                    spectrograms2 = []
                    for channel in ecg:
                        f, t, Sxx2 = signal.spectrogram(channel, fs=500)
                        Sxx2 = np.log1p(Sxx2) #logscale
                        Sxx2 = (Sxx2 - Sxx2.min()) / (Sxx2.max() - Sxx2.min()) #normalize the spectrogram
                        spectrograms2.append(Sxx2)

                    # Concatenate spectrograms along the time axis
                    Sxx2_concatenated = np.concatenate(spectrograms2, axis=-1)

                    cm = plt.get_cmap('viridis')
                    spectrogram2_colormap = cm(Sxx2_concatenated)  # Apply colormap
                    spectrogram2_image = (spectrogram2_colormap[..., :3] * 255).astype(np.uint8)  # Convert to RGB image
                
                    image_spectrogram2 = self.patch_resize_transform(Image.fromarray(spectrogram2_image))   
                    data_dict2['ecg'] = image_spectrogram2             
        
        elif self.data_type == 'wavelet_image':
            def process_ecg(ecg_path):
                ecg, fields = wfdb.rdsamp(ecg_path)
                fs = fields['fs']
                ecg = ecg.T
                
                scales = np.arange(1, 128)
                wavelet = 'morl'
                
                lead_scalograms = []
                for lead in ecg:
                    coefficients, _ = pywt.cwt(lead, scales, wavelet, 1.0 / fs)
                    scalogram = np.abs(coefficients)
                    lead_scalograms.append(scalogram)
                
                # Normalize across all scalograms
                all_scalograms = np.array(lead_scalograms)
                min_val = np.min(all_scalograms)
                max_val = np.max(all_scalograms)
                epsilon = 1e-8
                if max_val - min_val > epsilon:
                    normalized_scalograms = (all_scalograms - min_val) / (max_val - min_val)
                else:
                    normalized_scalograms = np.zeros_like(all_scalograms)
                
                # Create a composite image of all lead scalograms
                num_leads = len(normalized_scalograms)
                grid_size = int(np.ceil(np.sqrt(num_leads)))
                composite_image = np.zeros((grid_size * 128, grid_size * normalized_scalograms[0].shape[1], 3))
                
                for idx, scalogram in enumerate(normalized_scalograms):
                    i, j = divmod(idx, grid_size)
                    rgb_scalogram = plt.get_cmap('viridis')(scalogram)[:, :, :3]
                    
                    # Ensure the scalogram fits into the 128xN patch
                    padded_scalogram = np.zeros((128, scalogram.shape[1], 3))
                    padded_scalogram[:scalogram.shape[0], :scalogram.shape[1], :3] = rgb_scalogram
                    
                    composite_image[i*128:(i+1)*128, j*scalogram.shape[1]:(j+1)*scalogram.shape[1]] = padded_scalogram
                
                img_pil = Image.fromarray(np.uint8(composite_image * 255))
                img_tensor = self.patch_resize_transform(img_pil)
                return img_tensor

            data_dict2['ecg'] = process_ecg(ecg_paths[0])

            if len(ecg_paths) == 2:
                data_dict2['ecg2'] = process_ecg(ecg_paths[1])
            else:
                data_dict2['ecg2'] = torch.zeros(3, 224, 224)
        
        return data_dict2, data_dict2['ecg']

class SupervisedDatasetimage_test(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 image_folder: str,
                 image_processor,
                 image_aspect_ratio,
                 tokenizer,
                 preprocess_func,
                 ecg_dataset,
                 ecg_text,
                 encoding,
                 data_type,
                 is_multimodal: bool = True):
        super(SupervisedDatasetimage_test, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.preprocess_func = preprocess_func
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
        self.is_multimodal = is_multimodal
        self.data_path = data_path
        self.image_folder = image_folder
        self.list_data_dict = list_data_dict
        self.ecg_dataset = ecg_dataset
        self.ecg_text = ecg_text
        self.encoding = encoding
        self.data_type = data_type

        self.patch_resize_transform = transforms.Compose([
        transforms.Resize(256),  # Resize the shorter side to 256 pixels
        transforms.CenterCrop(224),  # Crop the center 224x224 pixels
        transforms.ToTensor()
        ])

        #ecg dataset 
        if self.ecg_dataset == 'all':
            self.base_path = ["/nfs_edlab/hschung/output/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/template/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/paraphrased/"]
        elif self.ecg_dataset == 'ptbxl':
            self.base_path = ["/nfs_edlab/hschung/output/"]
        else:
            self.base_path = ["/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/template/", "/nfs_edlab/hschung/output_mimic/mimic-iv-ecg/paraphrased/"]  

        self.dataset2 = {'test': []}

        for base in self.base_path:
            for dataset_type in ['test']:
                dataset_path = os.path.join(base, dataset_type)
                for file_name in os.listdir(dataset_path):
                    if file_name.endswith('.json'):
                        full_path = os.path.join(dataset_path, file_name)
                        with open(full_path, 'r') as file:
                            data = orjson.loads(file.read())
                            for item in data:
                                self.dataset2[dataset_type].append(item)
                                if len(self.dataset2[dataset_type]) >= 1000:
                                    break  # Break out of the innermost loop
                            else:
                                continue  # Continue if the inner loop wasn't broken
                            break  # Break out of the file_name loop
                    else:
                        continue  # Continue if the file doesn't end with .json
                    break  # Break out of the dataset_type loop
                else:
                    continue  # Continue if the dataset_type loop wasn't broken
                break  # Break out of the base loop

        print(f"Total items in test dataset: {len(self.dataset2['test'])}")

    def __len__(self):
        return len(self.dataset2['test'])

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
        #input_ids, labels, ecg 
        source2_original = self.dataset2['test'][i]
        ecg_paths = source2_original['ecg_path']               
        
        #process QA
        source2 = [
            {
                'from': 'human',
                'value': f'<image>\n{source2_original["question"]}'
            },
            {
                'from': 'gpt',
                'value': ', '.join(source2_original['answer'])
            }
        ]
        
        source2 = preprocess_multimodal(copy.deepcopy([source2]),
                                        is_multimodal=self.is_multimodal)               

        data_dict2 = self.preprocess_func(
            source2,
            self.tokenizer,
            has_image=True
            )

        data_dict2 = dict(input_ids=data_dict2["input_ids"][0],
                            labels=data_dict2["labels"][0])  
        data_dict2["ecg2"] = None 
        if self.data_type == 'raw_signal':
            #ecg
            ecg, _ = wfdb.rdsamp(ecg_paths[0])
            ecg = torch.from_numpy(ecg.T) 
            data_dict2["ecg"] = ecg 
              
                
            if len(ecg_paths) == 2: 
                ecg2, _ = wfdb.rdsamp(ecg_paths[1])
                ecg2 = torch.from_numpy(ecg2.T)   
                data_dict2["ecg2"] = ecg2
                
        elif self.data_type == 'signal_image':
            actual_path = get_actual_path(ecg_paths[0])
            with Image.open(actual_path) as img:
                image_spectrogram = img.convert('RGB')
                data_dict2['ecg'] = self.patch_resize_transform(image_spectrogram)
            if len(ecg_paths) == 2:
                actual_path2 = get_actual_path(ecg_paths[1])
                with Image.open(actual_path2) as img:
                    image_spectrogram2 = img.convert('RGB')
                    data_dict2['ecg2'] = self.patch_resize_transform(image_spectrogram2)

        elif self.data_type == 'spectrogram_image':
                # Compute spectrogram for each channel
                ecg, _ = wfdb.rdsamp(ecg_paths[0])
                ecg = torch.from_numpy(ecg.T) 
                spectrograms = []
                for channel in ecg:
                    f, t, Sxx = signal.spectrogram(channel, fs=500, nperseg=256, noverlap=128)
                    Sxx = np.log1p(Sxx) #logscale
                    Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min()) #normalize the spectrogram
                    spectrograms.append(Sxx)

                # Concatenate spectrograms along the time axis
                Sxx_concatenated = np.concatenate(spectrograms, axis=-1)

                cm = plt.get_cmap('viridis')
                spectrogram_colormap = cm(Sxx_concatenated)  # Apply colormap
                spectrogram_image = (spectrogram_colormap[..., :3] * 255).astype(np.uint8)  # Convert to RGB image
            
                image_spectrogram = self.patch_resize_transform(Image.fromarray(spectrogram_image))
                data_dict2['ecg'] = image_spectrogram
                if len(ecg_paths) == 2:
                    ecg2, _ = wfdb.rdsamp(ecg_paths[1])
                    ecg2 = torch.from_numpy(ecg2.T)       
                    spectrograms2 = []
                    for channel in ecg:
                        f, t, Sxx2 = signal.spectrogram(channel, fs=500)
                        Sxx2 = np.log1p(Sxx2) #logscale
                        Sxx2 = (Sxx2 - Sxx2.min()) / (Sxx2.max() - Sxx2.min()) #normalize the spectrogram
                        spectrograms2.append(Sxx2)

                    # Concatenate spectrograms along the time axis
                    Sxx2_concatenated = np.concatenate(spectrograms2, axis=-1)

                    cm = plt.get_cmap('viridis')
                    spectrogram2_colormap = cm(Sxx2_concatenated)  # Apply colormap
                    spectrogram2_image = (spectrogram2_colormap[..., :3] * 255).astype(np.uint8)  # Convert to RGB image
                
                    image_spectrogram2 = self.patch_resize_transform(Image.fromarray(spectrogram2_image))   
                    data_dict2['ecg'] = image_spectrogram2             
        
        elif self.data_type == 'wavelet_image':
            def process_ecg(ecg_path):
                ecg, fields = wfdb.rdsamp(ecg_path)
                fs = fields['fs']
                ecg = ecg.T
                
                scales = np.arange(1, 128)
                wavelet = 'morl'
                
                lead_scalograms = []
                for lead in ecg:
                    coefficients, _ = pywt.cwt(lead, scales, wavelet, 1.0 / fs)
                    scalogram = np.abs(coefficients)
                    lead_scalograms.append(scalogram)
                
                # Normalize across all scalograms
                all_scalograms = np.array(lead_scalograms)
                min_val = np.min(all_scalograms)
                max_val = np.max(all_scalograms)
                epsilon = 1e-8
                if max_val - min_val > epsilon:
                    normalized_scalograms = (all_scalograms - min_val) / (max_val - min_val)
                else:
                    normalized_scalograms = np.zeros_like(all_scalograms)
                
                # Create a composite image of all lead scalograms
                num_leads = len(normalized_scalograms)
                grid_size = int(np.ceil(np.sqrt(num_leads)))
                composite_image = np.zeros((grid_size * 128, grid_size * normalized_scalograms[0].shape[1], 3))
                
                for idx, scalogram in enumerate(normalized_scalograms):
                    i, j = divmod(idx, grid_size)
                    rgb_scalogram = plt.get_cmap('viridis')(scalogram)[:, :, :3]
                    
                    # Ensure the scalogram fits into the 128xN patch
                    padded_scalogram = np.zeros((128, scalogram.shape[1], 3))
                    padded_scalogram[:scalogram.shape[0], :scalogram.shape[1], :3] = rgb_scalogram
                    
                    composite_image[i*128:(i+1)*128, j*scalogram.shape[1]:(j+1)*scalogram.shape[1]] = padded_scalogram
                
                img_pil = Image.fromarray(np.uint8(composite_image * 255))
                img_tensor = self.patch_resize_transform(img_pil)
                return img_tensor

            data_dict2['ecg'] = process_ecg(ecg_paths[0])

            if len(ecg_paths) == 2:
                data_dict2['ecg2'] = process_ecg(ecg_paths[1])
            else:
                data_dict2['ecg2'] = torch.zeros(3, 224, 224)
        
        return data_dict2, data_dict2['ecg']

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels, ecg, ecg2 = tuple([instance[0][key] for instance in instances]
                                        for key in ("input_ids", "labels", "ecg", "ecg2"))
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

        ecgs = torch.stack([instance[0]['ecg'] for instance in instances])
        ecgs2 = torch.stack([instance[0]['ecg2'] if 'ecg2' in instance[0] and instance[0]['ecg2'] is not None else torch.zeros(12, 5000) for instance in instances])
        ecg_padding_mask = torch.zeros_like(ecgs, dtype=torch.bool)
        ecg2_padding_mask = torch.tensor([ecg2 is None for ecg2 in ecg2], dtype=torch.bool).unsqueeze(1).unsqueeze(2).expand_as(ecgs2).clone()
        batch['pixel_values'] = ecgs
        batch['ecg_values2'] = ecgs2
        batch['ecg_padding_mask'] = ecg_padding_mask
        batch['ecg2_padding_mask'] = ecg2_padding_mask

        return batch

@dataclass
class DataCollatorForSupervisedDataset_ecg_text(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    tokenizer2: BertTokenizerFast

    def __call__(self, instances):
        input_ids, input_ids_m3ae, labels, ecg, ecg2 = tuple([instance[0][key] for instance in instances]
                                        for key in ("input_ids", "input_ids_m3ae", "labels", "ecg", "ecg2"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        input_ids_m3ae = torch.nn.utils.rnn.pad_sequence(
            input_ids_m3ae,
            batch_first=True,
            padding_value=self.tokenizer2.pad_token_id)
        input_ids_m3ae = input_ids_m3ae[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            input_ids_m3ae=input_ids_m3ae,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            text_padding_mask=input_ids_m3ae.eq(self.tokenizer2.pad_token_id)
        )

        ecgs = torch.stack([instance[0]['ecg'] for instance in instances])
        ecgs2 = torch.stack([instance[0]['ecg2'] if 'ecg2' in instance[0] and instance[0]['ecg2'] is not None else torch.zeros(12, 5000) for instance in instances])
        ecg_padding_mask = torch.zeros_like(ecgs, dtype=torch.bool)
        ecg2_padding_mask = torch.tensor([ecg2 is None for ecg2 in ecg2], dtype=torch.bool).unsqueeze(1).unsqueeze(2).expand_as(ecgs2).clone()
        batch['pixel_values'] = ecgs
        batch['ecg_values2'] = ecgs2
        batch['ecg_padding_mask'] = ecg_padding_mask
        batch['ecg2_padding_mask'] = ecg2_padding_mask

        return batch 