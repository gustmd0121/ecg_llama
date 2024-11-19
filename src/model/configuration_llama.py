from transformers import PretrainedConfig, AutoConfig
import os 
from utils.constants import LORA_CONFIG
from huggingface_hub import hf_hub_download
from fairseq_signals.models import build_model_from_checkpoint
import yaml
import json 

class MultimodalLlamaConfig(PretrainedConfig):
    model_type = "multimodal_llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vision_model_id=None,
        text_model_id=None,
        ignore_index=-100,
        load_in_8bit=False,  # Add 8-bit support
        use_bfloat16=True,  # Add this parameter
        data_type="image",
        projector_hidden_act="gelu",
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        freeze_multimodal_projector=False,
        freeze_language_model=False,
        freeze_vision_model=False,
        tokenizer_len=None,
        pad_token_index=None,
        image_token_index=None,
        lora_config=LORA_CONFIG,
        device="cuda",
        local_rank=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        #data type
        self.data_type = data_type
        
        self.ignore_index = ignore_index
        self.projector_hidden_act = projector_hidden_act
        self.load_in_8bit = load_in_8bit  # Add 8-bit flag
        self.use_bfloat16 = use_bfloat16

        self.vision_model_id = vision_model_id
        self.text_model_id = text_model_id

        self.freeze_multimodal_projector = freeze_multimodal_projector
        self.freeze_language_model = freeze_language_model
        self.freeze_vision_model = freeze_vision_model
        self.tokenizer_len = tokenizer_len
        self.lora_config = lora_config

        # Vision feature selection
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        
        #token_index
        self.image_token_index = image_token_index
        self.pad_token_index = pad_token_index

        self.device = device
        self.local_rank = local_rank

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        # Instantiate the pretraining configs for text and vision models
        if text_model_id is not None:
            text_config = AutoConfig.from_pretrained(text_model_id)
            self.text_config = text_config
        else:
            self.text_config = None

        if vision_model_id is not None and data_type == "image":
            vision_config = AutoConfig.from_pretrained(vision_model_id)
            if vision_config.model_type == "clip":
                vision_config = vision_config.vision_config
            self.vision_config = vision_config
        else:
            # remember to hf_donwload the .pt and yaml file of the mimic_iv_ecg_physionet_pretrained model <important>
            with open(os.path.join('/home/hschung/ecg-llm/llama-multimodal-vqa/', 'ckpts', 'mimic_iv_ecg_physionet_pretrained.json'), 'r') as file:
                self.vision_config = json.load(file)
