import sys
sys.path.append("/home/hschung/fairseq-signals/")

from transformers import PretrainedConfig, AutoConfig, PatchTSTConfig
from dataclasses import asdict
from utils.constants import LORA_CONFIG
from fairseq_signals.models.m3ae import M3AEModel, M3AEConfig

class MultimodalLlamaConfig(PretrainedConfig):
    model_type = "multimodal_llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vision_model_id=None,
        ecg_model_id=None,
        text_model_id=None,
        ignore_index=-100,
        load_in_4bit=False,
        projector_hidden_act="gelu",
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        freeze_multimodal_projector=False,
        freeze_language_model=False,
        freeze_vision_model=False,
        tokenizer_len=None,
        pad_token_index=None,
        image_token_index=None,
        encoding=False,
        lora_config=LORA_CONFIG,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ignore_index = ignore_index
        self.projector_hidden_act = projector_hidden_act
        self.load_in_4bit = load_in_4bit
        self.encoding = encoding

        self.vision_model_id = vision_model_id
        self.text_model_id = text_model_id
        self.ecg_model_id = ecg_model_id

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

        if self.encoding == "vision_encoder":
            vision_config = AutoConfig.from_pretrained(vision_model_id)
            if vision_config.model_type == "clip":
                vision_config = vision_config.vision_config
            self.vision_config = vision_config
            self.vision_config.hidden_size =768
        else:
            self.vision_config = None

        if self.encoding == "ecg_encoder": 
            ecg_config = M3AEConfig()
            self.ecg_config = ecg_config
        elif self.encoding == 'ecg_clip':
            ecg_config = AutoConfig.from_pretrained("/nfs_edlab/hschung/ecg_clip/ecg_clip/ecg_clip_model_epoch50/")
            self.ecg_config = ecg_config
        else:
            self.ecg_config = None
            
        if self.encoding == "patchtst":
            self.patchtst_config = PatchTSTConfig(
            num_input_channels=12,
            patch_length=100,
            stride=50,
            d_model=768,
            context_length=5000,
        ) 
        else:
            self.patchtst_config = None
        