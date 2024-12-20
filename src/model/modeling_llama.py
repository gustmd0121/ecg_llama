import os
from typing import Optional, List, Union, Tuple
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, Cache, PreTrainedModel, BitsAndBytesConfig
from transformers.generation.utils import GenerationMixin
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast

from model.configuration_llama import MultimodalLlamaConfig
from model.multimodal_projector import MultiModalLlamaProjector

from fairseq_signals.models import build_model_from_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultimodalLlamaForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = MultimodalLlamaConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: MultimodalLlamaConfig):
        super().__init__(config)

        # Get model dtype first
        self.model_dtype = torch.float32 if config.load_in_8bit else torch.bfloat16
        
        self.config = config
        # Change from (16, 1, 1024) to (1, 1024) - single separator vector
        try:
            if self.config.vision_config.hidden_size:
                self.sep_embedding = nn.Parameter(torch.zeros(1, self.config.vision_config.hidden_size, dtype=self.model_dtype))
        except AttributeError:
            self.sep_embedding = nn.Parameter(torch.zeros(1, self.config.vision_config['hidden_size'], dtype=self.model_dtype))

        # Configure quantization
        quantization_config = None
        if config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )

        # Set device map based on local rank
        if config.local_rank != -1:
            device_map = {"": config.local_rank}
        else:
            device_map = "auto" if config.load_in_8bit else None

        # Instantiate models with proper dtype
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.text_model_id,
            resume_download=True,
            torch_dtype=self.model_dtype,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2",
        ).to(device)
        
        
        if config.data_type == "image":
            self.vision_model = AutoModel.from_pretrained(
                config.vision_model_id
            ).to(config.device).to(self.model_dtype)

            # Keep only vision model's encoder if using CLIP
            if "clip" in self.config.vision_model_id:
                self.vision_model = self.vision_model.vision_model
        elif config.data_type == "signal":
            self.vision_model = build_model_from_checkpoint(
            os.path.join(os.path.dirname(__file__), '../../ckpts/mimic_iv_ecg_physionet_pretrained.pt')
            ).to(config.device).to(self.model_dtype)

        # Instantiate the multimodal projector
        self.vocab_size = self.config.text_config.vocab_size
        self.multi_modal_projector = MultiModalLlamaProjector(self.config)

        # Resize embedding layer from the model
        self.resize_token_embeddings(self.config.tokenizer_len)

        # Freeze/unfreeze parameters according to the flags
        self._adjust_model_parameters(freeze_multimodal_projector=self.config.freeze_multimodal_projector,
                                      freeze_language_model=self.config.freeze_language_model,
                                      freeze_vision_model=self.config.freeze_vision_model)

        self.post_init()

    def _adjust_model_parameters(self, freeze_multimodal_projector=False, freeze_language_model=False,
                                 freeze_vision_model=False):
        """
        Adjust model parameters.
        """
        if freeze_multimodal_projector:
            for p in self.multi_modal_projector.parameters():
                p.requires_grad = False

        if freeze_language_model:
            for p in self.language_model.parameters():
                p.requires_grad = False
        else:
            # Fix for gradient checkpointing with Lora
            if hasattr(self.language_model, "enable_input_require_grads"):
                self.language_model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                self.language_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # Add Lora to the language model
            config = LoraConfig(**self.config.lora_config)
            self.language_model = get_peft_model(self.language_model, config)
            self.language_model.print_trainable_parameters()

        if freeze_vision_model:
            self.vision_model.training = False
            for p in self.vision_model.parameters():
                p.requires_grad = False

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        # Convert image_features to match inputs_embeds dtype
        image_features = image_features.to(dtype=inputs_embeds.dtype)
        
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.config.pad_token_index))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, 
            dtype=inputs_embeds.dtype,  # Use same dtype as inputs_embeds
            device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        # Ensure dtype match before assignment
        image_features_reshaped = image_features.contiguous().reshape(-1, embed_dim).to(
            device=target_device, 
            dtype=final_embedding.dtype
        )
        final_embedding[image_to_overwrite] = image_features_reshaped
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.config.pad_token_index)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        images2: torch.FloatTensor = None,
        ecg: torch.FloatTensor = None,
        ecg2: torch.FloatTensor = None,
        ecg_padding_mask: torch.BoolTensor = None,
        ecg_padding_mask2: torch.BoolTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        inference: Optional[bool] = None
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        # Convert inputs to correct dtype
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.model_dtype)
            if images2 is not None:
                images2 = images2.to(self.model_dtype)
        
        if ecg is not None:
            ecg = ecg.to(self.model_dtype)
            if ecg2 is not None:
                ecg2 = ecg2.to(self.model_dtype)

        if pixel_values is not None and pixel_values.shape[0] == 2:
            images2 = pixel_values[1].unsqueeze(0)
            pixel_values = pixel_values[0].unsqueeze(0)

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                # Ensure vision model outputs match language model dtype
                image_outputs = self.vision_model(pixel_values, output_hidden_states=True)
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

                # Check if images2 is not all zeros (presence of a second image)
                images2_not_all_zeros = torch.any(images2 != 0, dim=(1, 2, 3))  # Convert to float for compatibility

                # Process second images
                image_outputs2 = self.vision_model(images2, output_hidden_states=True)
                selected_image_feature2 = image_outputs2.hidden_states[vision_feature_layer]

                # Create a presence mask for the second image features
                presence_mask = images2_not_all_zeros.view(-1, 1, 1)  # [bs, 1, 1], same as zeros_mask before

                # Use the mask to inform the model about second image presence
                # Concatenate features with a separator embedding
                batch_size = selected_image_feature.size(0)
                sep_expanded = self.sep_embedding.unsqueeze(0).expand(batch_size, 1, -1)

                # Concatenate along the sequence dimension
                ecg_concat = torch.cat([
                    selected_image_feature,
                    sep_expanded,                  
                    selected_image_feature2
                ], dim=1)

                # Incorporate the presence mask as an additional feature dimension
                # Expand and append to the concatenated features
                presence_mask_expanded = presence_mask.expand(-1, ecg_concat.size(1), 1)  # Expand to match concatenated feature shape
                ecg_concat_with_mask = torch.cat([ecg_concat, presence_mask_expanded], dim=-1)  # Concatenate as a new feature dimension

                # Pass concatenated features with the mask to the projector
                image_features = self.multi_modal_projector(ecg_concat_with_mask)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)
                    
            elif ecg is not None and input_ids.shape[1] != 1:
                # Ensure vision model outputs match language model dtype
                ecg_outputs = self.vision_model.extract_features(ecg, padding_mask=ecg_padding_mask)
                selected_ecg_feature = ecg_outputs['x']

                # Check if ecg2 is not all zeros (presence of a second image)
                ecg2_not_all_zeros = ~torch.all(ecg_padding_mask2, dim=(1, 2))  

                # Process second images
                ecg_outputs2 = self.vision_model.extract_features(ecg2, padding_mask=ecg_padding_mask2)
                selected_ecg_feature2 = ecg_outputs2['x']

                # Create a presence mask for the second image features
                presence_mask = ecg2_not_all_zeros.view(-1, 1, 1)  # [bs, 1, 1], same as zeros_mask before false means two ecgs are present

                # Use the mask to inform the model about second ecg presence
                # Concatenate features with a separator embedding
                batch_size = selected_ecg_feature.size(0)
                sep_expanded = self.sep_embedding.unsqueeze(0).expand(batch_size, 1, -1)

                # Concatenate along the sequence dimension
                ecg_concat = torch.cat([
                    selected_ecg_feature,
                    sep_expanded,                  
                    selected_ecg_feature2
                ], dim=1)

                # Incorporate the presence mask as an additional feature dimension
                # Expand and append to the concatenated features
                presence_mask_expanded = presence_mask.expand(-1, ecg_concat.size(1), 1).to(device)  # Expand to match concatenated feature shape
                ecg_concat_with_mask = torch.cat([ecg_concat, presence_mask_expanded], dim=-1)  # Concatenate as a new feature dimension

                # Pass concatenated features with the mask to the projector
                ecg_features = self.multi_modal_projector(ecg_concat_with_mask)
                inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                    ecg_features, inputs_embeds, input_ids, attention_mask, labels
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)


            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            # elif past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
            #     # Retrieve the first layer to inspect the logits and mask out the hidden states
            #     # that are set to 0
            #     # that are set to 0
            #     first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

            #     # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
            #     batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

            #     # Get the target length
            #     target_length = input_ids.shape[1]
            #     past_length = first_layer_past_key_value.shape[-1]

            #     extended_attention_mask = torch.ones(
            #         (attention_mask.shape[0], past_length),
            #         dtype=attention_mask.dtype,
            #         device=attention_mask.device,
            #     )

            #     # Filter out only the tokens that can be un-attended, this can happen
            #     # if one uses Llava + Fused modules where the cache on the
            #     # first iteration is already big enough, or if one passes custom cache
            #     valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
            #     new_batch_index = batch_index[valid_indices]
            #     new_non_attended_tokens = non_attended_tokens[valid_indices]

            #     # Zero-out the places where we don't need to attend
            #     extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

            #     attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
            #     position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]
        
        if inference:
            return outputs

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        inputs_embeds=None, 
        pixel_values=None, 
        images2=None,  # Include images2 here
        attention_mask=None, 
        **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1:]
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]):]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Update model inputs to include images2
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "images2": images2,  # Add images2 to model inputs
            }
        )
        
        if 'ecg' in kwargs:
            model_inputs['ecg'] = kwargs['ecg']
            model_inputs['ecg_padding_mask'] = kwargs['ecg_padding_mask']
        if 'ecg2' in kwargs:
            model_inputs['ecg2'] = kwargs['ecg2']
            model_inputs['ecg_padding_mask2'] = kwargs['ecg_padding_mask2']
        
        return model_inputs


    def can_generate(self) -> bool:
        """Whether this model can generate."""
        return True
