import torch 
from torch import nn
from transformers.activations import ACT2FN

from model.configuration_llama import MultimodalLlamaConfig


class MultiModalLlamaProjector(nn.Module):
    def __init__(self, config: MultimodalLlamaConfig):
        super().__init__()

        # Update the input size to account for the presence mask (1 additional feature dimension)
        self.linear_1 = nn.Linear(
            config.vision_config['hidden_size']+1,  # Adding 1 for the presence mask
            config.text_config.hidden_size,
            bias=True
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=True
        )

    def forward(self, image_features_with_mask):
        # Forward pass with the concatenated input features
        hidden_states = self.linear_1(image_features_with_mask.to(torch.float32))
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
