import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP
import torch.nn as nn

class ModifiedLlamaMLP(LlamaMLP):
    def __init__(self, config, scale_factors):
        super().__init__(config=config)
        self.intermediate_size = config.intermediate_size
        self.scale_factors = scale_factors  # List of scale factors for 's', 'm', 'l', 'xl'
        self.current_subset_hd = None

    def configure_subnetwork(self, flag):
        """Configure subnetwork size based on flag."""
        hd = self.intermediate_size
        if flag == 's':
            scale = self.scale_factors[0]  # hd/8
        elif flag == 'm':
            scale = self.scale_factors[1]  # hd/4
        elif flag == 'l':
            scale = self.scale_factors[2]  # hd/2
        else:  # 'xl'
            scale = self.scale_factors[3]  # hd

        self.current_subset_hd = int(hd * scale)
    
    def forward(self, x):
        if self.current_subset_hd is None:
            raise ValueError("Subnetwork size not configured. Call `configure_subnetwork` first.")
        gate_proj = self.gate_proj.weight[:self.current_subset_hd]
        up_proj = self.up_proj.weight[:self.current_subset_hd]
        down_proj = self.down_proj.weight[:, :self.current_subset_hd]
        down_proj = F.linear(self.act_fn(F.linear(x, gate_proj) * F.linear(x, up_proj)), down_proj)
        self.current_subset_hd = None

        return down_proj


class ModifiedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        scale_factors = [1/8, 1/4, 1/2, 1]  # s, m, l, xl

        # Replace FFN in each layer with ModifiedFFN
        for layer_idx in range(config.num_hidden_layers):
            self.model.layers[layer_idx].mlp = ModifiedLlamaMLP(config, scale_factors)

    def configure_subnetwork(self, flag):
        """Configure the subnetwork for all layers based on the flag."""
        for layer_idx in range(len(self.model.layers)):
            self.model.layers[layer_idx].mlp.configure_subnetwork(flag)
