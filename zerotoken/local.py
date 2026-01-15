from torch import nn
import torch

class ReLULocalZeroToken(nn.Module):
    def __init__(self, layer, config, l1_reg_coeff=1.0):
        super().__init__()
        self.layer = layer

        # FP32 router
        self.sparsifyer = nn.Linear(
            config.hidden_size, 1, bias=False, dtype=torch.float32
        )

        # Initialize close to identity
        self.sparsifyer.weight.data.fill_(1.0)

        self.relu = nn.ReLU()
        self.l1_reg_coeff = l1_reg_coeff

        self.attention_type = layer.attention_type


    def forward(self, *args, **kwargs):
        hidden_states = args[0]  # [b, s, h]
        orig_dtype = hidden_states.dtype

        hs_fp32 = hidden_states.to(torch.float32)
        logits = self.sparsifyer(hs_fp32)          # [b, s, 1]
        logits = torch.tanh(logits) # the logits explode
        sparsity_weights = self.relu(logits)
        # sparsity_weights = sparsity_weights.clamp(max=1.0)

        if self.training:
            self.reg_loss = sparsity_weights.mean() * self.l1_reg_coeff
            self.mean_zero_tokens = (sparsity_weights == 0).float().mean()
        else:
            self.reg_loss = None

        weighted_hidden = hs_fp32 * sparsity_weights
        weighted_hidden = weighted_hidden.to(orig_dtype)

        output = self.layer(weighted_hidden, **kwargs)

        return output
