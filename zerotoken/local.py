from torch import nn
import torch

class ReLULocalZeroToken(nn.Module):
    def __init__(self, layer, config, l1_reg_coeff=0.05):
        super().__init__()
        self.layer = layer

        # FP32 router
        self.sparsifyer = nn.Sequential(
            nn.Linear(
            config.hidden_size, 1, bias=True, dtype=torch.float32
            ),
            # nn.GELU(),
            # nn.Linear(
            # round(config.hidden_size // 4), 1, bias=False, dtype=torch.float32
            # )
        )

        # Initialize close to identity
        self.sparsifyer.apply(self.init_weights)

        self.relu = nn.ReLU()
        self.l1_reg_coeff = l1_reg_coeff

        self.attention_type = layer.attention_type

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            # module.weight.data.fill_(1.0)
            module.weight.data.fill_(0.0)
            module.bias.data.fill_(1.0)

    def forward(self, *args, **kwargs):
        hidden_states = args[0]  # [b, s, h]
        labels = kwargs.get("labels")
        cos, sin = kwargs.get("position_embeddings")
        b, s, h = hidden_states.shape
        orig_dtype = hidden_states.dtype

        hs_fp32 = hidden_states #.to(torch.float32)

        logits = self.sparsifyer(hs_fp32.to(torch.float32))          # [b, s, 1]
        # logits = torch.tanh(logits) # the logits explode
        sparsity_weights = torch.nn.functional.sigmoid(logits).to(orig_dtype)  # [b, s, 1]
        # hidden_states = hidden_states * sparsity_weights
        # sparsity_weights = sparsity_weights.clamp(max=1.0)

        if self.training:
            hard_mask = (sparsity_weights >= 0.5).float()
            ste_mask = hard_mask + (sparsity_weights - sparsity_weights.detach())
            self.reg_loss = ste_mask.mean() * self.l1_reg_coeff
            self.mean_zero_tokens = (sparsity_weights < 0.5).float().mean()
        else:
            self.reg_loss = None

        mask = (sparsity_weights >= 0.5).squeeze(-1)
        mask[labels == -100] = True
        
        # If no tokens survive, just weight the output
        if mask.sum() == 0:
            return hidden_states

        # weighted_hidden = hs_fp32 * sparsity_weights
        # weighted_hidden = weighted_hidden.to(orig_dtype)
        
        # Handle FlashAttention parameters
        cu_seq_lens = kwargs.get("cu_seq_lens_q")
        seq_lens = torch.diff(cu_seq_lens)
        
        splitted_masks = torch.split(mask.view(-1), tuple(seq_lens.tolist()), dim=0)
        new_lengths = torch.tensor([s.sum().item() for s in splitted_masks], device=mask.device)
        
        max_length = new_lengths.max().item()
        new_cu_seq_lens = torch.cat([
            torch.tensor([0], device=mask.device), 
            torch.cumsum(new_lengths, dim=0)
        ])
        
        new_position_ids = torch.cat([
            torch.arange(length, device=mask.device) 
            for length in new_lengths.tolist()
        ]).unsqueeze(0)
        
        # Extract non-masked tokens
        hs_fp32 = hs_fp32 * sparsity_weights
        hidden_states_masked = hs_fp32[mask].view(b, -1, h)
        
        # Extract corresponding position embeddings
        cos_masked = cos[mask].view(b, -1, cos.shape[-1])
        sin_masked = sin[mask].view(b, -1, sin.shape[-1])
        
        # Update kwargs
        kwargs["position_embeddings"] = (cos_masked, sin_masked)
        kwargs["position_ids"] = new_position_ids
        kwargs["cu_seq_lens_q"] = kwargs["cu_seq_lens_k"] = new_cu_seq_lens.to(torch.int32)
        kwargs["max_length_q"] = kwargs["max_length_k"] = max_length

        if "attention_mask" in kwargs:
            del kwargs["attention_mask"]
        # sparsity_weights_masked = sparsity_weights.squeeze(-1)[mask].view(b, -1, 1)
        output_masked = self.layer(hidden_states_masked, **kwargs)

        # Create output tensor and scatter back
        output = torch.zeros_like(hidden_states)
        output_flat = output.view(-1, h)
        output_masked_flat = output_masked.view(-1, h)
        mask_flat = mask.view(-1)
        
        output_flat[mask_flat] = output_masked_flat
        output = output_flat.view(b, s, h) # * sparsity_weights

        return output