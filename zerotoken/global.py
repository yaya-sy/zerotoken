from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutput


class ReLUGlobalZeroToken(nn.Module):
    def __init__(self, model, l1_reg_coeff=0.005):
        super().__init__()
        self.model = model
        # self.backbone = getattr(model, "model", model)
        config_ = deepcopy(model.config)
        config_.num_hidden_layers = 3
        self.encoder = model.__class__._from_config(config_)
        self.decoder = model.__class__._from_config(config_)
        # self.lm_head = getattr(model, "lm_head", None)

        self.encoder.to(dtype=model.dtype)
        self.decoder.to(dtype=model.dtype)
        self.sparsifyer = nn.Sequential(
            nn.Linear(model.config.hidden_size, 1, bias=True, dtype=torch.float32)
        )
        # self.sparsifyer.weight.data.fill_(0.0)
        # self.sparsifyer.bias.data.fill_(1.0)
        self.sparsifyer.apply(self.init_weights)
        self.relu = nn.ReLU()
        self.l1_reg_coeff = l1_reg_coeff
        self.reg_loss = None
        self.mean_zero_tokens = None

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            # module.weight.data.fill_(1.0)
            module.weight.data.fill_(0.0)
            module.bias.data.fill_(1.0)

    def forward(self, *args, **kwargs):
        orig_kwargs = deepcopy(kwargs)
        del orig_kwargs["input_ids"]
        input_ids = kwargs.pop("input_ids", None)
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        labels = kwargs.pop("labels", None)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        return_dict = kwargs.pop("return_dict", True)

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("ReLUGlobalZeroToken requires input_ids or inputs_embeds.")
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        b, s, _ = inputs_embeds.shape
        orig_dtype = inputs_embeds.dtype

        enc_out = self.encoder.model(
            inputs_embeds=inputs_embeds,
            # attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        ).last_hidden_state
        logits = self.sparsifyer(enc_out.to(torch.float32))
        # logits = torch.tanh(logits)  # the logits explode
        sparsity_weights = torch.nn.functional.sigmoid(logits).squeeze(-1)

        zero_mask = (sparsity_weights < 0.5)
        mask = ~zero_mask
        if labels is not None:
            mask[labels == -100] = True
        sparsity_weights_masked = sparsity_weights[mask]

        if self.training:
            # self.reg_loss = sparsity_weights.mean() * self.l1_reg_coeff
            # STE: use hard mask in forward, pass gradients from sparsity_weights
            hard_mask = (sparsity_weights >= 0.5).float()
            ste_mask = hard_mask + (sparsity_weights - sparsity_weights.detach())
            self.reg_loss = ste_mask.mean() * self.l1_reg_coeff
            self.mean_zero_tokens = zero_mask.float().mean()
        else:
            self.reg_loss = None

        # weighted_embeds = (enc_out.to(torch.float32) * sparsity_weights.unsqueeze(-1)).to(orig_dtype)
        weighted_embeds = enc_out

        cu_seq_lens = kwargs.get("cu_seq_lens_q")
        seq_lens = torch.diff(cu_seq_lens)

        splitted_masks = torch.split(mask.view(-1), tuple(seq_lens.tolist()), dim=0)
        new_lengths = torch.tensor([s.sum().item() for s in splitted_masks], device=mask.device)

        max_length = new_lengths.max().item()
        new_cu_seq_lens = torch.cat(
            [torch.tensor([0], device=mask.device), torch.cumsum(new_lengths, dim=0)]
        )

        new_position_ids = torch.cat(
            [torch.arange(length, device=mask.device) for length in new_lengths.tolist()]
        ).unsqueeze(0)

        # Extract non-masked tokens
        weighted_embeds_masked = weighted_embeds[mask].view(b, -1, weighted_embeds.shape[-1])
        # print(sparsity_weights_masked.shape, weighted_embeds_masked.shape)
        weighted_embeds_masked = weighted_embeds_masked * sparsity_weights_masked.view(b, -1, 1)

        backbone_kwargs = dict(kwargs)
        if "attention_mask" in backbone_kwargs:
            del backbone_kwargs["attention_mask"]

        # if "position_embeddings" in backbone_kwargs and backbone_kwargs["position_embeddings"] is not None:
        #     cos, sin = backbone_kwargs["position_embeddings"]
        #     cos_masked = cos[mask].view(b, -1, cos.shape[-1])
        #     sin_masked = sin[mask].view(b, -1, sin.shape[-1])
        #     backbone_kwargs["position_embeddings"] = (cos_masked, sin_masked)

        backbone_kwargs["position_ids"] = new_position_ids
        backbone_kwargs["cu_seq_lens_q"] = backbone_kwargs["cu_seq_lens_k"] = new_cu_seq_lens.to(
            torch.int32
        )
        backbone_kwargs["max_length_q"] = backbone_kwargs["max_length_k"] = max_length
        backbone_out_masked = self.model.model(
            inputs_embeds=weighted_embeds_masked,
            return_dict=True,
            **backbone_kwargs,
        ).last_hidden_state

        # After getting backbone_out_masked
        backbone_out_masked_result = backbone_out_masked

        # Create a full tensor with the original shape
        backbone_out = torch.zeros(b, s, backbone_out_masked.shape[-1], 
                                dtype=backbone_out_masked.dtype, 
                                device=backbone_out_masked.device)

        # Fill in the non-masked positions with backbone results
        backbone_out[mask] = backbone_out_masked_result.view(-1, backbone_out_masked.shape[-1])

        # Reset positions where zero_mask is True to original enc_out values
        backbone_out[zero_mask] = enc_out[zero_mask]
        dec_out = self.decoder.model(
            inputs_embeds=backbone_out,
            return_dict=True,
            **orig_kwargs
        ).last_hidden_state

        logits = self.model.lm_head(dec_out.to(torch.float32))#.to(torch.float32)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return CausalLMOutput(loss=loss, logits=logits)

    def save_pretrained(self, *args, **kwargs):
        return self.model.save_pretrained(*args, **kwargs)
