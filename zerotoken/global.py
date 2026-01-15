from torch import nn

class NaiveGlobalZeroToken(nn.Module):
    def __init__(self, llm):
        super(NaiveGlobalZeroToken, self).__init__()
        self.llm = llm
        self.encoder = llm.__class__.from_config(llm.config, num_hidden_layers=3)
        self.sparsifyer = nn.Linear(llm.config.hidden_size, 1, bias=False, dtype=llm.dtype)
        self.relu = nn.ReLU()
        self.decoder = llm.__class__.from_config(llm.config, num_hidden_layers=3)

        delattr(self.base, "embed_tokens")
        delattr(self.decoder, "embed_tokens")

        self.lm_head = llm.lm_head

        del self.llm.embeddings
    
    def forward(self, *args, **kwargs):
        input_ids = kwargs.pop('input_ids')
        labels = kwargs.pop('labels', None)

        e = self.llm.embed_tokens(input_ids)
        e = self.encoder(inputs_embeds=e, attention_mask=kwargs.get('attention_mask', None)).last_hidden_state
        s = self.relu(self.sparsifyer(e))
        # TODO: get non-zero tokens
        kwargs["input_embeds"] = e

        o = self.llm.model(*args, **kwargs).last_hidden_state
        d = self.decoder(inputs_embeds=o.last_hidden_state, attention_mask=kwargs.get('attention_mask', None)).last_hidden_state
        pass