"""Decoder-only Transformer language model over the 258-token byte vocabulary."""

import torch
import torch.nn as nn

from evaluator import PAD_ID, VOCAB_SIZE


class TransformerLanguageModel(nn.Module):
    def __init__(self, num_embeddings, max_length, d_model, nhead, dim_feedforward, num_layers, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings, d_model, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.output = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, input_ids):
        """(B, T) token ids -> (B, T, 258) logits, with causal attention."""
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        x = self.drop(self.token_embedding(input_ids) + self.position_embedding(positions))
        mask = nn.Transformer.generate_square_subsequent_mask(length, device=input_ids.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        return self.output(x)
