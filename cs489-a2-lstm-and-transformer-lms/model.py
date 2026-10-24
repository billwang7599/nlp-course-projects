"""LSTM language model over the 258-token byte vocabulary."""

import torch.nn as nn
import torch.nn.functional as F

from evaluator import PAD_ID, VOCAB_SIZE


class LSTMLanguageModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout_percentage, hidden_size, neural_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=PAD_ID)
        self.drop1 = nn.Dropout(dropout_percentage)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, neural_layers, bias=True,
                batch_first=True, dropout=dropout_percentage if neural_layers > 1 else 0.0,
                bidirectional=False)
        self.drop2 = nn.Dropout(dropout_percentage)
        self.output = nn.Linear(hidden_size, VOCAB_SIZE, bias=True)

    def forward(self, input_ids):
        embedding_o = self.embedding(input_ids)
        drop1_o = self.drop1(embedding_o)
        lstm_o, _ = self.lstm(drop1_o)
        drop2_o = self.drop2(lstm_o)
        final_output = self.output(drop2_o)
        return final_output


def next_token_loss(logits, targets):
    return F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), ignore_index=-100)
