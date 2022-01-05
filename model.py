import torch
import torch.nn as nn

device = 'cpu'


class AttentionModel(nn.Module):
    def __init__(self, output_size, hidden_size, input_size, embedding_length):
        super(AttentionModel, self).__init__()

        self.hidden_size = hidden_size

        self.word_embeddings = nn.Embedding(input_size, embedding_length)
        self.lstm_1 = nn.LSTM(embedding_length, hidden_size)
        self.lstm_2 = nn.LSTM(embedding_length, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=1, batch_first=True)
        self.linear_final = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_sentences):

        input_ = self.word_embeddings(input_sentences)
        input_ = input_.permute(1, 0, 2)
        query = [(torch.zeros(input_.shape[0], self.hidden_size),
                          torch.zeros(input_.shape[0], self.hidden_size)) for _ in range(2)]

        key, _ = self.lstm_1(input_)
        value, _ = self.lstm_2(input_)
        attn_output, _ = self.attention(query[1][0].unsqueeze(1).to(device), key.to(device), value.to(device))
        logits = self.linear_final(attn_output)
        return logits.squeeze(1)