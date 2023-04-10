import torch
from torch import nn
from config import Confiq
import math

cfg = Confiq()

class BILSTM(nn.Module):
    def __init__(self, num_embedding, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.2, device='cuda'):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)
        self.bi_lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_dim*2, out_features=output_size)

    def forward(self, x):
        x = x.long()
        h0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_dim).to(self.device)
        embeds=self.embedding(x)
        lstm_out, _ = self.bi_lstm(embeds, (h0, c0))
        lstm_out = lstm_out[:, -1, :]#take last hidden state
        #lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        return out

class GRU(nn.Module):
    def __init__(self, num_embedding, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.2, device='cuda'):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_size)

    def forward(self, x):
        x = x.long()
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(self.device)
        embeds=self.embedding(x)
        gru_out, _ = self.gru(embeds, h0)
        gru_out = gru_out[:, -1, :]#take last hidden state
        out = self.fc(gru_out)
        return out

class LSTM(nn.Module):
    def __init__(self, num_embedding, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.2, device='cuda'):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_size)

    def forward(self, x):
        x = x.long()
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(self.device)
        embeds=self.embedding(x)
        lstm_out, _ = self.lstm(embeds, (h0, c0))
        lstm_out = lstm_out[:, -1, :]#take last hidden state
        out = self.fc(lstm_out)
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        nhead=10,
        dim_feedforward=50,
        num_layer=6,
        dropout=0.0,
    ):
        super().__init__()

        self.emb = nn.Embedding(num_embeddings=len(cfg.AMINO)+1, embedding_dim=cfg.number_family)

        amino_size, d_model = len(cfg.AMINO), cfg.number_family
        assert amino_size % nhead == 0, "nhead harus habis dibagi dengan amino size"

        self.pos_encoder = PositionalEncoding(
            d_model=d_model, 
            vocab_size=250, 
            dropout=dropout
        )
        
        encoderLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoderLayer,
            num_layers=num_layer
        )

        self.classifier = nn.Linear(250, 512)
        self.classifier2 = nn.Linear(512, 512)
        self.classifier3 = nn.Linear(512, cfg.number_family)
        self.d_model = d_model
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.emb(x)* math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        x = self.relu(x)
        x = self.classifier2(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, vocab_size=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class BiGRU(nn.Module):
    def __init__(self, num_embedding, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.2, device='cuda'):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=drop_prob,bidirectional=True, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_dim*2, out_features=output_size)

    def forward(self, x):
        x = x.long()
        h0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_dim).to(self.device)
        embeds=self.embedding(x)
        gru_out, _ = self.gru(embeds, h0)
        gru_out = gru_out[:, -1, :]#take last hidden state
        out = self.fc(gru_out)
        return out