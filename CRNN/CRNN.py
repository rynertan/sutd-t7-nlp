import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pretrained_embeddings=None, freeze_embeddings=False, dropout_prob=0.3):
        super(CRNN, self).__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding_dropout = nn.Dropout(p=dropout_prob)

        # conv layer
        self.conv = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.conv_dropout = nn.Dropout(p=dropout_prob)

        # recurr layer (switched to use GRU instead of LSTM, performed better)
        self.gru = nn.GRU(64, hidden_dim, batch_first=True)
        self.gru_dropout = nn.Dropout(p=dropout_prob)

        # fc
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = self.conv_dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.gru_dropout(x)
        x = self.fc(x[:, -1, :])
        return x