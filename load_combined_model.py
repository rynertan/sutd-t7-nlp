import torch
import torch.nn as nn
import torch.nn.functional as F


# Define CNN model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        embeddings = torch.cat(
            (self.embedding(inputs), self.constant_embedding(inputs)), dim=2
        ).permute(0, 2, 1)
        encoding = torch.cat(
            [self.relu(self.pool(conv(embeddings)).squeeze(-1)) for conv in self.convs],
            dim=1,
        )
        outputs = self.decoder(self.dropout(encoding))
        return outputs


# Define RNN model
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(
            embed_size, num_hiddens, num_layers=num_layers, bidirectional=True
        )
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.T)
        outputs, _ = self.encoder(embeddings)
        outs = torch.cat((outputs[0], outputs[-1]), dim=1)
        return self.decoder(outs)


class CRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        output_dim,
        pretrained_embeddings=None,
        freeze_embeddings=False,
        dropout_prob=0.3,
    ):
        super(CRNN, self).__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=freeze_embeddings
            )
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


# Define Combined Model
class CombinedModel(nn.Module):
    def __init__(
        self, cnn, rnn, crnn, combined_dim, hidden_dim, output_dim, dropout_rate=0.5
    ):
        super(CombinedModel, self).__init__()
        self.cnn = cnn
        self.rnn = rnn
        self.crnn = crnn
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        cnn_embedding = self.extract_cnn_embedding(self.cnn, x)
        rnn_embedding = self.extract_rnn_embedding(self.rnn, x)
        crnn_embedding = self.extract_crnn_embedding(self.crnn, x)
        combined_embedding = torch.cat(
            (cnn_embedding, rnn_embedding, crnn_embedding), dim=1
        )
        x = self.fc1(combined_embedding)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def extract_cnn_embedding(cnn, x):
        embeddings = torch.cat(
            (cnn.embedding(x), cnn.constant_embedding(x)), dim=2
        ).permute(0, 2, 1)
        conv_outputs = [cnn.relu(conv(embeddings)) for conv in cnn.convs]
        pooled_outputs = [cnn.pool(conv_out).squeeze(-1) for conv_out in conv_outputs]
        cnn_embedding = torch.cat(pooled_outputs, dim=1)
        return cnn_embedding

    @staticmethod
    def extract_rnn_embedding(rnn, x):
        embeddings = rnn.embedding(x.T)
        outputs, _ = rnn.encoder(embeddings)
        rnn_embedding = torch.cat((outputs[0], outputs[-1]), dim=1)
        return rnn_embedding

    @staticmethod
    def extract_crnn_embedding(crnn, x):
        x = crnn.embedding(x)
        x = crnn.embedding_dropout(x)
        x = x.permute(0, 2, 1)
        x = F.relu(crnn.conv(x))
        x = crnn.conv_dropout(x)
        x = x.permute(0, 2, 1)
        _, hidden_state = crnn.gru(x)
        crnn_embedding = hidden_state[-1]
        return crnn_embedding


def load_combined_model(
    combined_model_path,
    vocab_size,
    combined_dim,
    hidden_dim,
    output_dim,
    dropout_rate,
    device,
):
    """
    Load the combined model only (using saved weights).

    Args:
        combined_model_path: Path to the Combined model weights.
        vocab_size: Vocabulary size for defining the sub-models.
        combined_dim: Dimension of the combined input.
        hidden_dim: Hidden dimension of the final combined model.
        output_dim: Output dimension (number of classes).
        dropout_rate: Dropout rate for the combined model.
        device: Device to load the model onto.

    Returns:
        combined_model: The loaded combined model.
    """
    # Define the sub-model architectures (no weights loaded)
    cnn = TextCNN(
        vocab_size, embed_size=100, kernel_sizes=[3, 4, 5], num_channels=[100, 100, 100]
    )
    rnn = BiRNN(vocab_size, embed_size=100, num_hiddens=100, num_layers=2)
    crnn = CRNN(vocab_size, embed_dim=100, hidden_dim=64, output_dim=2)

    # Initialize Combined Model
    combined_model = CombinedModel(
        cnn, rnn, crnn, combined_dim, hidden_dim, output_dim, dropout_rate
    )

    # Load combined model weights
    combined_model.load_state_dict(torch.load(combined_model_path, map_location=device))
    combined_model = combined_model.to(device)
    combined_model.eval()

    return combined_model
