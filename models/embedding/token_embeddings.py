
from torch import nn
import torch

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

class ModelWithBatchNorm(nn.Module):
    def __init__(self, input_size, output_size):
        super(ModelWithBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        batch_size, num_blocks, input_size = x.size()
        x = x.reshape(batch_size * num_blocks, input_size)
        x = self.bn(x)
        x = self.fc(x)
        x = x.view(batch_size, num_blocks, -1)
        return x