import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    A class that returns sinusoidal positional encodings for given position IDs.

    Args:
        d_model (int): The dimensionality of the positional encoding.
        max_len (int): The maximum index (position) for which to precompute encodings.
    """
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        
        # Create a (max_len, d_model) tensor for the positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # Positions range: 0, 1, 2, ..., max_len-1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # The term used for dividing positions in the sin/cos formula
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(500_000.0) / d_model)
        )
        
        # Apply the standard sinusoidal formula:
        #   PE(pos, 2i)   = sin( pos / (10000^(2i/d_model)) )
        #   PE(pos, 2i+1) = cos( pos / (10000^(2i/d_model)) )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        
        # Register the precomputed encoding so it is moved to the correct device
        # and not trained
        self.register_buffer('pe', pe)

        self.d_model = d_model

    def forward(self, ids):
        """
        Given a tensor of shape (N, S) with position IDs, 
        returns a tensor of shape (N, S, d_model) with their positional encodings.

        Args:
            ids (torch.LongTensor): A tensor of shape (N, S) containing the position IDs.

        Returns:
            torch.FloatTensor: Positional encodings of shape (N, S, d_model).
        """
        # Simply use the IDs as indices into the precomputed table:
        # shape of result is (N, S, d_model)
        return self.pe[ids]
