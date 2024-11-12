


import torch 
from torch import nn
import torch.nn.functional as F


class DiffSegParamModel(nn.Module):
    def __init__(
        self,
        out_thresh_len = 3,
        latent_sequence = 256,
        latent_channel = 256,
        nhead = 8,
        batch_first = True, 
    ) -> None:
        super().__init__()
        encoderLayer = nn.TransformerEncoderLayer(
            d_model = latent_channel,
            nhead = nhead,
            batch_first = batch_first,#  True, 
        )
        self.encoder = nn.TransformerEncoder(
            encoderLayer,
            num_layers=6,
        )
        self.pool = nn.AvgPool1d(
            3,
            stride=2
        )

    def forward(
        self,
        x, 
    ): 
        # In_shape :(batch, seq, feature)
        x  = self.encoder(x)
        # Out_shape: (batch, seq, feature)
        # TODO::
        # Out_shape: (batch, out_feature)
        return F.sigmoid(x) 




if __name__ == "__main__":

    x = torch.randn(2, 64, 64, 256)
    x = x.view(2, -1, 256)

    model = DiffSegParamModel(
        out_thresh_len = 3,
        latent_sequence = 256,
        latent_channel = 256,
        batch_first = True, 
    )
    output = model(x)
    print(output.shape)

