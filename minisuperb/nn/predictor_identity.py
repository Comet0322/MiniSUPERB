import torch.nn as nn

from minisuperb import Output


class PredictorIdentity(nn.Module):
    """
    This nn module is used as a predictor placeholder for certain SSL problems.
    """

    def __init__(self, **kwargs):
        super(PredictorIdentity, self).__init__()

    def forward(self, output: Output):
        """
        Args:
            output (minisuperb.Output): An Output module

        Return:
            output (minisuperb.Output): exactly the same as input, an Output module
        """
        return output
