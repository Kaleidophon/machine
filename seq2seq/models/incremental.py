"""
Module defining classes that were used to conduct experiments in order to develop an encoder which encodes
linguistic information more incrementally and therefore closer to the way that humans process language.
"""

from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN


class AnticipatingEncoderRNN(EncoderRNN):
    """
    Special kind of encoder which tries to also predict the next token of the sequence, where wrong predictions are
    penalized with a loss function similar to the predictions of a decoder.
    """
    # TODO: Implement
    pass


class BottleneckDecoderRNN(DecoderRNN):
    """
    Special kind of decoder which only has access to a certain part of the encoded sequence at every time step.
    """
    # TODO: Implement
    pass
