"""
Module defining classes that were used to conduct experiments in order to develop an encoder which encodes
linguistic information more incrementally and therefore closer to the way that humans process language.
"""

from .EncoderRNN import EncoderRNN
from .DecoderRNN import DecoderRNN
from .seq2seq import Seq2seq

from torch import nn
import torch.nn.functional as F


class IncrementalSeq2Seq(Seq2seq):
    """
    Extension of the Seq2Seq model class to enable more problem-specific capabilities.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_encoder_predictions = None

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden, encoder_predictions = self.encoder(input_variable, input_lengths)

        decoder_outputs, decoder_hidden, other = self.decoder(
            inputs=target_variable,
            encoder_hidden=encoder_hidden,
            encoder_outputs=encoder_outputs,
            function=self.decode_function,
            teacher_forcing_ratio=teacher_forcing_ratio
        )

        # Add predictions of the encoder as well as the actual words in the input sequence to compute anticipation loss
        other["encoder_predictions"] = encoder_predictions
        other["input_variable"] = input_variable

        return decoder_outputs, decoder_hidden, other

    @property
    def encoder_predictions(self):
        return self.last_encoder_predictions


class AnticipatingEncoderRNN(EncoderRNN):
    """
    Special kind of encoder which tries to also predict the next token of the sequence, where wrong predictions are
    penalized with a loss function similar to the predictions of a decoder.
    """
    def __init__(self, vocab_size, max_len, hidden_size, embedding_size, input_dropout_p=0, dropout_p=0, n_layers=1,
                 bidirectional=False, rnn_cell='gru', variable_lengths=False):
        super().__init__(
            vocab_size, max_len, hidden_size, embedding_size, input_dropout_p, dropout_p, n_layers,bidirectional,
            rnn_cell, variable_lengths
        )
        self.prediction_layer = nn.Linear(hidden_size, vocab_size)  # Layer to predict next token in input sequence

    def forward(self, input_var, input_lengths=None):
        output, hidden = super().forward(input_var, input_lengths)

        # Try to predict next word in sequence
        encoder_predictions = []

        # output[:, :-1]: Don't use last output for prediction because there is no more token in the sequence that could
        # be predicted
        for o_t in output[:, :-1].split(1, dim=1):
            o_t = o_t.squeeze(1)
            predictive_dist = self.prediction_layer(o_t)
            predictive_dist = F.log_softmax(predictive_dist)
            encoder_predictions.append(predictive_dist)

        return output, hidden, encoder_predictions


class BottleneckDecoderRNN(DecoderRNN):
    """
    Special kind of decoder which only has access to a certain part of the encoded sequence at every time step.
    """
    # TODO: Implement
    pass
