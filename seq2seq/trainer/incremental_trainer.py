"""
Subclass of the supervised trainer class in order to train different models that try to encourage incremental
representation in the encoder.
"""

from .supervised_trainer import SupervisedTrainer


class IncrementalTrainer(SupervisedTrainer):
    """
    Trainer that trains the anticipating encoder by attaching an additional loss to predictions of the next token
    on the encoder side.
    """
    # TODO: Implement
    pass
