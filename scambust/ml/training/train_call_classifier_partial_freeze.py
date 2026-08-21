"""
Experiment A: freeze the bottom 6 of MuRIL's 12 encoder layers (+ embeddings)
statically from step 1. Only the top 6 layers + pooler + classifier head
train. Hypothesis: less capacity to memorize domain-specific surface
patterns -> better generalization to unseen scam domains, at some possible
cost to in-domain accuracy (which has headroom to give, currently ~99%).

Usage:
    python -m ml.training.train_call_classifier_partial_freeze
"""
import os

from .common import NUM_ENCODER_LAYERS, set_trainable_layers
from ._call_classifier_core import run_experiment

MIN_TRAINABLE_LAYER = 6  # freeze layers 0-5, train layers 6-11 + pooler + head
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "models_out", "call_classifier_partial_freeze")


def configure(model):
    set_trainable_layers(model, MIN_TRAINABLE_LAYER)


if __name__ == "__main__":
    run_experiment("partial_freeze(bottom6)", OUT_DIR, configure)
