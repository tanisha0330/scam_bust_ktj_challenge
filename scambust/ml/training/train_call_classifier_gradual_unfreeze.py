"""
Experiment B: ULMFiT-style gradual unfreezing. Starts with only the
pooler + classifier head trainable (everything else frozen), then unfreezes
2 more encoder layers (top-down) at each epoch boundary, ending at the same
final trainable set as the partial-freeze experiment (layers 6-11 + pooler +
head) by the last epoch -- so the two experiments are a clean A/B on
"reach the same trainable-parameter set gradually vs. all at once."

Usage:
    python -m ml.training.train_call_classifier_gradual_unfreeze
"""
import os

from .common import GradualUnfreezeCallback, NUM_ENCODER_LAYERS, set_trainable_layers
from ._call_classifier_core import run_experiment

# One entry per epoch: min_trainable_layer for that epoch. Finer-grained than
# the first attempt (1 layer/epoch instead of 2) and longer (8 epochs instead
# of 4), so the model gets 2 full epochs at max depth (layers 6-11 trainable)
# instead of just 1 -- the first run's biggest handicap vs. partial_freeze,
# which trained at full depth for all 4 of its epochs.
# Epoch 0: only pooler+head (nothing in encoder trainable)
# Epoch 1-5: unfreeze one more layer per epoch, top-down (11, 10, 9, 8, 7)
# Epoch 6: layers 6-11 all trainable (matches partial_freeze's static target)
# Epoch 7: stays at full depth -- second full epoch to let it consolidate
UNFREEZE_SCHEDULE = [NUM_ENCODER_LAYERS, 11, 10, 9, 8, 7, 6, 6]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "models_out", "call_classifier_gradual_unfreeze_v2")


def configure(model):
    # IMPORTANT: register the FULL eventual trainable range (layers 6-11) with
    # requires_grad=True *before* the Trainer builds its optimizer, since HF's
    # Trainer filters params by requires_grad only once, at optimizer-creation
    # time. If we only registered the head here, layers unfrozen later by the
    # callback would compute gradients but never actually get updated -- they
    # wouldn't be in any optimizer param group. The callback below then
    # re-freezes/re-thaws within this already-registered set each epoch,
    # which *is* dynamic and safe (optimizer just skips params with grad=None).
    set_trainable_layers(model, min(UNFREEZE_SCHEDULE))


def extra_callbacks(model):
    return [GradualUnfreezeCallback(model, UNFREEZE_SCHEDULE)]


if __name__ == "__main__":
    import sys
    resume = sys.argv[1] if len(sys.argv) > 1 else None
    run_experiment("gradual_unfreeze_v2", OUT_DIR, configure, extra_callbacks_fn=extra_callbacks,
                    epochs=len(UNFREEZE_SCHEDULE), resume_from_checkpoint=resume)
