"""
Meowify Library - A package for audio processing and vocal-to-MIDI conversion.

This package provides tools for audio processing, source separation, and
vocal-to-MIDI conversion using neural networks.
"""

# Version information
__version__ = "0.1.0"

# Import key components for easier access
from meowifylib.constants import (
    FMIN,
    SAMPLE_RATE,
    HOP_LENGTH,
    N_BINS,
    BINS_PER_OCTAVE,
    FRAMES_PER_IMAGE,
)
from meowifylib.neural import MeowifyVocal2MIDINet
from meowifylib.processing import (
    tensor_to_midi,
    midi_to_audio,
    mix_tracks,
    make_predict_dataset,
    make_train_dataset,
)
from meowifylib.separate import extract_vocals
from meowifylib.eval import evaluate_transcription, calculate_weighted_accuracy
from meowify.run import meowify_song
