import librosa

# This decides horizontal side of each 'image'
FRAMES_PER_IMAGE = 512
# Number of bins starting at fmin for cqt. 120 / 2 = 60 semitones
N_BINS = 120
# Bins per semitone * semitones in octave
BINS_PER_OCTAVE = 24
# Min frequency for cqt.
FMIN = librosa.note_to_hz("C2")
# Time resolution in cqt
HOP_LENGTH = 128
# What sample rate to use for songs
SAMPLE_RATE = 16000
