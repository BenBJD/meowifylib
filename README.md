# MeowifyLib

A Python library for audio processing and vocal-to-MIDI conversion with Meowify.

> [!IMPORTANT]
> A pretrained checkpoint is not yet available.

## Description
MeowifyLib provides the code behind Meowify. Features include:
- Running the full pipeline with a saved model checkpoint (`meowifylib.run.meowify_song())
- The audio and MIDI processing used throughout the system
- The whole neural network source code
- Audio separation functions using `torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS`
- Evaluation metrics from `mir_eval` to compare against Basic Pitch and pYin

## Installation

```bash
uv add "meowifylib @ git+https://github.com/BenBJD/meowifylib"
```
or
```bash
pip install git+https://github.com/BenBJD/meowifylib
```

## Usage

> [!NOTE]
> Very incomplete

### Run Meowify pipeline on a track

```python
import os
import soundfile as sf
import librosa
from meowifylib.run import meowify_song

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Assign as many pitches as you want.
# If no sample is available at a pitch, the closest will be pitch shifted
sample_choices = [
    {"name": "samples/a3.wav", "pitch": librosa.note_to_midi("A3")},
    {"name": "samples/a4.wav", "pitch": librosa.note_to_midi("A4")},
    {"name": "samples/b3.wav", "pitch": librosa.note_to_midi("B3")},
    {"name": "samples/b4.wav", "pitch": librosa.note_to_midi("B4")},
    {"name": "samples/c4.wav", "pitch": librosa.note_to_midi("C4")},
    {"name": "samples/c5.wav", "pitch": librosa.note_to_midi("C5")},
    {"name": "samples/d4.wav", "pitch": librosa.note_to_midi("D4")},
    {"name": "samples/d5.wav", "pitch": librosa.note_to_midi("D5")},
    {"name": "samples/e4.wav", "pitch": librosa.note_to_midi("E4")},
    {"name": "samples/e5.wav", "pitch": librosa.note_to_midi("E5")},
    {"name": "samples/f4.wav", "pitch": librosa.note_to_midi("F4")},
]

output = meowify_song("input_song.wav", sample_choices, "trained.ckpt")

# Save the final mix and midi
sf.write(f"output_song.wav", output, 22050)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
