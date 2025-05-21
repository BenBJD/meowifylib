# MeowifyLib

A Python library for audio processing and vocal-to-MIDI conversion.

## Description

MeowifyLib provides tools for audio processing, source separation, and vocal-to-MIDI conversion using neural networks. It can be used to:

- Extract vocals from mixed audio tracks
- Convert vocal recordings to MIDI
- Generate cat-like sounds from MIDI data
- Mix audio tracks together

## Installation

```bash
pip install meowifylib
```

## Usage

### Extract vocals from a mixed audio track

```python
import torchaudio
from meowifylib import extract_vocals

# Load audio file
waveform, sample_rate = torchaudio.load("input.wav")

# Extract vocals and accompaniment
vocals, accompaniment = extract_vocals(waveform, sample_rate)

# Save the separated tracks
torchaudio.save("vocals.wav", vocals, sample_rate)
torchaudio.save("accompaniment.wav", accompaniment, sample_rate)
```

### Convert vocals to MIDI

```python
import torch
import lightning as L
from meowifylib import MeowifyVocal2MIDINet, make_predict_dataset, tensor_to_midi

# Load the model
model = MeowifyVocal2MIDINet.load_from_checkpoint("trained.ckpt")
model.eval()
trainer = L.Trainer(enable_progress_bar=False)

# Create dataset from vocal track
dataset = make_predict_dataset("vocals.wav", "song_name")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Generate predictions
predictions = trainer.predict(model, dataloader)
full_output = torch.cat(predictions, dim=2).squeeze(0)

# Process the output
processed_notes = full_output > 0.5

# Convert to MIDI
midi_file = tensor_to_midi(processed_notes, dataset.num_samples)
```

### Generate audio from MIDI

```python
from meowifylib import midi_to_audio, SAMPLE_RATE

# Define sample choices (cat sounds)
sample_choices = [
    {"name": "cat_a3.wav", "pitch": 57},  # A3
    {"name": "cat_c4.wav", "pitch": 60},  # C4
    # Add more samples as needed
]

# Generate audio from MIDI
new_vocals = midi_to_audio(
    midi_file,
    sample_choices,
    sample_rate=SAMPLE_RATE,
    sample_length_min_secs=0.5
)
```

### Mix tracks together

```python
from meowifylib import mix_tracks

# Mix the new vocals with the original accompaniment
mixed_audio = mix_tracks(
    "accompaniment.wav",
    new_vocals,
    accomp_volume=0.7,
    vocals_volume=1.3
)

# Save the final mix
import soundfile as sf
sf.write("output.wav", mixed_audio, SAMPLE_RATE)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.