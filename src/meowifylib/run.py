import os
import shutil
import tempfile

import librosa
import lightning as L
import torch
import torchaudio
from torch.utils.data import DataLoader

from meowifylib.constants import (
    FMIN,
    SAMPLE_RATE,
)
from meowifylib.neural import MeowifyVocal2MIDINet
from meowifylib.processing import (
    tensor_to_midi,
    midi_to_audio,
    mix_tracks,
    make_predict_dataset,
)
from meowifylib.separate import extract_vocals

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")

def meowify_song(song_path, sample_choices, checkpoint_path):
    """
    Process a song by extracting vocals, converting to MIDI, and generating new vocals with samples.

    Args:
        song_path (str): Path to the song file without the .wav extension
        sample_choices (list): List of dictionaries containing sample paths with their pitches
        checkpoint_path (str): Path to the checkpoint file for the Meowify model.

    Returns:
        numpy.ndarray: The new vocals mixed with the original accompaniment
    """
    # Extract track name from song path
    track_name = os.path.basename(song_path)

    # Create a temporary directory for storing split files
    temp_dir = tempfile.mkdtemp()

    try:
        # Set up Meowify model
        model = MeowifyVocal2MIDINet.load_from_checkpoint(checkpoint_path)
        model.eval()
        trainer = L.Trainer(enable_progress_bar=False)

        # Split and save
        waveform, sr = torchaudio.load(f"{song_path}.wav")
        vocals, accompaniment = extract_vocals(waveform, sr)

        # Create an output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)

        # Save vocals and accompaniment to temporary directory
        vocals_path = os.path.join(temp_dir, f"{track_name}.vocals.wav")
        accomp_path = os.path.join(temp_dir, f"{track_name}.accomp.wav")
        torchaudio.save(vocals_path, vocals, sr)
        torchaudio.save(accomp_path, accompaniment, sr)

        print("\n")

        song_dataset = make_predict_dataset(
            vocals_path, track_name
        )

        song_dataloader = DataLoader(
            song_dataset, batch_size=1, shuffle=False, num_workers=3
        )
        original_samples_length = song_dataset.num_samples
        padding = song_dataset.padding

        # Generate predictions
        test_output = trainer.predict(model, song_dataloader)
        predictions = []
        for output in test_output:
            predictions.append(output)
        full_test_output = torch.cat(predictions, dim=2).squeeze(0)
        full_test_output = full_test_output[:, :-padding]

        # Normalize
        full_test_output = (full_test_output - full_test_output.min()) / (
            full_test_output.max() - full_test_output.min()
        )

        processed_notes = full_test_output > 0.5

        # Pad to full 128 notes
        full_notes = torch.nn.functional.pad(
            processed_notes,
            (
                0,
                0,
                int(librosa.hz_to_midi(FMIN)),
                int(128 - librosa.hz_to_midi(FMIN) - processed_notes.shape[0]),
            ),
            mode="constant",
            value=0,
        )

        # Make a midi file
        midi_file = tensor_to_midi(full_notes, original_samples_length)
        # Gen the new vocals
        new_vocals = midi_to_audio(
            midi_file, sample_choices, sample_rate=SAMPLE_RATE,
            sample_length_min_secs=0.5
        )

        # Mix the new vocals with the original audio
        mixed_audio = mix_tracks(
            accomp_path,
            new_vocals,
            accomp_volume=1.0,
            vocals_volume=1.0,
        )

        # Return the processed audio
        return mixed_audio
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
