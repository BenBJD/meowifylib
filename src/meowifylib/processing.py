import random

import librosa
import mido
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.nn import functional as F
from torch.utils.data import Dataset
from hmmlearn import hmm
from matplotlib import pyplot as plt
import librosa.display as librosadisplay

from meowifylib.constants import (
    SAMPLE_RATE,
    HOP_LENGTH,
    N_BINS,
    BINS_PER_OCTAVE,
    FMIN,
    FRAMES_PER_IMAGE,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset for training
class Vocal2MIDIDataset(Dataset):
    def __init__(self, cqts, midis):
        self.cqts = cqts
        self.midis = midis

    def __len__(self):
        return len(self.cqts)

    def __getitem__(self, idx):
        return self.cqts[idx], self.midis[idx]


# Dataset for predictions only (no labels)
class Vocal2MIDIPredictDataset(Dataset):
    def __init__(self, cqts, num_samples, name="", padding=0):
        self.cqts = cqts
        self.num_samples = num_samples
        self.name = name
        self.padding = padding

    def __len__(self):
        return len(self.cqts)

    def __getitem__(self, idx):
        return self.cqts[idx]


# Generate a CQT from a waveform
def generate_spectrogram(waveform):
    # Convert to mono
    mono_waveform = waveform.mean(dim=0, keepdim=True).squeeze()

    vocals_cqt = librosa.hybrid_cqt(
        mono_waveform.cpu().numpy(),
        hop_length=HOP_LENGTH,
        sr=SAMPLE_RATE,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
        fmin=FMIN,
    )
    vocals_cqt = np.abs(vocals_cqt)

    # Convert to dB and normalize between 0 and 1
    vocals_cqt_db = librosa.amplitude_to_db(vocals_cqt, ref=np.max)
    vocals_cqt_db = torch.Tensor(vocals_cqt_db).to(device)
    vocals_cqt_db = (vocals_cqt_db - vocals_cqt_db.min()) / (
        vocals_cqt_db.max() - vocals_cqt_db.min()
    )

    # Power scaling increases the difference between low and high loudness
    p = 2
    vocals_cqt_db = vocals_cqt_db**p

    # Experimemnt with hpss again
    H, _ = librosa.decompose.hpss(vocals_cqt_db.cpu().numpy())

    return torch.tensor(H, device=device)
    # return vocals_cqt_db


# Make a piano roll tensor from a midi file
def midi_file_to_tensor(track_path, cqt_length, waveform_seconds=None):
    """Convert the MIDI file to tensor, aligning it with cqt frames."""
    midi = mido.MidiFile(track_path)
    merged_track = mido.merge_tracks(midi.tracks)

    # Find the total number of ticks in a midi file
    total_ticks = 0
    for msg in merged_track:
        total_ticks += msg.time

    # Initialise
    midi_notes = torch.zeros(128, total_ticks, dtype=torch.float32)

    # Parse MIDI events
    time_in_ticks = 0
    note_events = []  # Store note start and end events
    for msg in merged_track:
        time_in_ticks += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            note_events.append((msg.note, time_in_ticks, 1))  # Note-on
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            note_events.append((msg.note, time_in_ticks, 0))  # Note-off

    # If waveform_seconds was passed in, calculate how much to pad.
    # This is so the midi track doesnt end at the end of the last note, but instead the end of the song
    if not waveform_seconds is None:
        padding_seconds = waveform_seconds - midi.length
        if padding_seconds > 0.001:
            BPM = 0
            for msg in merged_track:
                if msg.type == "set_tempo":
                    BPM = msg.tempo
            padding_ticks = mido.second2tick(padding_seconds, midi.ticks_per_beat, BPM)
            # Add the padding
            midi_notes = torch.cat((midi_notes, torch.zeros(128, padding_ticks)), dim=1)
            total_ticks += padding_ticks

    # Fill the MIDI tensor between note_ons and note_offs
    active_notes = set()
    current_tick = 0
    for note, event_tick, state in note_events:
        # Fill up to the current event time
        while current_tick < event_tick:
            for active_note in active_notes:
                midi_notes[active_note, current_tick] = 1
            current_tick += 1

        # Update active notes
        if state == 1:  # Note-on
            active_notes.add(note)
        elif state == 0:  # Note-off
            active_notes.discard(note)

    # # Fill remaining active notes to handle anything weird
    while current_tick < total_ticks:
        for active_note in active_notes:
            midi_notes[active_note, current_tick] = 1
        current_tick += 1

    # Interpolate using torch.nn.functional.interpolate
    midi_notes_expanded = midi_notes.unsqueeze(0).unsqueeze(
        0
    )  # Add batch and channel dimensions
    scaled_midi_notes_interpolated = torch.nn.functional.interpolate(
        midi_notes_expanded, size=(128, cqt_length), mode="nearest-exact"
    )
    scaled_midi_notes = scaled_midi_notes_interpolated.squeeze(0).squeeze(0)

    return scaled_midi_notes


# For data augmentation
def pitch_shift(waveform, midi_tensor, n_steps):
    waveform_shifted = librosa.effects.pitch_shift(
        waveform.cpu().numpy(), sr=SAMPLE_RATE, n_steps=n_steps
    )
    midi_tensor_shifted = torch.roll(midi_tensor, shifts=n_steps, dims=0)
    return torch.tensor(waveform_shifted), midi_tensor_shifted


# Turn the model output back into a MIDI file
def tensor_to_midi(midi_tensor, original_samples_length):
    midi_file = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    # Default tempo
    tempo = mido.bpm2tempo(120)
    runtime_seconds = original_samples_length / SAMPLE_RATE
    total_ticks = int(
        mido.second2tick(runtime_seconds, midi_file.ticks_per_beat, tempo)
    )

    # Set instrument
    track.append(mido.Message("program_change", program=0, time=0))

    # Resample to number of ticks
    scaled_notes = (
        F.interpolate(
            midi_tensor.float().unsqueeze(0),
            size=total_ticks,
            mode="linear",
            align_corners=False,
        )
        .squeeze(0)
        .round()
        .int()
        .cpu()
        .numpy()
    )  # shape: [128, total_ticks]

    # Transpose to [time, notes]
    scaled_notes = scaled_notes.T

    # Track note states
    note_states = np.zeros(128, dtype=bool)
    last_tick = 0

    # Vectorized processing of note events
    for tick, notes in enumerate(scaled_notes):
        changes = notes != note_states
        if np.any(changes):
            for note in np.where(changes)[0]:
                velocity = 64
                msg_type = "note_on" if notes[note] else "note_off"
                track.append(
                    mido.Message(
                        msg_type, note=note, velocity=velocity, time=(tick - last_tick)
                    )
                )
                last_tick = tick
            note_states = notes.copy()

    # Ensure proper ending
    remaining_time = total_ticks - last_tick
    if remaining_time > 0:
        track.append(mido.MetaMessage("end_of_track", time=remaining_time))
    else:
        track.append(mido.MetaMessage("end_of_track", time=0))

    return midi_file


# Generate an audio track from midi and samples
def midi_to_audio(
    midi,
    samples,
    sample_rate=SAMPLE_RATE,
    sample_length_min_secs=0.2,
    fade_duration=0.02,
):
    # Load the samples into a dictionary {pitch: (audio, sr)}
    sample_data = {}
    for sample_info in samples:
        sample_audio, sr = librosa.load(sample_info["name"], sr=sample_rate)
        sample_audio = sample_audio / np.max(np.abs(sample_audio))
        sample_data[sample_info["pitch"]] = sample_audio

    # Convert MIDI ticks to time (120 BPM default)
    tempo = 500000  # Microseconds per beat (default = 120 BPM)
    tick_to_sec = lambda ticks: mido.tick2second(ticks, midi.ticks_per_beat, tempo)

    # Process MIDI events
    active_notes = {}  # {note: (start_time, velocity)}
    events = []  # (start_time, end_time, note)

    current_time = 0
    for msg in midi.tracks[0]:
        current_time += tick_to_sec(msg.time)  # Update time in seconds
        if msg.type == "note_on" and msg.velocity > 0:
            active_notes[msg.note] = (current_time, msg.velocity)
        elif msg.type == "note_off":
            if msg.note in active_notes:
                start_time, velocity = active_notes.pop(msg.note)
                events.append((start_time, current_time, msg.note))

    total_duration = midi.length
    audio_length = int(total_duration * sample_rate)
    output_audio = np.zeros(audio_length)

    fade_samples = int(fade_duration * sample_rate)  # Number of samples for fade-in/out

    # Synthesize audio
    for start_time, end_time, note in events:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Find the closest sample pitch
        closest_pitch = min(sample_data.keys(), key=lambda p: abs(p - note))
        sample = sample_data[closest_pitch]

        # Pitch shift the sample to match the MIDI note
        note_sample = librosa.effects.pitch_shift(
            sample, sr=sample_rate, n_steps=note - closest_pitch
        )

        # Stretch sample to fit note duration
        target_length = max(
            end_sample - start_sample, round(sample_rate * sample_length_min_secs)
        )
        scale_factor = len(sample) / target_length
        note_sample = librosa.effects.time_stretch(note_sample, rate=scale_factor)

        # Apply fade-in and fade-out
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)

        if len(note_sample) > 2 * fade_samples:
            note_sample[:fade_samples] *= fade_in  # Apply fade-in
            note_sample[-fade_samples:] *= fade_out  # Apply fade-out

        # Place in output buffer
        end_sample = min(start_sample + len(note_sample), len(output_audio))
        output_audio[start_sample:end_sample] += note_sample[
            : end_sample - start_sample
        ]

    # Normalize and return
    output_audio = output_audio / np.max(np.abs(output_audio))
    return output_audio


def mix_tracks(track_name, new_vocals, accomp_volume=1.0, vocals_volume=1.0):
    # Load accompaniment and new vocals
    accompaniment, sr1 = librosa.load(track_name, sr=22050)

    new_vocals = librosa.resample(new_vocals, orig_sr=SAMPLE_RATE, target_sr=22050)

    # Scale the shorter track to match the longer track
    len_acc, len_vocals = len(accompaniment), len(new_vocals)
    max_length = max(len_acc, len_vocals)

    if len_acc < max_length:
        accompaniment = librosa.effects.time_stretch(
            accompaniment, rate=len_acc / max_length
        )
    if len_vocals < max_length:
        new_vocals = librosa.effects.time_stretch(
            new_vocals, rate=len_vocals / max_length
        )

    # Apply volume control
    accompaniment *= accomp_volume
    new_vocals *= vocals_volume

    # Mix tracks together
    mixed_audio = accompaniment + new_vocals

    # Normalize to prevent clipping
    mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))

    return mixed_audio


# Process wav files and midi labels to make a train dataset
def make_train_dataset(wav_path, midi_path, augment=True):
    waveform, sr = torchaudio.load(wav_path)
    resample = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
    waveform = resample(waveform)
    cqt = generate_spectrogram(waveform)

    # Generate a midi tensor
    midi_tensor = midi_file_to_tensor(midi_path, cqt.shape[1])

    # Put MIDI to CQT range
    midi_tensor = midi_tensor[
        librosa.note_to_midi("C2") : librosa.note_to_midi("C2") + 60, :
    ]

    # Display overlaid

    midi_chunks = []
    cqt_chunks = []

    def split(split_cqt, split_midi):
        num_images = split_cqt.shape[-1] // FRAMES_PER_IMAGE
        for j in range(num_images):
            start = j * FRAMES_PER_IMAGE
            end = (j + 1) * FRAMES_PER_IMAGE
            cqt_chunk = split_cqt[:, start:end]
            midi_chunk = split_midi[:, start:end]

            # Add some padding to start and end of chunk. Only for training data?
            # cqt_chunk = torch.nn.functional.pad(cqt_chunk, (FRAMES_PER_IMAGE // 10, FRAMES_PER_IMAGE // 10))
            # midi_chunk = torch.nn.functional.pad(midi_chunk, (FRAMES_PER_IMAGE // 10, FRAMES_PER_IMAGE // 10))

            cqt_chunks.append(cqt_chunk.cpu())
            midi_chunks.append(midi_chunk.cpu())

    display_midi_and_cqt(midi_tensor, cqt)
    split(cqt, midi_tensor)

    if augment:
        # Apply pitch shift data augmentation
        waveform_shifted, midi_shifted = pitch_shift(
            waveform, midi_tensor, n_steps=random.randint(-2, 8)
        )
        cqt_shifted = generate_spectrogram(waveform_shifted)
        waveform_shifted2, midi_shifted2 = pitch_shift(
            waveform, midi_tensor, n_steps=random.randint(-2, 8)
        )
        cqt_shifted2 = generate_spectrogram(waveform_shifted2)
        waveform_shifted3, midi_shifted3 = pitch_shift(
            waveform, midi_tensor, n_steps=random.randint(-2, 8)
        )
        cqt_shifted3 = generate_spectrogram(waveform_shifted3)

        # Split and add all augmented data
        split(cqt_shifted, midi_shifted)
        split(cqt_shifted2, midi_shifted2)
        split(cqt_shifted3, midi_shifted3)

    return Vocal2MIDIDataset(cqt_chunks, midi_chunks)


# Process a wav file to make a prediction dataset
def make_predict_dataset(wav_path, song_name):
    waveform, sr = torchaudio.load(wav_path)
    resample = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
    waveform = resample(waveform)
    waveform_num_samples = waveform.shape[-1]
    cqt = generate_spectrogram(waveform)
    del waveform

    cqt_chunks = []

    # Split spectrogram and midi tensor into tensors of size FRAMES_PER_IMAGE
    num_images = cqt.shape[-1] // FRAMES_PER_IMAGE
    for j in range(num_images):
        start = j * FRAMES_PER_IMAGE
        end = (j + 1) * FRAMES_PER_IMAGE
        cqt_chunk = cqt[:, start:end]
        cqt_chunks.append(cqt_chunk.cpu())
    # Pad the last chunk
    if cqt.shape[-1] - (num_images * FRAMES_PER_IMAGE) > 0:
        start = num_images * FRAMES_PER_IMAGE
        end = cqt.shape[-1]
        cqt_chunk = cqt[:, start:end]
        padding = FRAMES_PER_IMAGE - cqt_chunk.shape[-1]
        cqt_chunk = torch.cat(
            [cqt_chunk, torch.zeros(size=(cqt.shape[0], padding), device=device)],
            dim=-1,
        )
        cqt_chunks.append(cqt_chunk.cpu())
    else:
        padding = 0
    return Vocal2MIDIPredictDataset(
        cqt_chunks, waveform_num_samples, song_name, padding=padding
    )


# Unused
def apply_hmm(predictions, transition_prob=0.8):
    batch_size, num_notes, num_frames = predictions.shape
    smoothed = torch.zeros_like(predictions)

    for b in range(batch_size):
        for n in range(num_notes):
            # Define the HMM model
            model = hmm.GaussianHMM(
                n_components=2, covariance_type="diag", init_params=""
            )
            model.startprob_ = np.array([1 - transition_prob, transition_prob])
            model.transmat_ = np.array(
                [
                    [transition_prob, 1 - transition_prob],
                    [1 - transition_prob, transition_prob],
                ]
            )
            # Means and covariances manually
            model.means_ = np.array([[0.0], [1.0]])
            model.covars_ = np.array([[0.1], [0.1]])  # Small variance
            # Fit HMM to the note probabilities over time
            note_probs = predictions[b, n, :].cpu().numpy().reshape(-1, 1)
            states = model.predict(note_probs)
            smoothed[b, n, :] = torch.tensor(states)
    return smoothed


def display_midi_and_cqt(midi_tensor, cqt):
    plt.figure(figsize=(50, 10))
    librosadisplay.specshow(
        cqt.cpu().numpy(),
        sr=SAMPLE_RATE,
        bins_per_octave=BINS_PER_OCTAVE,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        y_axis="cqt_note",
        x_axis="time",
    )
    librosadisplay.specshow(
        midi_tensor.cpu().numpy(),
        sr=SAMPLE_RATE,
        bins_per_octave=12,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        y_axis="cqt_note",
        x_axis="time",
        cmap="binary",
        alpha=0.5,
    )
    plt.show()
