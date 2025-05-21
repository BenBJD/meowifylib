# Mostly from https://docs.pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html#sphx-glr-tutorials-hybrid-demucs-tutorial-py
import torch
import torchaudio.transforms as T
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

torch.cuda.empty_cache()
bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
sample_rate = bundle.sample_rate


def separate_sources(model, mix, segment=10.0, overlap=0.1, device=None):
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)
    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = T.Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.inference_mode():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final


def extract_vocals(waveform, sample_rate):
    # If the audio is mono, convert it to stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    # If the sample rate is not 44100, resample
    if sample_rate != 44100:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=44100)
        waveform = resampler(waveform)

    waveform = waveform.to(device)
    segment: int = 10
    overlap = 0.1
    print("Separating sources...")

    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()

    sources = separate_sources(
        model,
        waveform[None],
        device=device,
        segment=segment,
        overlap=overlap,
    )[0]

    sources = sources * ref.std() + ref.mean()

    sources_list = model.sources
    sources = list(sources)

    audios = dict(zip(sources_list, sources))
    vocals_waveform = audios["vocals"]
    # Mix the other sources
    other_sources_waveform = sum(
        audios[source] for source in audios if source != "vocals"
    )
    del audios
    resampler = T.Resample(orig_freq=44100, new_freq=sample_rate)
    vocals_waveform = resampler(vocals_waveform.cpu())
    other_sources_waveform = resampler(other_sources_waveform.cpu())

    return vocals_waveform, other_sources_waveform
