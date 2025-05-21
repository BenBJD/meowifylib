import numpy as np
import librosa
import mir_eval.transcription
from meowifylib.constants import SAMPLE_RATE, HOP_LENGTH, FMIN


def calculate_weighted_accuracy(predictions, targets):
    # Flatten the tensors to compare all elements
    predictions_flat = predictions.flatten()
    targets_flat = targets.flatten()

    # Calculate true positives, true negatives, false positives, and false negatives
    true_positives = ((predictions_flat == 1) & (targets_flat == 1)).sum().item()
    true_negatives = ((predictions_flat == 0) & (targets_flat == 0)).sum().item()
    false_positives = ((predictions_flat == 1) & (targets_flat == 0)).sum().item()
    false_negatives = ((predictions_flat == 0) & (targets_flat == 1)).sum().item()

    # Calculate the total number of positives and negatives
    total_positives = true_positives + false_negatives
    total_negatives = true_negatives + false_positives

    # Calculate weighted accuracy
    if total_positives == 0:
        positive_accuracy = 1.0
    else:
        positive_accuracy = true_positives / total_positives

    if total_negatives == 0:
        negative_accuracy = 1.0
    else:
        negative_accuracy = true_negatives / total_negatives

    weighted_accuracy = (positive_accuracy + negative_accuracy) / 2
    return weighted_accuracy


def pianoroll_to_notes(pianoroll, fs=SAMPLE_RATE / HOP_LENGTH):
    """Convert a piano roll to a list of note events (onset, offset, pitch).

    Args:
        pianoroll (numpy.ndarray): Piano roll matrix of shape (num_pitches, num_frames)
        fs (float, optional): Frame rate in Hz. Defaults to SAMPLE_RATE / HOP_LENGTH.

    Returns:
        numpy array of note events, each row containing (onset_time, offset_time, pitch)
    """
    notes = []
    for pitch in range(pianoroll.shape[0]):
        # Find all note onsets and offsets
        diff = np.diff(pianoroll[pitch].astype(int), prepend=0)
        onsets = np.where(diff > 0)[0]
        offsets = np.where(diff < 0)[0]

        # If there are more onsets than offsets, add an offset at the end
        if len(onsets) > len(offsets):
            offsets = np.append(offsets, pianoroll.shape[1])

        # Create note events
        for i in range(len(onsets)):
            if i < len(offsets):
                onset_time = onsets[i] / fs
                offset_time = offsets[i] / fs
                # Only include notes with a minimum duration
                if offset_time > onset_time:
                    notes.append(
                        (onset_time, offset_time, pitch + int(librosa.hz_to_midi(FMIN)))
                    )

    return np.array(notes) if notes else np.empty((0, 3))


def evaluate_transcription(
    gt_pianoroll, pred_pianoroll, onset_tolerance=0.05, offset_ratio=0.2
):
    """Evaluate transcription using mir_eval.

    Args:
        gt_pianoroll: Ground truth piano roll
        pred_pianoroll: Predicted piano roll
        onset_tolerance: Tolerance for onset in seconds. Defaults to 0.05.
        offset_ratio: Tolerance for offset as a ratio of the note length. Defaults to 0.2.

    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Convert to numpy if needed
    if hasattr(gt_pianoroll, "cpu"):
        gt_pianoroll = gt_pianoroll.cpu().numpy()
    if hasattr(pred_pianoroll, "cpu"):
        pred_pianoroll = pred_pianoroll.cpu().numpy()

    # Convert piano rolls to note events
    gt_notes = pianoroll_to_notes(gt_pianoroll)
    pred_notes = pianoroll_to_notes(pred_pianoroll)

    # If either set of notes is empty, return zeros for all metrics
    if len(gt_notes) == 0 or len(pred_notes) == 0:
        return {
            "Precision": 0.0,
            "Recall": 0.0,
            "F-measure": 0.0,
            "Average_Overlap_Ratio": 0.0,
            "Onset_Precision": 0.0,
            "Onset_Recall": 0.0,
            "Onset_F-measure": 0.0,
            "Offset_Precision": 0.0,
            "Offset_Recall": 0.0,
            "Offset_F-measure": 0.0,
        }

    # Extract intervals and pitches
    ref_intervals = gt_notes[:, :2]
    ref_pitches = gt_notes[:, 2]

    est_intervals = pred_notes[:, :2]
    est_pitches = pred_notes[:, 2]

    # Sort by onset time
    ref_sort_idx = np.argsort(ref_intervals[:, 0])
    ref_intervals = ref_intervals[ref_sort_idx]
    ref_pitches = ref_pitches[ref_sort_idx]

    est_sort_idx = np.argsort(est_intervals[:, 0])
    est_intervals = est_intervals[est_sort_idx]
    est_pitches = est_pitches[est_sort_idx]

    # Calculate all metrics using mir_eval
    scores = mir_eval.transcription.evaluate(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio,
    )

    return scores
