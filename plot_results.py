"""Overlay a predicted HRTF against the measured ground truth for one subject."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import sofar
import torchvision.transforms as transforms

from utils import SonicomDatabase

NFFT = 256
SAMPLING_RATE_HZ = 48000


def plot_prediction_for_id(
    subject_id: str,
    predicted_sofa_path: str,
    root_dir: str,
    transforms: transforms.Compose,
    task_id: int = 1,
):
    """
    Plot the direction-averaged magnitude response of a predicted HRTF against the
    measured one for a given subject.

    Args:
        subject_id (str): Subject ID, e.g. 'P0004'.
        predicted_sofa_path (str): SOFA file written by inference.py for this subject.
        root_dir (str): Dataset root directory.
        transforms (transforms.Compose): Preprocessing pipeline for the pinna images.
        task_id (int): Which image subset the prediction was made from (0, 1 or 2).
    """
    dataset = SonicomDatabase(
        root_dir=root_dir,
        training_data=False,
        transforms=transforms,
        task_id=task_id,
    )

    if subject_id not in dataset.all_subjects:
        raise SystemExit(f"Subject {subject_id} not found in {root_dir}")

    predicted_hrir = sofar.read_sofa(predicted_sofa_path, verbose=False).Data_IR
    predicted_hrtf = np.fft.rfft(predicted_hrir, n=NFFT)

    _, reference_hrtf = dataset[dataset.all_subjects.index(subject_id)]

    # Average the magnitude response over all directions and both ears.
    predicted_avg = np.abs(predicted_hrtf).mean(axis=(0, 1))
    reference_avg = np.abs(np.asarray(reference_hrtf)).mean(axis=(0, 1))

    frequency_axis = np.linspace(0, SAMPLING_RATE_HZ / 2, reference_avg.shape[-1]) / 1000

    plt.figure(figsize=(12, 6))
    plt.plot(frequency_axis, 20 * np.log10(reference_avg), label="measured")
    plt.plot(frequency_axis, 20 * np.log10(predicted_avg), label="predicted")
    plt.title(f"Direction-averaged HRTF magnitude — subject {subject_id} (task {task_id})")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--subject", default="P0004", help="Subject ID, e.g. P0004")
    parser.add_argument("-p", "--predicted", default="predicted_hrtf_P0004.sofa",
                        help="Predicted SOFA file for this subject")
    parser.add_argument("--root_dir", default="data/", help="Dataset root directory")
    parser.add_argument("-t", "--task_id", type=int, default=2, choices=[0, 1, 2],
                        help="Image subset used for the prediction")
    args = parser.parse_args()

    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    plot_prediction_for_id(
        subject_id=args.subject,
        predicted_sofa_path=args.predicted,
        root_dir=args.root_dir,
        transforms=val_transforms,
        task_id=args.task_id,
    )
