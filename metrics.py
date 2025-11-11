from os.path import dirname, join
import numpy as np
import sofar
import torch
import os

ROOT_DIR = "data/"

class MeanSpectralDistortion:
    """
    Metric Class used for evaluation, can also be used as loss function.
    MeanSpectalDistortion().get_spectral_distortion(ground_truth, predicted) for calculating error.
    """

    def __init__(self):

        self.avg_hrir = sofar.read_sofa(os.path.join(ROOT_DIR, "Average_HRTFs.sofa"), verbose=False)
        self.source_positions = self.avg_hrir.SourcePosition
        self.elevation_index = self._get_elevation_index()
        self.weights = self._get_weights()

    def _get_weights(self):
        """
        This function load the weights which are used when you calculate the spectral distortion/ baseline predictions
        weights were calculated based on the paper "Looking for a relevant similarity criterion fo HRTF clustering: a comparative study - Rozenn Nicol".

        Returns:
               normalized_weights: torch.tensor

        """
        # Generate a list of frequencies up to 24 kHz
        frequencies_Hz = np.linspace(0, 24000, 129)  # 129 points between 0 Hz and 24 kHz
        frequencies_kHz = frequencies_Hz / 1000
        inv_cb = 1 / (25 + 75 * (1 + 1.4 * frequencies_kHz**2) ** 0.69)  # inverse of delta (critical bandwidth)
        a0 = sum(inv_cb)
        normalized_weights = inv_cb / a0
        return normalized_weights

    def _get_elevation_index(self):
        """
        Helper function to get elevation indexes.
        Args:
            You can change the elevation range as required. We will use the elvation range between -30 to 30
            Returns:
             all index for the elevation range"""
        # this function gives the index of the directions for which you need to evaluate your results.

        azimuths = self.source_positions[:, 0]
        elevations = self.source_positions[:, 1]

        # Define the elevation range
        elevation_min = -30
        elevation_max = 30
        # Find the indices for the specific elevation range
        elevation_indices = np.where((elevations >= elevation_min) & (elevations <= elevation_max))[0]

        # Ensure that elevation_indices is a NumPy array of integers
        return np.array(elevation_indices, dtype=int)

    def get_spectral_distortion(self, hrtf_ground_truth: torch.Tensor, hrtf_predicted: torch.Tensor) -> torch.Tensor:
        """
        Computes the spectral distortion between the inputs.

        Args:
            hrtf_ground_truth: torch.tensor
            hrtf_predicted: torch.tensor
        Returns:
            weighted_error: torch.tensor in dB

        """

        weighted_error = ((torch.from_numpy(self.weights) * (hrtf_ground_truth[:, self.elevation_index].abs() - hrtf_predicted[:, self.elevation_index].abs())) ** 2).mean()
        return weighted_error.log10() * 10

    def get_spectral_distortion_numpy(self, hrtf_ground_truth: np.ndarray, hrtf_predicted: np.ndarray) -> float:
        """
        Computes the spectral distortion between the inputs.

        Args:
            hrtf_ground_truth: np.ndarray
            hrtf_predicted: np.ndarray
        Returns:
            weighted_error: float
        """

        weighted_error = np.mean(self.weights * (np.abs(hrtf_ground_truth[:, self.elevation_index]) - np.abs(hrtf_predicted[:, self.elevation_index])) ** 2)
        return float(10. * np.log10(weighted_error))