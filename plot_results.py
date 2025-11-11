import random  # Ensure random is imported
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Tuple
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from utils import SonicomDatabase
import sofar

epsilon = 1e-10

def plot_prediction_for_id(
    patient_id: str,
    root_dir: str,
    transforms: transforms.Compose,
    task_id: int = 1
):
    """
    Generates and displays a comparison plot between predicted and reference HRTFs for a specific patient.

    Args:
        patient_id (str): The unique ID of the patient (e.g., 'P0002').
        model (nn.Module): Trained HRTF prediction model.
        root_dir (str): Root directory where the dataset is located.
        device (torch.device): Device to perform computations on ('cuda' or 'cpu').
        transforms (transforms.Compose): Transformation pipeline to preprocess images.
        task_id (int, optional): Task identifier determining the image subset. Defaults to 1.
    """
    # Initialize the dataset in evaluation mode (training_data=False)
    dataset = SonicomDatabase(
        root_dir=root_dir,
        training_data=False,
        transforms=transforms,
        task_id=task_id
    )
    predicted_sofa = sofar.read_sofa("predicted_hrtf_P0004.sofa", verbose=False).Data_IR
    predicted_sofa = np.fft.rfft(predicted_sofa, n=256)
    print(predicted_sofa.shape)

    pred_hrtf_avg_left = np.mean(predicted_sofa[:, 0, :], axis=0)
    pred_hrtf_avg_right = np.mean(predicted_sofa[:, 1, :], axis=0)

    pred_hrtf_avg = (pred_hrtf_avg_left+pred_hrtf_avg_right)/2
    

    
    # Check if the patient_id exists in the dataset
    if patient_id not in dataset.all_subjects:
        print(f"Patient ID {patient_id} not found in the dataset.")
        return
    
    # Get the index of the patient_id
    idx = dataset.all_subjects.index(patient_id)
    
    # Retrieve images and HRTF for the specified patient
    images, hrtf = dataset[idx]  # images: [2, num_images, 1, H, W], hrtf: [directions, 2, freq_bins]
    
    hrtf_reference = hrtf
    print(predicted_sofa.shape)
    print(hrtf_reference.shape)
   
    # Assuming hrtf_reference is of shape [directions, 2, freq_bins]
    freq_bins = hrtf_reference.shape[2]
    directions = hrtf_reference.shape[0]

    # Create a frequency axis for plotting
    frequency_axis = np.linspace(0, 0.5, freq_bins)  # Adjust according to your frequency range

    plt.figure(figsize=(12, 6))
    
    # Average across all directions for both left and right channels
    hrtf_avg_left = np.mean(hrtf_reference[:, 0, :], axis=0)
    hrtf_avg_right = np.mean(hrtf_reference[:, 1, :], axis=0)

    hrtf_avg = (hrtf_avg_left+hrtf_avg_right)/2

    # Plot the averaged HRTF

    print("truth shape", hrtf_avg.shape)
    print("pred shape", pred_hrtf_avg.shape)
    plt.plot(frequency_axis, 20 * np.log10(np.abs(hrtf_avg)), label='ground truth')
    plt.plot(frequency_axis, 20 * np.log10(np.abs(pred_hrtf_avg)), label='predicted')

    plt.title(f'HRTF Frequency Response for Patient ID: {patient_id}')
    plt.xlabel('Frequency (normalized)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":

    val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    plot_prediction_for_id("P0004",
                           root_dir = "dataset/", transforms = val_transforms,
                           task_id = 2)