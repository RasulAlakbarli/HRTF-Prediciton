import argparse
import os
from os.path import abspath, dirname, join
import numpy as np
import sofar
import tqdm
from PIL import Image
from imageio.v3 import imread

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import SonicomDatabase, save_sofa
from transformations import ConsistentTransform
from model import HRTFModel 
from metrics import MeanSpectralDistortion


AVG_HRTF_PATH = "data/Average_HRTFs.sofa"


class BaselineHRTFPredictor:
    def __init__(self, average_hrtf_path: str = AVG_HRTF_PATH):
        """
        Creates a predictor instance. 
        average_HRTF_path is the path the file 'Average_HRTFs.sofa' that was delivered as part of the project.
        """
        self.average_hrir = sofar.read_sofa(average_hrtf_path, verbose=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model
        model_path = 'best_model_vf5c.pth'
        self.model = HRTFModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()


    def predict(self, images: torch.Tensor) -> np.ndarray:
        """
        Predict the HRTF based on left and right images.

        Args:
            images: images for left and right pinna as 4-dimensional tensor of size (number of ears, number of images per ear, image height, image width)

        Returns:
            sofar.Sofa: Predicted HRIR in SOFA format.
        """

        with torch.no_grad():
            hrir = self.model(images.to(self.device))

        return hrir
    

def evaluate():
    """
    Evaluate the model on all 3 tasks
    """
    
    val_transforms = ConsistentTransform(image_size=(256, 256), apply_augmentation=False)

    predictor = BaselineHRTFPredictor()
    metric = MeanSpectralDistortion()
    results = {}

    for task in range(3):
        sd = SonicomDatabase(root_dir="data/", training_data=None, transforms=val_transforms, task_id=task)
        test_dataloader = DataLoader(sd, batch_size=1, shuffle=False)
        
        total_error= []
        for image_batch, hrtf_batch in tqdm.tqdm(test_dataloader):
            hrir = predictor.predict(image_batch)
            predicted_hrtf = torch.fft.rfft(hrir, n=256)

            total_error.append(metric.get_spectral_distortion(hrtf_batch, predicted_hrtf))
        results[task] = np.mean(total_error)

    print("Average Mean Spectral Distortion for Task 1: ", results[0])
    print("Average Mean Spectral Distortion for Task 2: ", results[1])
    print("Average Mean Spectral Distortion for Task 3: ", results[2])


def main():
    """
    Main function to run the inference script.
    How to run:
    For task 0: python inference.py -l 19 left_image_pathes -r 19 right_image_pathes -o output_path
    For task 1: python inference.py -l 7 left_image_pathes -r 7 right_image_pathes -o output_path
    For task 2: python inference.py -l 3 left_image_pathes -r 3 right_image_pathes -o output_path
    """
    val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    parser = argparse.ArgumentParser(description="Baseline HRTF Inference Script")
    parser.add_argument("-l", "--left", metavar='IMAGE_PATH', type=str, nargs='+', required=True, help="List of left pinna images")
    parser.add_argument("-r", "--right", metavar='IMAGE_PATH', type=str, nargs='+', required=True, help="List of right pinna images")
    parser.add_argument("-o", "--output_path", metavar='SOFA_PATH', type=str, required=True, help="File path to save the predicted HRTF in SOFA format.")
    args = parser.parse_args()

    # load images
    left_images = [val_transforms(Image.fromarray(imread(path))) for path in args.left]
    right_images = [val_transforms(Image.fromarray(imread(path))) for path in args.right]
    left_images = torch.stack(left_images)  # Shape: [num_images, 1, H, W]
    right_images = torch.stack(right_images)  # Shape: [num_images, 1, H, W]
    images = torch.stack((left_images, right_images), dim=0)  # Shape: [2, num_images, 1, H, W]
    images = images.unsqueeze(0)  # Shape: [1, 2, 19, 1, 256, 256]

    # predict HRTFs
    predictor = BaselineHRTFPredictor()
    hrir_sofa = predictor.predict(images).squeeze(0).cpu().numpy()

    # Use the average HRTF template as a base
    hrtf_template = sofar.read_sofa(AVG_HRTF_PATH, verbose=False)
    # Replace the impulse response data in the template
    hrtf_template.Data_IR = hrir_sofa
    
    # Write the SOFA file
    sofar.write_sofa(args.output_path, hrtf_template, compression=0)


if __name__ == "__main__":
    # evaluate()
    main()