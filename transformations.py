import random
import torchvision.transforms.functional as TF
from PIL import Image
import torch

class ConsistentTransform:
    def __init__(self, image_size, apply_augmentation=True):
        self.image_size = image_size
        self.apply_augmentation = apply_augmentation
        self.params = None

    def generate_params(self):
        """Generate random augmentation parameters to apply consistently across a series."""
        self.params = {
            "resize": self.image_size,
            "rotation": random.uniform(-10, 10),
            "scale": random.uniform(0.9, 1.1),
            "translate": (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)),
            "brightness": random.uniform(0.9, 1.1),
            "contrast": random.uniform(0.9, 1.1),
            "blur_kernel": 3,
            "blur_sigma": random.uniform(0.1, 0.2)
        }

    def apply_transform(self, image: Image.Image) -> torch.Tensor:
        """Apply the transformation using the generated parameters."""
        if self.params is None:
            raise ValueError("Transformation parameters not initialized. Call `generate_params()` first.")

        image = TF.resize(image, self.params["resize"])
        image = TF.rotate(image, self.params["rotation"])
        image = TF.affine(image, angle=0, translate=self.params["translate"], scale=self.params["scale"], shear=0)
        image = TF.adjust_brightness(image, self.params["brightness"])
        image = TF.adjust_contrast(image, self.params["contrast"])
        image = TF.gaussian_blur(image, self.params["blur_kernel"], sigma=self.params["blur_sigma"])
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.5], std=[0.5])
        return image

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Main function to apply transformation on an image."""
        if self.apply_augmentation:
            if self.params is None:
                self.generate_params()  # Generate params if not already generated
            return self.apply_transform(image)
        else:
            # Validation or non-augmented transformation
            image = TF.resize(image, self.image_size)
            image = TF.to_tensor(image)
            image = TF.normalize(image, mean=[0.5], std=[0.5])
            return image
