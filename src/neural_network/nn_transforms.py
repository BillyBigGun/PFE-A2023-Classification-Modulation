import torch
from torchvision import transforms

# Define a custom transformation function
def transform_to_2d(signal):
    if signal.ndim == 1:
        # Reshape from [128] to [1, 128, 1] for Conv2D
        signal = signal.reshape(1, -1, 1)
    return signal


def get_transform_to_2d():
    # Create a transformation pipeline
    transform = transforms.Compose([
        transforms.Lambda(transform_to_2d),
        # Add any other transformations here
    ])

    return transform
