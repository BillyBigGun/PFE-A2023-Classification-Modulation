import torch
from torchvision import transforms

# Define a custom transformation function
def __transform_to_2d(signal):
    if signal.ndim == 1:
        # Reshape from [128] to [1, 128, 1] for Conv2D
        signal = signal.reshape(1, -1, 1)
    return signal

def __transform_t_cnn(signal):
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)  # add a channel dimension
    return signal


def get_transform_to_2d():
    # Create a transformation pipeline
    transform = transforms.Compose([
        transforms.Lambda(__transform_to_2d),
        # Add any other transformations here
    ])

    return transform

def get_transform_t_cnn():
    # Create a transformation pipeline
    transform = transforms.Compose([
        transforms.Lambda(__transform_t_cnn),
        # Add any other transformations here
    ])

    return transform