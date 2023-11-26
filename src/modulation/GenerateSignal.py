import numpy as np
from thinkdsp import Wave 
import matplotlib.pyplot as plt
from scipy.fft import fft

def create_wave(duration, framerate, normalization_amp=1.0):
    """
    Creates a normalized random wave and calculates its power using two methods.

    Args:
    duration (float): The duration of the wave in seconds.
    num_samples (int): The number of samples in the wave.
    normalization_amp (float): The amplitude to which the wave is normalized.
    Returns:
    """
    sample_cycle = 128
    num_cyle = 5

    num_samples = int(framerate*duration)

    # Generate random values between 0 and 255
    #random_values = np.random.randint(0, 256, size=sample_cycle*num_cycle) "pour avoir 5 cycles avec 128 points d'amplitude chacun"
    random_values = np.random.randint(0, 256, size=framerate)

    # Create a Wave object from the random values
    
    wave = Wave(random_values, framerate=framerate)


    # Normalize the wave
    wave.ys = wave.ys / max(abs(wave.ys)) * normalization_amp

    # Calculate power in both domains
    pow_signal = signal_power(wave)

    return wave, pow_signal

def signal_power(wave):
    """
    Calculates the power of a wave in both the time and frequency domain.

    Args:
    wave (Wave): The input wave.

    Returns:
    tuple: (float, float) Power calculated in time domain and frequency domain.
    """
    # Time domain calculation
    power_time_domain = np.mean(wave.ys ** 2)

    # Frequency domain calculation using FFT
    n = len(wave.ys)
    d = 1 / wave.framerate
    hs = fft(wave.ys)
    psd = np.abs(hs) ** 2
    power_freq_domain = np.sum(psd) * d / n  # Parseval's theorem

    return power_freq_domain

def noise_power(snr_db, p_signal):
    """
    Calculate the noise power given SNR in dB and signal power.

    Args:
    snr_db (float): SNR in decibels (dB).
    p_signal (float): Power of the signal.

    Returns:
    float: Calculated noise power.
    """
    # Convert SNR from dB to linear ratio and compute noise power
    p_noise = p_signal / pow(10, snr_db/10)
    return p_noise
