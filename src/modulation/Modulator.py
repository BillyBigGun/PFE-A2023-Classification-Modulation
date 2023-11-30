from thinkdsp import SinSignal
from thinkdsp import SquareSignal
from thinkdsp import UncorrelatedGaussianNoise
from thinkdsp import Wave 
from GenerateSignal import*

from scipy.interpolate import interp1d
from scipy.stats import rayleigh

import matplotlib.pyplot as plt
import numpy as np
import csv 
import math 


class PAM:
    def __init__(self, message_amplitude, carrier_amplitude, message_freq, duration, offset, snr_db, sampling, RF):
        """
        Initialize the PAM object with the given parameters.
        duration: duration of the signal in seconds
        framerate: int frames per second
        """

        periode_mess = 1/message_freq
        periode_carr = 1/(2*message_freq)
        sample_per_period = 128
        

        self.framerate_mess = round((sample_per_period*duration)/periode_mess)
        self.framerate_carr = round((sample_per_period*duration)/periode_carr)
        self.duration = duration

        self.carrier_ys = []
        self.carrier_ts = []
        self.pam_wave = []

        # Create the carrier signal (square wave)
        self.carrier_signal = SquareSignal(amp = carrier_amplitude, freq=message_freq*2, offset=offset)
        
        # Make the wave carrier object
        self.carrier_wave = self.carrier_signal.make_wave(duration=duration, framerate=self.framerate_carr)

        # Create the wave
        self.message_wave, signal_pow = create_wave(self.framerate_mess, duration, message_amplitude, norm_amp=1)
        self.noise_pow = noise_power(snr_db, signal_pow)

        # Ensuring the carrier wave values are between 0 and 1
        self.carrier_wave.ys = (self.carrier_wave.ys + carrier_amplitude) / 2  # Transforming from [-1, 1] to [0, 1]
              

        if sampling:
            self.natural_sampling(RF)
        else:
            self.flat_top_sampling(RF)

    def natural_sampling(self, RF):
        
        self.message_wave.ys = add_gaussian_noise(wave_ys=self.message_wave.ys, noise_pow=self.noise_pow, duration=self.duration, framerate=self.framerate_mess)

        if RF:
            Rayleigh(self.message_wave.ys)

        for i in range(len(self.carrier_wave.ts)):
            if i%2 == 0:
                self.carrier_ts.append(self.carrier_wave.ts[i])

        for i in range(len(self.carrier_wave.ys)):
            if i%2 == 0:
                self.carrier_ys.append(self.carrier_wave.ys[i])

        try:
            # Code that might raise an exception
            self.pam_wave_ys = self.message_wave.ys * self.carrier_ys
            self.pam_wave_ts = self.message_wave.ts * self.carrier_ts
        except ValueError as e:
            # Code to handle a specific type of exception (in this case, dividing by zero)
            print(e)
            print(f"le framerate de la carrier wave est : {self.carrier_wave.framerate}")
            print(f"le framerate de la message wave est : {self.message_wave.framerate}")
            print(f"le nombre de point de la carrier wave est : {len(self.carrier_wave.ys)}")
            print(f"le nombre de point de la message wave est : {len(self.message_wave.ys)}")
        
        # Extract the sample arrays for plotting
        self.ts = self.message_wave.ts
        self.message_ys = self.message_wave.ys
        #self.carrier_ys = self.carrier_wave.ys
        
    def flat_top_sampling(self, RF):
        """
        Performs flat-top sampling on the message signal.
        """
        # Add Gaussian noise to the message wave
        self.message_wave.ys = add_gaussian_noise(wave_ys=self.message_wave.ys, noise_pow=self.noise_pow, duration=self.duration, framerate=self.framerate, num_periode=self.num_periode)
        
        if RF:
            Rayleigh(self.message_wave.ys)

        # Interpolate message_wave to match the length of carrier_wave
        x_old = np.linspace(0, len(self.message_wave.ys), len(self.message_wave.ys))
        x_new = np.linspace(0, len(self.message_wave.ys), len(self.carrier_wave.ys))

        interpolator = interp1d(x_old, self.message_wave.ys, kind='linear')
        resampled_message_ys = interpolator(x_new)

        # Initialize an array for flat-top sampled signal
        self.pam_wave = np.zeros_like(self.carrier_wave.ys)

        # Perform flat-top sampling
        for i in range(len(self.carrier_wave.ys)):
            if self.carrier_wave.ys[i] > 0.5:  # Assuming carrier wave is a square wave
                self.pam_wave[i] = resampled_message_ys[i]
            else:
                self.pam_wave[i] = 0  # Set low when carrier is low

        # Extract the sample arrays for plotting
        self.ts = self.message_wave.ts[:len(self.pam_wave)]
        self.message_ys = resampled_message_ys
        self.carrier_ys = self.carrier_wave.ys[:len(self.pam_wave)]

        return self.pam_wave
    
    def write_to_csv(self, filename):
        """
        Writes the amplitude values of the noisy PAM signal to a CSV file.

        filename: name of the CSV file
        """
        data = list(zip(self.pam_wave_ys[:128]))
        headers = ["Amplitude"]

        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)
            csv_writer.writerows(data)

        #print(f"CSV file '{filename}' written successfully.")

class PWM:
    def __init__(self, message_amplitude, carrier_amplitude, message_freq, duration, framerate, offset, snr_db, RF, noise):
        """
        Initialize the PWM object with the given parameters

        message_frequency: frequency of the message wave (Hz)
        carrier_frequency: frequency of the carrier wave (Hz)
        duration: duration of the signal in seconds
        framerate: int frames per second
        """
        periode = 1/message_freq
        self.num_periode = int(duration/periode)
        self.duration = duration
        self.framerate = framerate
        
        # Create the carrier signal (sin wave)
        self.carrier_signal = SinSignal(amp=carrier_amplitude, freq=message_freq*2, offset=offset)

        # Make the wave objects
        self.carrier_wave = self.carrier_signal.make_wave(duration=duration, framerate=framerate*self.num_periode)
        self.message_wave, signal_pow = create_wave(duration, framerate*self.num_periode, normalization_amp=message_amplitude)
        
        # Calculate nooise power in dB
        self.noise_pow = noise_power(snr_db, signal_pow)

        # Ensuring the carrier wave values are between 0 and 1
        self.carrier_wave.ys = (self.carrier_wave.ys + carrier_amplitude) / 2  # Transforming from [-1, 1] to [0, 1]

        # Modulate the carrier wave using PWM
        self.pwm_wave = self.modulate_pwm(self.message_wave, self.carrier_wave, RF=RF, noise=noise)
        
        # Extract the sample arrays for plotting
        self.ts = self.message_wave.ts
        self.message_ys = self.message_wave.ys
        self.carrier_ys = self.carrier_wave.ys
        self.pwm_ys = self.pwm_wave.ys       

    def modulate_pwm(self, message_wave, carrier_wave, RF, noise):
        """
        Create the PWM signal by modulating the width of the carrier pulse based on the message signal.
        
        message_wave: wave object containing the message signal
        carrier_wave: wave object containing the carrier signal
        
        returns: Wave object containing the PWM signal
        """
        five_period_sample = 128*5
        # Initialize an array to store the PWM signal
        pwm_wave = Wave(ys=np.zeros_like(message_wave.ys), ts=message_wave.ts, framerate=message_wave.framerate)

        if noise==1:
            self.message_wave.ys = add_gaussian_noise(wave_ys=self.message_wave.ys, noise_pow=self.noise_pow, duration=self.duration, framerate=self.framerate, num_periode=self.num_periode)
        elif noise==2:
            self.carrier_wave.ys = add_gaussian_noise(wave_ys=self.carrier_wave.ys, noise_pow=self.noise_pow, duration=self.duration, framerate=self.framerate, num_periode=self.num_periode)
        else:
            self.message_wave.ys = add_gaussian_noise(wave_ys=self.message_wave.ys, noise_pow=self.noise_pow, duration=self.duration, framerate=self.framerate, num_periode=self.num_periode)
            self.carrier_wave.ys = add_gaussian_noise(wave_ys=self.carrier_wave.ys, noise_pow=self.noise_pow, duration=self.duration, framerate=self.framerate, num_periode=self.num_periode)


        # Iterate through each point in the wave
        for i in range(five_period_sample):
            # Compare message wave amplitude with carrier wave amplitude
            if message_wave.ys[i] > carrier_wave.ys[i]:
                pwm_wave.ys[i] = 1  # Set high if message amplitude is greater
            else:
                pwm_wave.ys[i] = 0  # Set low otherwise

        if RF:
            Rayleigh(pwm_wave.ys)

        return pwm_wave
    
    def write_to_csv(self, filename):
        """
        Writes a limited number of amplitude values of the noisy PWM signal to a CSV file.

        filename: name of the CSV file
        """
        # Ensure the limit does not exceed the length of the wave data
        limited_data = self.pwm_wave.ys[:min(len(self.pwm_wave.ys), 128)]
        data = list(zip(limited_data))
        headers = ["Amplitude"]

        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)
            csv_writer.writerows(data)

def Rayleigh(pam_wave_ys):
    r = rayleigh.rvs(size=1)[0]  # Extract the scalar value
    for i in range(len(pam_wave_ys)):
        pam_wave_ys[i] = pam_wave_ys[i] * r
        if i % 128 == 0:
            r = rayleigh.rvs(size=1)[0]  # Extract the scalar value
        i = i + i

def add_gaussian_noise(wave_ys, noise_pow, duration, framerate):
    """
    Adds Gaussian noise to the PWM signal using the UncorrelatedGaussianNoise class.

    amp: amplitude of the Gaussian noise
    """
    amplitude = math.sqrt(noise_pow)

    noise_signal = UncorrelatedGaussianNoise(amp=amplitude)
    
    noise_wave = noise_signal.make_wave(duration=duration, start=0, framerate=framerate)
    
    try:
        noisy_wave = wave_ys + noise_wave.ys
    except ValueError as e:
        # Code to handle a specific type of exception (in this case, dividing by zero)
        print(e)
        print(f"{len(noise_wave.ys)}")
        print(f"le nombre de point de la message wave est : {len(wave_ys)}")

    # Normalize the noisy PAM wave
    noisy_wave = noisy_wave / max(abs(noisy_wave))

    return noisy_wave

def plot_signals(ts, message_ys, carrier_ys, modulated_ys, message_freq, n_periods, title_suffix=''):
    plt.figure(figsize=(15, 10))
    
    # Calculate the period of the message signal
    period = 1 / message_freq
    n_periods_duration = n_periods * period

    # Find the index where ts is at least n periods
    index_n_periods = np.where(ts >= n_periods_duration)[0][0]
    
    # Plot the message signal
    plt.subplot(3, 1, 1)
    plt.plot(ts[:index_n_periods], message_ys[:index_n_periods], label="Message Signal")
    plt.title(f'Message Signal {title_suffix}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    # Plot the carrier signal
    plt.subplot(3, 1, 2)
    plt.plot(ts[:index_n_periods], carrier_ys[:index_n_periods], label="Carrier Signal")
    plt.title(f'Carrier Signal {title_suffix}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    # Plot the modulated signal
    plt.subplot(3, 1, 3)
    plt.plot(ts[:index_n_periods], modulated_ys[:index_n_periods], label="Modulated Signal")
    plt.title(f'Modulated Signal {title_suffix}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()



# Test 
n = 9  # For example, if you want to plot 5 periods
freq = 1
pam = PAM(message_amplitude=2, carrier_amplitude=1, message_freq=freq, duration=1, offset=0, snr_db=18, sampling=1, RF=1)
# Example usage:
# Assuming pam.ts is the common timestamps array for all signals
# and message_freq is the frequency of the message signal
#plot_signals(pam.ts, pam.message_ys, pam.carrier_ys, pam.pam_wave_ys, message_freq=freq, n_periods=n, title_suffix=' PAM')

#pam = PAM(message_amplitude=5, carrier_amplitude=1, message_freq=5000, duration=1, framerate=128, offset=0, snr_db=18, sampling=0, RF=0, noise=1)
#plot_signals(pam.ts, pam.message_ys, pam.carrier_ys, pam.pam_wave, title_suffix=' PAM')



