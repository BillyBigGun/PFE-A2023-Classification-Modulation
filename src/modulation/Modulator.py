from thinkdsp import SinSignal
from thinkdsp import SquareSignal
from thinkdsp import TriangleSignal
from thinkdsp import UncorrelatedGaussianNoise
from thinkdsp import Wave 
from GenerateSignal import*
import matplotlib.pyplot as plt
import numpy as np
import csv 
import math 

class PAM:
    def __init__(self, message_amplitude, carrier_amplitude, message_freq, duration, framerate, offset, snr_db):
        """
        Initialize the PAM object with the given parameters.
        duration: duration of the signal in seconds
        framerate: int frames per second
        """

        periode = 1/message_freq
        self.num_periode = int(duration/periode)
        self.duration = duration
        self.framerate = framerate
        self.pam_wave = []

        # Create the carrier signal (square wave)
        self.carrier_signal = SquareSignal(amp = carrier_amplitude, freq=message_freq*2, offset=offset)
        
        # Make the wave carrier object
        self.carrier_wave = self.carrier_signal.make_wave(duration=duration, framerate=framerate*self.num_periode)
        self.message_wave, signal_pow = create_wave(duration, 640, normalization_amp=message_amplitude)
        self.noise_pow = noise_power(snr_db, signal_pow)

        # Ensuring the carrier wave values are between 0 and 1
        self.carrier_wave.ys = (self.carrier_wave.ys + carrier_amplitude) / 2  # Transforming from [-1, 1] to [0, 1]

        # Perform pulse amplitude modulation
        #self.pam_wave = self.message_wave.ys * self.carrier_wave.ys[:640]

        try:
            # Code that might raise an exception
            self.pam_wave = self.message_wave.ys * self.carrier_wave.ys[:640]
        except ValueError as e:
            # Code to handle a specific type of exception (in this case, dividing by zero)
            print(e)
            print(f"le framerate de la carrier wave est : {framerate*self.num_periode}")
            print(f"La frequence est : {message_freq}")
        
        # Extract the sample arrays for plotting
        self.ts = self.message_wave.ts[:640]
        self.message_ys = self.message_wave.ys[:640]
        self.carrier_ys = self.carrier_wave.ys[:640]

    def add_gaussian_noise(self):
        """
        Adds Gaussian noise to the PWM signal using the UncorrelatedGaussianNoise class.

        amp: amplitude of the Gaussian noise
        """
        amp = math.sqrt(self.noise_pow)

        noise_signal = UncorrelatedGaussianNoise(amp=amp)
        
        noise_wave = noise_signal.make_wave(duration=self.duration, start=0, framerate=self.framerate*self.num_periode)
        
        self.noisy_pam_wave = self.pam_wave + noise_wave.ys[:640]

        # Normalize the noisy PAM wave
        self.noisy_pam_wave = self.noisy_pam_wave / max(abs(self.noisy_pam_wave))

        return self.noisy_pam_wave
    
    def plot(self):
        """Plots the message, carrier, and PAM signals."""
        plt.figure(figsize=(15, 10))
        
        # Message signal
        plt.subplot(3, 1, 1)
        plt.plot(self.ts, self.message_ys, label="Message Signal")
        plt.title('Message Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
        
        # Carrier signal
        plt.subplot(3, 1, 2)
        plt.plot(self.ts, self.carrier_ys, label="Carrier Signal")
        plt.title('Carrier Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
        
        # PAM signal
        plt.subplot(3, 1, 3)
        plt.plot(self.ts, self.pam_wave, label="PAM Signal")
        plt.title('Pulse Amplitude Modulated Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

    def plot_signals_with_and_without_noise(self):
        """Plots the PAM signal with and without added Gaussian noise."""
        plt.figure(figsize=(15, 6))

        # Plot the original PAM signal
        plt.subplot(2, 1, 1)
        plt.plot(self.ts, self.pam_wave, label="Original PAM Signal")
        plt.title('Original PAM Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()

        # Plot the PAM signal with noise
        plt.subplot(2, 1, 2)
        plt.plot(self.ts, self.noisy_pam_wave, label="Noisy PAM Signal", color='orange')
        plt.title('PAM Signal with Gaussian Noise at 10dB')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def write_to_csv(self, filename):
        """
        Writes the amplitude values of the noisy PAM signal to a CSV file.

        filename: name of the CSV file
        """
        data = list(zip(self.noisy_pam_wave))
        headers = ["Amplitude"]

        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)
            csv_writer.writerows(data)

        #print(f"CSV file '{filename}' written successfully.")

class PWM:
    def __init__(self, message_amplitude, carrier_amplitude, message_freq, duration, framerate, offset, snr_db):
        """
        Initialize the PWM object with the given parameters using the entities from Test_Diego file.

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

        # Modulate the carrier wave using PWM
        self.pwm_wave = self.modulate_pwm(self.message_wave, self.carrier_wave)
        
        # Extract the sample arrays for plotting
        self.ts = self.message_wave.ts
        self.message_ys = self.message_wave.ys
        self.carrier_ys = self.carrier_wave.ys
        self.pwm_ys = self.pwm_wave.ys       

    def modulate_pwm(self, message_wave, carrier_wave):
        """
        Create the PWM signal by modulating the width of the carrier pulse based on the message signal.
        
        message_wave: wave object containing the message signal
        carrier_wave: wave object containing the carrier signal
        
        returns: Wave object containing the PWM signal
        """
        five_period_sample = 128*5
        # Initialize an array to store the PWM signal
        pwm_wave = Wave(ys=np.zeros_like(message_wave.ys), ts=message_wave.ts, framerate=message_wave.framerate)

        # Iterate through each point in the wave
        for i in range(five_period_sample):
            # Compare message wave amplitude with carrier wave amplitude
            if message_wave.ys[i] > carrier_wave.ys[i]:
                pwm_wave.ys[i] = 1  # Set high if message amplitude is greater
            else:
                pwm_wave.ys[i] = 0  # Set low otherwise

        return pwm_wave

    def add_gaussian_noise(self):
        """
        Adds Gaussian noise to the PWM signal using the UncorrelatedGaussianNoise class.

        amp: amplitude of the Gaussian noise
        """
        amp = math.sqrt(self.noise_pow)

        noise_signal = UncorrelatedGaussianNoise(amp=amp)
        
        noise_wave = noise_signal.make_wave(duration=self.duration, start=0, framerate=self.framerate*self.num_periode)
        
        self.noisy_pwm_wave = self.pwm_wave.ys + noise_wave.ys

        # Normalize the noisy PAM wave
        self.noisy_pwm_wave = self.noisy_pwm_wave / max(abs(self.noisy_pwm_wave))

        return self.noisy_pwm_wave
    
    def plot(self):
        """Plots the message, carrier, and PAM signals."""
        plt.figure(figsize=(15, 10))
        
        # Message signal
        plt.subplot(3, 1, 1)
        plt.plot(self.ts, self.message_ys, label="Message Signal")
        #plt.plot(self.ts, self.message_ys, 'o', label="Message Signal")  # 'o' is the marker style for dots
        #plt.plot(self.ts, self.message_ys, 'o-', label="Message Signal")  # Line with dots
        plt.title('Message Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # Carrier signal
        plt.subplot(3, 1, 2)
        plt.plot(self.ts, self.carrier_ys, label="Carrier Signal")
        plt.title('Carrier Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # PAM signal
        plt.subplot(3, 1, 3)
        plt.plot(self.ts, self.pwm_ys, label="PAM Signal")
        plt.title('Pulse Amplitude Modulated Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_signals_with_and_without_noise(self):
        """Plots the PWM signal with and without added Gaussian noise."""
        plt.figure(figsize=(15, 6))

        # Plot the original PWM signal
        plt.subplot(2, 1, 1)
        plt.plot(self.pwm_wave.ts, self.pwm_wave.ys, label="Original PWM Signal")
        plt.title('Original PWM Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()

        # Plot the PWM signal with noise
        plt.subplot(2, 1, 2)
        plt.plot(self.pwm_wave.ts, self.noisy_pwm_wave, label="Noisy PWM Signal", color='orange')
        plt.title('PWM Signal with Gaussian Noise at 10dB')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def write_to_csv(self, filename, noisy_pwm_wave):
        """
        Writes a limited number of amplitude values of the noisy PWM signal to a CSV file.

        filename: name of the CSV file
        noisy_pwm_wave: list of amplitude values for the noisy PWM signal
        limit: the maximum number of amplitude values to write to the CSV file
        """
        # Ensure the limit does not exceed the length of noisy_pwm_wave
        limited_data = noisy_pwm_wave[:min(len(noisy_pwm_wave), 128*5)]
        data = list(zip(limited_data))
        headers = ["Amplitude"]

        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)
            csv_writer.writerows(data)

pam = PAM(message_amplitude=5, carrier_amplitude=1, message_freq=5000, duration=1, framerate=128, offset=0, snr_db=18)
pam.plot()
pam.add_gaussian_noise()
pam.plot_signals_with_and_without_noise()

#pwm = PWM(message_amplitude=1, carrier_amplitude=1, message_freq=5, duration=1, framerate=128, offset=0, snr_db=18)
#pwm.plot()
#pwm.add_gaussian_noise()
#pwm.plot_signals_with_and_without_noise()
