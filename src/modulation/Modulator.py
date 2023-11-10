from thinkdsp import CosSignal
from thinkdsp import SinSignal
from thinkdsp import SquareSignal
from thinkdsp import UncorrelatedGaussianNoise
from thinkdsp import decorate
from thinkdsp import Wave 
from thinkdsp import Signal as signal
import matplotlib.pyplot as plt
import numpy as np
import random
import csv 
import math

import matplotlib.pyplot as plt
import numpy as np
import csv 

X_LABEL = 'Time (s)'
Y_LABEL = 'Amplitude'

X_LABEL_FREQ = 'Frequence (Hz)'


class PAMTestDiego:
    def __init__(self, message_frequency, carrier_frequency, duration, framerate):
        """
        Initialize the PAM object with the given parameters using the entities from Test_Diego file.

        message_frequency: frequency of the message wave (Hz)
        carrier_frequency: frequency of the carrier wave (Hz)
        duration: duration of the signal in seconds
        framerate: int frames per second
        """
        # Create the message signal (sine wave)
        self.message_signal = SinSignal(freq=message_frequency)
        
        # Create the carrier signal (square wave)
        self.carrier_signal = SquareSignal(freq=carrier_frequency)
        
        # Make the wave objects
        self.message_wave = self.message_signal.make_wave(duration=duration, framerate=framerate)
        self.carrier_wave = self.carrier_signal.make_wave(duration=duration, framerate=framerate)

        # Perform pulse amplitude modulation
        self.pam_wave = self.message_wave * self.carrier_wave
        self.pam_wave.normalize()
        
        # Extract the sample arrays for plotting
        self.ts = self.message_wave.ts
        self.message_ys = self.message_wave.ys
        self.carrier_ys = self.carrier_wave.ys
        self.pam_ys = self.pam_wave.ys
        self.duration = duration

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
        
        # Carrier signal
        plt.subplot(3, 1, 2)
        plt.plot(self.ts, self.carrier_ys, label="Carrier Signal")
        plt.title('Carrier Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # PAM signal
        plt.subplot(3, 1, 3)
        plt.plot(self.ts, self.pam_ys, label="PAM Signal")
        plt.title('Pulse Amplitude Modulated Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def add_gaussian_noise(self, amp=1):
        """
        Adds Gaussian noise to the PWM signal using the UncorrelatedGaussianNoise class.

        amp: amplitude of the Gaussian noise
        """
        noise_signal = UncorrelatedGaussianNoise(amp=amp)
        noise_wave = noise_signal.make_wave(duration=self.duration, start=0, framerate=self.message_wave.framerate)
        self.noisy_pam_wave = self.pam_wave.ys + noise_wave.ys
        return self.noisy_pam_wave

    def plot_signals_with_and_without_noise(self):
        """Plots the PAM signal with and without added Gaussian noise."""
        plt.figure(figsize=(15, 6))

        # Plot the original PAM signal
        plt.subplot(2, 1, 1)
        plt.plot(self.pam_wave.ts, self.pam_wave.ys, label="Original PAM Signal")
        plt.title('Original PAM Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()

        # Plot the PAM signal with noise
        plt.subplot(2, 1, 2)
        plt.plot(self.pam_wave.ts, self.noisy_pam_wave, label="Noisy PAM Signal", color='orange')
        plt.title('PAM Signal with Gaussian Noise at 0dB')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def write_to_csv(self, wave, filename):
        with open(filename, mode='w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=';')  # Adjust delimiter as needed
            # Write header with the required parameter names
            csv_writer.writerow(["Time", "Amplitude", "Message Frequency", "Carrier Frequency", "Duration", "Frame Rate"])

            # Retrieve parameters, assuming the PAM class has these attributes
            parameters = [self.message_signal.freq, self.carrier_signal.freq, self.duration, self.framerate]

            # Write the parameters line
            csv_writer.writerow(["Parameters", "", *parameters])

            # Write data without parameters for the rest of the rows
            for t, y in zip(wave.ts, wave.ys):
                csv_writer.writerow([t, y])

        print(f"CSV file '{filename}' written with PAM data and parameters.")


class PWMTestDiego:
    def __init__(self, message_frequency, carrier_frequency, duration, framerate):
        """
        Initialize the PWM object with the given parameters using the entities from Test_Diego file.

        message_frequency: frequency of the message wave (Hz)
        carrier_frequency: frequency of the carrier wave (Hz)
        duration: duration of the signal in seconds
        framerate: int frames per second
        """
        # Create the message signal (sine wave)
        self.message_signal = SinSignal(freq=message_frequency)
        
        # Create the carrier signal (square wave) with a high frequency to represent the pulse train
        self.carrier_signal = SinSignal(freq=carrier_frequency)
        
        # Make the wave objects
        self.message_wave = self.message_signal.make_wave(duration=duration, framerate=framerate)
        self.carrier_wave = self.carrier_signal.make_wave(duration=duration, framerate=framerate)
        
        # Modulate the carrier wave using PWM
        self.pwm_wave = self._modulate_pwm(self.message_wave, self.carrier_wave)
        
        # Extract the sample arrays for plotting
        self.ts = self.message_wave.ts
        self.message_ys = self.message_wave.ys
        self.pwm_ys = self.pwm_wave.ys
        self.duration = duration

    def _modulate_pwm(self, message_wave, carrier_wave):
        """
        Create the PWM signal by modulating the width of the carrier pulse based on the message signal.
        
        message_wave: wave object containing the message signal
        carrier_wave: wave object containing the carrier signal
        
        returns: Wave object containing the PWM signal
        """
        # Initialize a PWM wave with zeros
        pwm_wave = Wave(ys=np.zeros_like(message_wave.ys), ts=message_wave.ts, framerate=message_wave.framerate)
        
        # The duration of each pulse in the carrier wave
        pulse_duration = 1.0 / self.carrier_signal.freq
        
        for i, (m, c) in enumerate(zip(message_wave.ys, carrier_wave.ys)):
            # Adjust the pulse width based on the message signal value
            if c > 0:
                if message_wave.ts[i] % pulse_duration < pulse_duration * (m + 1) / 2:
                    pwm_wave.ys[i] = 1
        return pwm_wave

    def plot_all(self):
        """Plots the message and PWM signals."""
        plt.figure(figsize=(15, 10))
        
        # Message signal
        plt.subplot(2, 1, 1)
        plt.plot(self.ts, self.message_ys, label="Message Signal")
        plt.title('Message Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # PWM signal
        plt.subplot(2, 1, 2)
        plt.plot(self.ts, self.pwm_ys, label="PWM Signal")
        plt.title('Pulse Width Modulated Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def add_gaussian_noise(self, amp=1):
        """
        Adds Gaussian noise to the PWM signal using the UncorrelatedGaussianNoise class.

        amp: amplitude of the Gaussian noise
        """
        noise_signal = UncorrelatedGaussianNoise(amp=amp)
        noise_wave = noise_signal.make_wave(duration=self.duration, start=0, framerate=self.message_wave.framerate)
        self.noisy_pwm_wave = self.pwm_wave.ys + noise_wave.ys
        return self.noisy_pwm_wave

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
        plt.title('PWM Signal with Gaussian Noise at 33dB')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def write_to_csv(self, wave, filename):
        
        #for x in zip(wave.ys):
            #print(len(wave.ts), wave.ts[x])
            
        for element in wave.ys:
            print(element)
            

        with open(filename, mode='w', newline='') as file:
            csv_writer = csv.writer(file, delimiter=';')  # Adjust delimiter as needed
            # Write header with additional parameter names
            csv_writer.writerow(["Time", "Amplitude", "Frequency", "Offset", "Duty Cycle", "Duration", "Frame Rate"])
            
            # Retrieve parameters
            parameters = [self.freq, self.offset, self.duty_cycle, self.duration, self.framerate]
            
            # Write data with parameters
            for t, y in zip(wave.ts, wave.ys):
                if(x == 0):
                    csv_writer.writerow([t, y] + parameters)  # Append parameters to each row
                else:
                    csv_writer.writerow([t, y])  # Append parameters to each row
                x = 1 
                
        print(f"CSV file '{filename}' written with PWM data and parameters.")

def Gaussian_Noise():
    signal = UncorrelatedGaussianNoise(amp=1)
    wave = signal.make_wave(duration=10, framerate=10000)
    return(wave)

def Gaussian_Noise_test(duration):
    signal = UncorrelatedGaussianNoise(amp=1)
    wave = signal.make_wave(duration=duration, framerate=10000)
    return(wave)


    
# PAM example

#pam_modulator.write_to_csv(pam_wave, 'PAM_output.csv')
#pwm_modulator.write_to_csv(pwm_wave, 'PWM_output.csv')

# Create an instance of PAM using classes from "Test_Diego"
pam_instance = PAMTestDiego(message_frequency=5, carrier_frequency=50, duration=1, framerate=10000)
pam_instance.add_gaussian_noise(amp=5)  # Add Gaussian noise with amplitude 0.1
pam_instance.plot_signals_with_and_without_noise()  # Plot the signals


# Instantiate the class and plot the signals
pwm_instance = PWMTestDiego(message_frequency=5, carrier_frequency=50, duration=1, framerate=10000)
pwm_instance.add_gaussian_noise(amp=0.1)  # Add Gaussian noise with amplitude 0.1
pwm_instance.plot_signals_with_and_without_noise()  # Plot the signals