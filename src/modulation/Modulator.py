from thinkdsp import SinSignal
from thinkdsp import SquareSignal
from thinkdsp import TriangleSignal
from thinkdsp import UncorrelatedGaussianNoise
from thinkdsp import Wave 
import matplotlib.pyplot as plt
import numpy as np
import csv 
import matplotlib.pyplot as plt
import numpy as np
import csv 

class PAM:
    def __init__(self, message_frequency, carrier_frequency, message_amplitude, carrier_amplitude, duration, framerate, wave_type):
        """
        Initialize the PAM object with the given parameters using the entities from Test_Diego file.

        message_frequency: frequency of the message wave (Hz)
        carrier_frequency: frequency of the carrier wave (Hz)
        duration: duration of the signal in seconds
        framerate: int frames per second
        """

        if wave_type == 0:
            self.message_signal = SinSignal(amp=message_amplitude, freq=message_frequency)
        elif wave_type == 1:
            self.message_signal = SquareSignal(amp=message_amplitude, freq=message_frequency)
        
        # Create the carrier signal (square wave)
        self.carrier_signal = SquareSignal(amp = carrier_amplitude, freq=carrier_frequency)
        
        # Make the wave objects
        self.message_wave = self.message_signal.make_wave(duration=duration, framerate=framerate)
        self.carrier_wave = self.carrier_signal.make_wave(duration=duration, framerate=framerate)

        # Perform pulse amplitude modulation
        self.pam_wave = self.message_wave * self.carrier_wave
        
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

    def add_gaussian_noise(self, amp):
        """
        Adds Gaussian noise to the PWM signal using the UncorrelatedGaussianNoise class.

        amp: amplitude of the Gaussian noise
        """
        noise_signal = UncorrelatedGaussianNoise(amp=amp)
        noise_wave = noise_signal.make_wave(duration=self.duration, start=0, framerate=self.message_wave.framerate)
        self.noisy_pam_wave = self.pam_wave.ys + noise_wave.ys

        # Normalize the noisy PAM wave
        self.noisy_pam_wave = self.noisy_pam_wave / max(abs(self.noisy_pam_wave))

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

        print(f"CSV file '{filename}' written successfully.")

class PWM:
    def __init__(self, message_frequency, carrier_frequency, message_amplitude, carrier_amplitude, duration, framerate, wave_type):
        """
        Initialize the PWM object with the given parameters using the entities from Test_Diego file.

        message_frequency: frequency of the message wave (Hz)
        carrier_frequency: frequency of the carrier wave (Hz)
        duration: duration of the signal in seconds
        framerate: int frames per second
        """

        if wave_type == 0:
            self.message_signal = SinSignal(amp=message_amplitude, freq=message_frequency)
            self.carrier_signal = SinSignal(amp=carrier_amplitude, freq=carrier_frequency)
        elif wave_type == 1:
            self.message_signal = TriangleSignal(amp=message_amplitude, freq=message_frequency)
            self.carrier_signal = TriangleSignal(amp=carrier_amplitude, freq=carrier_frequency)

        # Make the wave objects
        self.message_wave = self.message_signal.make_wave(duration=duration, framerate=framerate)
        self.carrier_wave = self.carrier_signal.make_wave(duration=duration, framerate=framerate)
        
        # Modulate the carrier wave using PWM
        self.pwm_wave = self._modulate_pwm(self.message_wave, self.carrier_wave)
        
        # Extract the sample arrays for plotting
        self.ts = self.message_wave.ts
        self.message_ys = self.message_wave.ys
        self.carrier_ys = self.carrier_wave.ys
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
        plt.plot(self.ts, self.pwm_ys, label="PAM Signal")
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
        self.noisy_pwm_wave = self.pwm_wave.ys + noise_wave.ys

        # Normalize the noisy PAM wave
        self.noisy_pwm_wave = self.noisy_pwm_wave / max(abs(self.noisy_pwm_wave))

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
        plt.title('PWM Signal with Gaussian Noise at 10dB')
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
        data = list(zip(self.noisy_pwm_wave))
        headers = ["Amplitude"]

        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)
            csv_writer.writerows(data)

        print(f"CSV file '{filename}' written successfully.")