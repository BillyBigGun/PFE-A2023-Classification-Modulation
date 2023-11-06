from ModulatorComparator import CosSignal
from ModulatorComparator import SquareSignal_test
from ModulatorComparator import UncorrelatedGaussianNoise
from ModulatorComparator import decorate
from ModulatorComparator import Wave as wave
from ModulatorComparator import Signal as signal
import matplotlib.pyplot as plt
import numpy as np
import random
import csv 

X_LABEL = 'Time (s)'
Y_LABEL = 'Amplitude'

X_LABEL_FREQ = 'Frequence (Hz)'

def simulate_pam():

    signal_m = CosSignal(freq=500, amp=5)
    #duration = signal_m.period*100
    duration = 0.01
    wave_m = signal_m.make_wave(duration, framerate=10000)
    #plot_temporel(wave_m, title='Domaine temporel')

    signal_c = CosSignal(freq=1000)
    #dure = signal_c.period*100
    wave_c = signal_c.make_wave(duration, framerate=10000)
    
    modulated = wave_m * wave_c
    #plot_temporel(modulated, title='Domaine frequentiel wave module')
    #plot_frequentiel(modulated, title='Domaine frequentiel wave module')

    #noise_wave = Gaussian_Noise()
    noise_wave = Gaussian_Noise_test(duration)
    #plot_frequentiel(wave, title='Bruit Gaussien')

    mod_noise_wave = Add_Gaussian_Noise(modulated, noise_wave)
    plot_temporel(mod_noise_wave, title='Noise added')


def simulate_pwm():
    """
    Simulates PWM and plots a square wave with the given parameters.
    """

    duty_cycle_value = random_zero_to_one()
    square_signal = SquareSignal_test(freq=1, amp=2, offset=0, duty_cycle=duty_cycle_value)
    dur = square_signal.period*10
    wave = square_signal.make_wave(dur, framerate=10000)


    noise_wave = Gaussian_Noise()
    plot_frequentiel(wave, title='Bruit Gaussien')

    mod_noise_wave = Add_Gaussian_Noise(wave, noise_wave)
    plot_temporel(mod_noise_wave, title='Noise added')


def simulate_psk():
    print('fuck')

def random_zero_to_one():
    return round(random.random(), 2)

def randomize_frequence(min_frequence, max_frequence):
    return random.uniform(min_frequence, max_frequence)

def plot_temporel(wave, title):
    wave.plot()
    decorate(xlabel=X_LABEL)
    decorate(ylabel=Y_LABEL)
    decorate(title=title)
    plt.margins(x=0.1)
    plt.margins(y=0.1)
    plt.grid(True)
    plt.show()

def plot_frequentiel(wave, title):
    spectrum = wave.make_spectrum()
    spectrum.plot()
    decorate(xlabel=X_LABEL_FREQ)
    decorate(ylabel=Y_LABEL)
    decorate(title=title)
    plt.margins(x=0.1)
    plt.margins(y=0.1)
    plt.grid(True)
    plt.show()

def Gaussian_Noise():
    signal = UncorrelatedGaussianNoise(amp=1)
    wave = signal.make_wave(duration=10, framerate=10000)
    return(wave)

def Gaussian_Noise_test(duration):
    signal = UncorrelatedGaussianNoise(amp=1)
    wave = signal.make_wave(duration=duration, framerate=10000)
    return(wave)

def Add_Gaussian_Noise(wave, noise):
    noisy_wave = wave + noise
    return noisy_wave
    

simulate_pam()
#simulate_pwm()
#Gaussian_Noise()


    # Define the CSV filename
    #csv_filename = 'Test_ecriture.csv'  # Make sure to include .csv extension

    # Write the wave data to a CSV file
    #with open(csv_filename, mode='w', newline='') as file:  # Corrected line
        #csv_writer = csv.writer(file)
        # Write header
       # csv_writer.writerow(["Time", "Amplitude"])
        # Write the time and amplitude data
      #  for t, y in zip(wave.ts, wave.ys):
     #       csv_writer.writerow([t, y])

    #print(f"CSV file '{csv_filename}' written with PWM data.")