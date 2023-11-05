from ModulatorComparator import CosSignal, SinSignal, SquareSignal
from ModulatorComparator import SquareSignal_test

from ModulatorComparator import decorate
from ModulatorComparator import Wave as wave
from ModulatorComparator import Signal as signal
import matplotlib.pyplot as plt
import numpy as np
import time, random, math

X_LABEL = 'Time (s)'
Y_LABEL = 'Amplitude'

X_LABEL_FREQ = 'Frequence (Hz)'

def simulate_pam():

    signal_m = CosSignal(freq=500)
    duration = signal_m.period*10
    wave_m = signal_m.make_wave(duration, framerate=10000)
    plot_temporel(wave_m, title='Domaine temporel')

    signal_c = CosSignal(freq=500)
    dure = signal_c.period*10
    wave_c = signal_c.make_wave(dure, framerate=10000)
    plot_frequentiel(wave_c, title='Domaine frequentiel porteuse')
    
    modulated = wave_m * wave_c
    plot_temporel(modulated, title='Domaine frequentiel wave module')
    plot_frequentiel(modulated, title='Domaine frequentiel wave module')

def simulate_pwm():
    """
    Simulates PWM and plots a square wave with the given parameters.
    """

    duty_cycle_value = random_zero_to_one()
    square_signal = SquareSignal_test(freq=1, amp=2, offset=0, duty_cycle=duty_cycle_value)
    dur = square_signal.period*10
    wave = square_signal.make_wave(dur, framerate=10000)
    plot_temporel(wave, title='PWM Signal - Duty Cycle: {:.2f}%'.format(duty_cycle_value))

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
    plt.grid(True)
    plt.show()

def plot_frequentiel(wave, title):
    spectrum = wave.make_spectrum()
    spectrum.plot()
    decorate(xlabel=X_LABEL_FREQ)
    decorate(ylabel=Y_LABEL)
    decorate(title=title)
    plt.grid(True)
    plt.show()

simulate_pam()
simulate_pwm()