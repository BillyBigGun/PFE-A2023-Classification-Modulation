from lib.ThinkDSP.code.thinkdsp import Wave, SumSignal, CosSignal, SinSignal, UncorrelatedGaussianNoise
import sys
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.stats import rayleigh
from GenerateSignal import signal_power, noise_power
sys.path.append('../../')

class QPSK:

    def __init__(self, frequency=500, nb_bits=10, framerate=11085):
        self.carrier_frequency = frequency
        self.frame_rate = framerate
        self.size_message = nb_bits
        self.nb_period = 0

    def modulate_QPSK(self):
        i = 0
        I_Phase = 0
        Q_Phase = 0
        start = 0
        wave = 0
        k = 1
        QPSK = 0
        num_symbol = 25

        datastream = np.random.randint(0, 2, self.size_message)
        print(datastream)

        shp = datastream.shape
        nb_element = int(shp[0])

        if (nb_element % 2) == 0:
            compt_mod = nb_element
        else:
            datastream = np.concatenate((np.array([0]), datastream))
            compt_mod = nb_element + 1

        constellation_angle = np.zeros(int(compt_mod / 2))

        while i < (compt_mod):
            # Splitting I and Q
            I_Phase = datastream[i]
            if i == (compt_mod - 1):
                break
            Q_Phase = datastream[i + 1]
            # =================================================================

            # Constellation diagram angles logic
            if I_Phase == 0 and Q_Phase == 0:
                print("5pi/4")
                constellation_angle[int((i / 2))] = (5 * np.pi) / 4
            elif I_Phase == 0 and Q_Phase == 1:
                print("3pi/4")
                constellation_angle[int((i / 2))] = (3 * np.pi) / 4
            elif I_Phase == 1 and Q_Phase == 0:
                print("7pi/4")
                constellation_angle[int((i / 2))] = (7 * np.pi) / 4
            elif I_Phase == 1 and Q_Phase == 1:
                print("pi/4")
                constellation_angle[int((i / 2))] = (1 * np.pi) / 4
            print("--------------------")
            # =================================================================

            # NRZ encoder
            if I_Phase == 0:
                I_Phase = -1

            if Q_Phase == 0:
                Q_Phase = -1
            # =================================================================

            # Carrier Oscillator and applying I and Q to the carrier signal
            r = rayleigh.rvs(size=1)
            s = rayleigh.rvs(size=1)
            cos_sig_carrier = CosSignal(freq=self.carrier_frequency, amp=np.sqrt(2) * r * I_Phase, offset=0)
            sin_sig_carrier = SinSignal(freq=self.carrier_frequency, amp=np.sqrt(2) * r * Q_Phase, offset=0)

            # =================================================================

            # Sum I and Q signals to form the QPSK modulation
            QPSK = cos_sig_carrier + sin_sig_carrier
            # =================================================================
            i = i + 2

            wave = wave + QPSK.make_wave(duration=QPSK.period, start=start, framerate=self.frame_rate)
            start = start + QPSK.period
            self.nb_period = self.nb_period + 1

        return wave

    def add_gaussian_noise(self, wave, snr):
        pow_ = signal_power(wave)
        amp = np.sqrt(noise_power(snr, pow_))

        noise_sig = UncorrelatedGaussianNoise(amp=amp)

        noise = noise_sig.make_wave(duration=(self.nb_period-1)*(1/self.carrier_frequency), start=0, framerate=self.frame_rate)

        noisy_QPSK = wave + noise

        # Normalize the noisy PAM wave
        # noisy_QPSK = noisy_QPSK / max(abs(noisy_QPSK))

        return noisy_QPSK

    def write_to_CSV(self, filename, noisy_qpsk_wave):

        data = list(zip(noisy_qpsk_wave.ys))
        headers = ["Amplitude"]
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)
            csv_writer.writerows(data)


freq = 610
frame_rate = 9800
nbbits = 15
QPSK_mod = QPSK(freq, nbbits, frame_rate)

qpsk = QPSK_mod.modulate_QPSK()

noisy_QPSK = QPSK_mod.add_gaussian_noise(qpsk,20)

noisy_QPSK.plot()
plt.show()

print(noisy_QPSK.ys.size)



