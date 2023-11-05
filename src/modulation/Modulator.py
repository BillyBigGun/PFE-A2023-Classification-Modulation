from ModulatorComparator import Wave, CosSignal, SinSignal, decorate

import matplotlib.pyplot as plt


cos_sig = CosSignal(freq=440, amp=1.0, offset=0)
sin_sig = SinSignal(freq=880, amp=0.5, offset=0)

cos_sig.plot()
decorate(xlabel='Time (s)')

sin_sig.plot()
decorate(xlabel='Time (s)')

plt.show()


class modulator:
    def PWM_mod():
        cos(s)

    def PAM_mod():
        filename = '105977__wcfl10__favorite-station.wav'
        wave = thinkdsp.read_wave(filename)
        wave.unbias()
        wave.normalize()
        
        #And here’s the carrier:
        carrier_sig = thinkdsp.CosSignal(freq=10000)
        carrier_wave = carrier_sig.make_wave(duration=wave.duration, framerate=wave.framerate)

        #We can multiply them using the * operator, which multiplies the wave arrays elementwise:
        modulated = wave * carrier_wave
        modulated.plot()
        decorate(xlabel='Time (s)')
        plt.show()

    def PSK():
        tan(x)