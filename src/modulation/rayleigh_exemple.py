from lib.ThinkDSP.code.thinkdsp import Wave, SumSignal, CosSignal, SinSignal, UncorrelatedGaussianNoise
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rayleigh

amp = 1
i = 500
scale = 2
snr = np.zeros(i)
rng = np.random.default_rng()

for j in range(i):

    #r = rayleigh.rvs(size=1)
    #s = rayleigh.rvs(size=1)
    #t = rayleigh.rvs(size=1)
    #u = rayleigh.rvs(size=1)


    r = rng.rayleigh()
    s = rng.rayleigh()
    t = rng.rayleigh()
    u = rng.rayleigh()

    r_pdf = rayleigh.pdf(r)
    s_pdf = rayleigh.pdf(s)
    t_pdf = rayleigh.pdf(t)
    u_pdf = rayleigh.pdf(u)


    sin_original = SinSignal(freq=60, amp=amp, offset=0)
    sin_ori = sin_original.make_wave(duration=sin_original.period, start=0, framerate=11085)
    sin_path = SinSignal(freq=60, amp=s_pdf*amp+s_pdf*amp, offset=0)
    sin_p = sin_path.make_wave(duration=sin_path.period, start=0, framerate=11085)

    # t_pdf*2*np.pi+u_pdf*2*np.pi
    p_original = np.mean(np.abs(sin_ori.ys) ** 2)
    #p_original = np.power(sin_original.amp, 2)
    #p_faded = np.power(sin_path.amp, 2)
    p_faded = np.mean(np.abs(sin_p.ys) ** 2)

    print("p_original : ", p_original)
    print("P_original (dB) : ", 10*np.log10(p_original))
    print("p_faded : ", p_faded)
    print("p_faded (dB) : ", 10*np.log10(p_faded))

    snr[j] = 10 * np.log10((p_original / p_faded))
    #snr[j] = ((p_original / p_faded))
    #print(snr[j])

t = np.arange(0, i, 1)
print("SNR Moyen : ", np.mean(snr))

fig = plt.subplot()
fig.plot(t, snr)
plt.title("SNR depending on signal paths")
plt.xlabel('Path number')
plt.ylabel('SNR')
plt.grid()
plt.show()
