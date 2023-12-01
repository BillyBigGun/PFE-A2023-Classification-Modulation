from lib.ThinkDSP.code.thinkdsp import Wave, SumSignal, CosSignal, SinSignal, UncorrelatedGaussianNoise
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rayleigh

amp = 2
i = 100
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
    sin_path = SinSignal(freq=60, amp=r_pdf*amp+s_pdf*amp, offset=t_pdf*2*np.pi+u_pdf*2*np.pi)

    p_original = np.power(sin_original.amp, 2)
    p_faded = np.power(sin_path.amp, 2)
    print(20*np.log10(p_original))
    print(20*np.log10(p_faded))

    snr[j] = -20*np.log10((p_original/p_faded))
    #print(snr[j])

t = np.arange(0, 100, 1)

fig = plt.subplot()
fig.plot(t, snr)
plt.show()
