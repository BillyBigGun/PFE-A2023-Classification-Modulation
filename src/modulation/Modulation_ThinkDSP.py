import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../../')

from lib.ThinkDSP.code.thinkdsp import Wave, SumSignal, CosSignal, SinSignal

# cos_sig = CosSignal(freq=440, amp=1.0, offset=0)
# sin_sig = SinSignal(freq=880, amp=0.5, offset=0)

# cos_sig.plot()
# plt.show()

data_even = False
i = 0
I_Phase = 0
Q_Phase = 0
datastream = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1])
print(datastream)

shp = datastream.shape
print(shp)
print(shp[0])
nb_element = int(shp[0])

if (nb_element % 2) == 0:
    data_even = True
    compt_mod = nb_element
else:
    datastream = np.concatenate((np.array([0]), datastream))
    compt_mod = nb_element + 1

print(datastream)
constellation_angle = np.zeros(int(compt_mod/2))
print(constellation_angle)

while i < (compt_mod):
    # Splitting I and Q
    I_Phase = datastream[i]
    if i == (compt_mod-1):
        break
    Q_Phase = datastream[i+1]
    print(I_Phase)
    print(Q_Phase)
    print("--------------------")
    # =================================================================

    # Constellation diagram angles logic
    if I_Phase == 0 and Q_Phase == 0:
        print("5pi/4")
        constellation_angle[int((i/2))] = (5 * np.pi) / 4
    elif I_Phase == 0 and Q_Phase == 1:
        print("3pi/4")
        constellation_angle[int((i/2))] = (3 * np.pi) / 4
    elif I_Phase == 1 and Q_Phase == 0:
        print("7pi/4")
        constellation_angle[int((i/2))] = (7 * np.pi) / 4
    elif I_Phase == 1 and Q_Phase == 1:
        print("pi/4")
        constellation_angle[int((i/2))] = (1 * np.pi) / 4
    print("--------------------")
    # =================================================================

    # NRZ encoder
    if I_Phase == 0:
        I_Phase = -1

    if Q_Phase == 0:
        Q_Phase = -1
    # =================================================================

    # Carrier Oscillator and applying I and Q to the carrier signal
    cos_sig_carrier = CosSignal(freq=60, amp=I_Phase, offset=0)
    sin_sig_carrier = SinSignal(freq=60, amp=Q_Phase, offset=0)

    # =================================================================

    # Sum I and Q signals to form the QPSK modulation
    QPSK = cos_sig_carrier + sin_sig_carrier
    # =================================================================
    QPSK.plot()
    plt.show()
    i = i + 2




