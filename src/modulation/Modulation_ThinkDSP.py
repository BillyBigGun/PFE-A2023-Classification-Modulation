import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../../')

from lib.ThinkDSP.code.thinkdsp import Wave, SumSignal, CosSignal, SinSignal

#cos_sig = CosSignal(freq=440, amp=1.0, offset=0)
#sin_sig = SinSignal(freq=880, amp=0.5, offset=0)

#cos_sig.plot()
#plt.show()

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
while i < compt_mod:

    I_Phase = datastream[i]
    if i == 12:
        break
    Q_Phase = datastream[i+1]
    print(I_Phase)
    print(Q_Phase)
    print("--------------------")
    i = i + 1




