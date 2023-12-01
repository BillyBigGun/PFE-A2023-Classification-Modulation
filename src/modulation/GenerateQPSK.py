from Modulation_ThinkDSP import QPSK
import numpy as np

num_samples = 1000
name = []
snr = 20

while snr > 0:
    for j in range(num_samples):
        # Randomize parameters within specified ranges

        # Generate PAM-modulated signal
        freq = 610
        frame_rate = 9800
        nbbits = 15
        QPSK_mod = QPSK(freq, nbbits, frame_rate)

        qpsk = QPSK_mod.modulate_QPSK()



        # Add Gaussian noise

        noisy_QPSK = QPSK_mod.add_gaussian_noise(qpsk, snr)

        filename = f"QPSK_{snr}_{j}.csv"
        name.append([filename])

        # Write the PAM-modulated signal to CSV
        QPSK_mod.write_to_CSV(filename, noisy_QPSK)
    snr = snr - 1

print("Le programme est finie")