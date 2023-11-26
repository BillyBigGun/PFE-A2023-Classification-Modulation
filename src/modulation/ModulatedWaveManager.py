from Modulator import PAM
from Modulator import PWM
import numpy as np

num_samples = 1000
name = []
snr = 20

while snr > 0 :
    for j in range(num_samples):
        # Randomize parameters within specified ranges
        mess_amp = np.random.randint(1, 5)
        carrier_amp = np.random.randint(1, 5)
        message_frequence = np.random.randint(5, 10000)  
        n_framerate = np.random.randint(1, 5)  
        framerate = 128
        offset = np.random.randint(1, 500)  

        # Generate PAM-modulated signal
        pam_instance = PAM(message_amplitude=mess_amp, carrier_amplitude=carrier_amp, message_freq=message_frequence, duration=1, framerate=framerate, offset=offset, snr_db=snr)

        # Add Gaussian noise
        pam_instance.add_gaussian_noise()

        filename = f"PAM_{snr}_{j}.csv"
        name.append([filename])

        # Write the PAM-modulated signal to CSV
        pam_instance.write_to_csv(filename)
    snr = snr - 1

print("Le programme est finie")