from Modulator import PAM
from Modulator import PWM
import numpy as np

num_samples = 1
name = []
snr = 20

def db_pwm(j):

    # Randomize parameters within specified ranges
    mess_amp = np.random.randint(1, 5)
    carrier_amp = np.random.randint(1, 5)
    message_frequence = np.random.randint(5, 10000)  
    offset = np.random.randint(1, 500)
    noise_add = np.random.randint(0, 2)  
    framerate = 128

    # Generate PAM-modulated signal
    pwm = PWM(mess_amp, carrier_amp, message_frequence, 1, framerate, offset, snr, 1, noise_add)

    filename = f"PWM_{snr}_{j}.csv"
    name.append([filename])

    # Write the PAM-modulated signal to CSV
    pwm.write_to_csv(filename)

def db_pam(j):
    
    # Randomize parameters within specified ranges
    mess_amp = np.random.randint(1, 5)
    carrier_amp = np.random.randint(1, 5)
    message_frequence = np.random.randint(5, 10000)  
    offset = np.random.randint(1, 500)
    noise_add = np.random.randint(0, 2)  
    framerate = 128

    # Generate PAM-modulated signal
    pam = PAM(mess_amp, carrier_amp, message_frequence, 1, offset, snr, 1, 1)
              
    filename = f"PAM_{snr}_{j}.csv"
    name.append([filename])

    # Write the PAM-modulated signal to CSV
    pam.write_to_csv(filename)

while snr > 0 :
    for j in range(num_samples):
        db_pam(j)
    snr = snr - 1

snr=0
 
while snr > 0 :
    for j in range(num_samples):
        db_pwm(j)
    snr = snr - 1

print("Le programme est finie")