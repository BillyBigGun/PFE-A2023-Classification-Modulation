from Modulator import *

num_samples = 5
name = []
snr = 0

for i in range(num_samples):
    # Randomize parameters within specified ranges
    carrier_amplitude = np.random.uniform(1, 5)
    message_amplitude = np.random.uniform(0.5, 2)
    noise_amplitude = np.random.uniform(0.01, 0.1)
    carrier_frequency = np.random.uniform(5, 100)
    message_frequency = np.random.uniform(1, 10)
    framerate = np.random.uniform(500, 15000)
    wave_type = np.random.uniform(0, 1)
    snr = 10 * np.log10(pow(message_amplitude,2) / pow(noise_amplitude,2))
    snr = round(snr, 2)

    # Generate PAM-modulated signal
    pam_instance = PAM(message_frequency=message_frequency, carrier_frequency=carrier_frequency, message_amplitude=message_amplitude, carrier_amplitude=carrier_amplitude, duration=1, framerate=10000, wave_type=wave_type)

    # Add Gaussian noise
    pam_instance.add_gaussian_noise(amp=noise_amplitude)

    filename = f"PAM_{i}_{snr}.csv"
    name.append([i, filename])

    # Write the PAM-modulated signal to CSV
    pam_instance.write_to_csv(filename)


for i in range(num_samples):
    # Randomize parameters within specified ranges
    carrier_amplitude = np.random.uniform(1, 5)
    message_amplitude = np.random.uniform(0.5, 2)
    noise_amplitude = np.random.uniform(0.01, 0.1)
    carrier_frequency = np.random.uniform(5, 100)
    message_frequency = np.random.uniform(1, 10)
    framerate = np.random.uniform(500, 15000)
    wave_type = np.random.uniform(0, 1)
    snr = 10 * np.log10(pow(message_amplitude,2) / pow(noise_amplitude,2))
    snr = round(snr, 2)

    # Generate PAM-modulated signal
    pwm_instance = PWM(message_frequency=message_frequency, message_amplitude=message_amplitude, carrier_amplitude=carrier_amplitude, carrier_frequency=carrier_frequency, duration=1, framerate=framerate, wave_type=wave_type)

    # Add Gaussian noise
    pwm_instance.add_gaussian_noise(amp=noise_amplitude)

    filename = f"PWM_{i}_{snr}.csv"
    name.append([i, filename])

    # Write the PAM-modulated signal to CSV
    pwm_instance.write_to_csv(filename)



