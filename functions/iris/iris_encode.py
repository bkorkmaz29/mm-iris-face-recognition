from functions.iris.utilities import gaborconvolve
import numpy as np

ksize = 31  # Size of the Gabor filter kernel
sigma = 4.0  # Standard deviation of the Gaussian envelope
# Orientation values for the Gabor filters
theta_values = [0, np.pi/4, np.pi/2, 3*np.pi/4]
lambda_values = [10, 20, 30]  # Wavelength values for the Gabor filters
gamma = 0.5  # Aspect ratio of the elliptical Gaussian envelope
psi = 0


def encode(polar_array, noise_array, minWaveLength, mult=1, sigmaOnf=0.5):
    # Convolve normalised region with Gabor filters
    filter_bank = gaborconvolve(polar_array, minWaveLength, mult, sigmaOnf)

    length = polar_array.shape[1]
    template = np.zeros([polar_array.shape[0], 2 * length])
    h = np.arange(polar_array.shape[0])

    mask = np.zeros(template.shape)
    eleFilt = filter_bank[:, :]

    # Phase quantization
    H1 = np.real(eleFilt) > 0
    H2 = np.imag(eleFilt) > 0

    # If amplitude is close to zero then phase data is not useful,
    H3 = np.abs(eleFilt) < 0.0001
    for i in range(length):
        ja = 2 * i

        # Construct the biometric template
        template[:, ja] = H1[:, i]
        template[:, ja + 1] = H2[:, i]

        # Create noise mask
        mask[:, ja] = noise_array[:, i] | H3[:, i]
        mask[:, ja + 1] = noise_array[:, i] | H3[:, i]

    return template, mask
