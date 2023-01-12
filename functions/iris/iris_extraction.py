from cv2 import imread
from functions.iris.iris_segment import segment
from functions.iris.iris_normalize import normalize
from functions.iris.iris_encode import encode

def iris_extract(im_filename):  
    # Normalisation parameters
	radial_res = 20
	angular_res = 240
	eyelashes_thres = 80
	# Feature encoding parameters
	minWaveLength = 18
	mult = 1
	sigmaOnf = 0.5

	# Perform segmentation
 
	im = imread(im_filename, 0)
	ciriris, cirpupil, imwithnoise = segment(im, eyelashes_thres, use_multiprocess=False)

	# Perform normalization
	polar_array, noise_array = normalize(imwithnoise, ciriris[1], ciriris[0], ciriris[2],
										 cirpupil[1], cirpupil[0], cirpupil[2],
										 radial_res, angular_res)

	# Perform feature encoding
	template, mask = encode(polar_array, noise_array, minWaveLength, mult, sigmaOnf)

	# Return
	return template, mask, im_filename