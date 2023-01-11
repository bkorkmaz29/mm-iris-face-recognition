import numpy as np
from os import listdir
from fnmatch import filter
import scipy.io as sio
from multiprocessing import Pool, cpu_count
from itertools import repeat
import re
import warnings
warnings.filterwarnings("ignore")

def calHammingDist(template1, mask1, template2, mask2):
	# Initialize
	hd = np.nan

	# Shift template left and right, use the lowest Hamming distance
	for shifts in range(-8,9):
		template1s = shiftbits(template1, shifts)
		mask1s = shiftbits(mask1, shifts)

		mask = np.logical_or(mask1s, mask2)
		nummaskbits = np.sum(mask==1)
		totalbits = template1s.size - nummaskbits

		C = np.logical_xor(template1s, template2)
		C = np.logical_and(C, np.logical_not(mask))
		bitsdiff = np.sum(C==1)

		if totalbits==0:
			hd = np.nan
		else:
			hd1 = bitsdiff / totalbits
			if hd1 < hd or np.isnan(hd):
				hd = hd1

	# Return
	return hd

def shiftbits(template, noshifts):
	# Initialize
	templatenew = np.zeros(template.shape)
	width = template.shape[1]
	s = 2 * np.abs(noshifts)
	p = width - s

	# Shift
	if noshifts == 0:
		templatenew = template

	elif noshifts < 0:
		x = np.arange(p)
		templatenew[:, x] = template[:, s + x]
		x = np.arange(p, width)
		templatenew[:, x] = template[:, x - p]

	else:
		x = np.arange(s, width)
		templatenew[:, x] = template[:, x - s]
		x = np.arange(s)
		templatenew[:, x] = template[:, p + x]

	# Return
	return templatenew

def matchingPool(file_temp_name, template_extr, mask_extr, temp_dir):
    # Load each account
    idx = int(re.search(r'\d+', file_temp_name).group())
    temp_name = temp_dir + str(idx) + "/i" + file_temp_name + ".mat"
    data_template = sio.loadmat(temp_name)
    template = data_template['template']
    mask = data_template['mask']
	
    # Calculate the Hamming distance
    hm_dist = calHammingDist(template_extr, mask_extr, template, mask)
    return (file_temp_name, hm_dist)
