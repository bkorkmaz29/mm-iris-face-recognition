import numpy as np


def iris_distance(template1, mask1, template2, mask2):
    hamming_distance = np.nan
    for shifts in range(-8, 9):
        template1s = shift_bits(template1, shifts)
        mask1s = shift_bits(mask1, shifts)

        mask = np.logical_or(mask1s, mask2)
        nummaskbits = np.sum(mask == 1)
        totalbits = template1s.size - nummaskbits

        C = np.logical_xor(template1s, template2)
        C = np.logical_and(C, np.logical_not(mask))
        bitsdiff = np.sum(C == 1)

        if totalbits == 0:
            hamming_distance = np.nan
        else:
            hd1 = bitsdiff / totalbits
            if hd1 < hamming_distance or np.isnan(hamming_distance):
                hamming_distance = hd1

    return hamming_distance


def shift_bits(template, noshifts):
    new_template = np.zeros(template.shape)
    width = template.shape[1]
    s = 2 * np.abs(noshifts)
    p = width - s

    # Shift
    if noshifts == 0:
        new_template = template

    elif noshifts < 0:
        x = np.arange(p)
        new_template[:, x] = template[:, s + x]
        x = np.arange(p, width)
        new_template[:, x] = template[:, x - p]

    else:
        x = np.arange(s, width)
        new_template[:, x] = template[:, x - s]
        x = np.arange(s)
        new_template[:, x] = template[:, p + x]

    return new_template
