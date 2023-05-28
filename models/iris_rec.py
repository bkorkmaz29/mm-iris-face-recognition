from utilities.functions import find_bottom_eyelid, find_top_eyelid, gabor_convolve, get_circle_coords, search_inner_bound, search_outer_bound, shift_bits
import numpy as np
import cv2
import PIL.Image
import os


class IrisRec:

    @classmethod
    def segment(cls, eye_image, eyelash_threshold):
        rowp, colp, rp = search_inner_bound(eye_image)
        row, col, r = search_outer_bound(eye_image, rowp, colp, rp)

        # Package pupil and iris boundaries
        rowp = np.round(rowp).astype(int)
        colp = np.round(colp).astype(int)
        rp = np.round(rp).astype(int)
        row = np.round(row).astype(int)
        col = np.round(col).astype(int)
        r = np.round(r).astype(int)
        pupil_circle = [rowp, colp, rp]
        iris_circle = [row, col, r]

        # Find top and bottom eyelid
        image_shape = eye_image.shape
        irl = np.round(row - r).astype(int)
        iru = np.round(row + r).astype(int)
        icl = np.round(col - r).astype(int)
        icu = np.round(col + r).astype(int)
        if irl < 0:
            irl = 0
        if icl < 0:
            icl = 0
        if iru >= image_shape[0]:
            iru = image_shape[0] - 1
        if icu >= image_shape[1]:
            icu = image_shape[1] - 1
        iris_image = eye_image[irl: iru + 1, icl: icu + 1]

        mask_top = find_top_eyelid(image_shape, iris_image, irl, icl, rowp, rp)
        mask_bottom = find_bottom_eyelid(
            image_shape, iris_image, rowp, rp, irl, icl)
        # Mask the eye image
        img_noise = eye_image.astype(float)
        img_noise = eye_image + mask_top + mask_bottom
        # Eliminate eyelashes by threshold
        ref = eye_image < eyelash_threshold
        coords = np.where(ref == 1)
        img_noise[coords] = np.nan

        return iris_circle, pupil_circle, img_noise

    @classmethod
    def normalize(cls, image, x_iris, y_iris, r_iris, x_pupil, y_pupil, r_pupil):
        rad_res = 22
        ang_res = 239
        r = np.arange(rad_res)
        theta = np.linspace(0, 2*np.pi, ang_res+1)

        # Calculate displacement of pupil center from the iris center
        ox = x_pupil - x_iris
        oy = y_pupil - y_iris

        if ox <= 0:
            sgn = -1
        elif ox > 0:
            sgn = 1
        if ox == 0 and oy > 0:
            sgn = 1
        a = np.ones(ang_res+1) * (ox**2 + oy**2)
        if ox == 0:
            phi = np.pi/2
        else:
            phi = np.arctan(oy/ox)
        b = sgn * np.cos(np.pi - phi - theta)
        # Calculate radius around the iris as a function of the angle
        r = np.sqrt(a)*b + np.sqrt(a*b**2 - (a - r_iris**2))
        r = np.array([r - r_pupil])

        rmat = np.dot(np.ones([rad_res, 1]), r)
        rmat = rmat * np.dot(np.ones([ang_res+1, 1]),
                             np.array([np.linspace(0, 1, rad_res)])).transpose()
        rmat = rmat + r_pupil

        # Exclude values at the boundary of the pupil iris border, and the iris scelra border
        rmat = rmat[1: rad_res-1, :]
        # Calculate cartesian location of each data point around the circular iris region
        xcosmat = np.dot(np.ones([rad_res-2, 1]),
                         np.array([np.cos(theta)]))
        xsinmat = np.dot(np.ones([rad_res-2, 1]),
                         np.array([np.sin(theta)]))

        xo = rmat * xcosmat
        yo = rmat * xsinmat

        xo = x_pupil + xo
        xo = np.round(xo).astype(int)
        coords = np.where(xo >= image.shape[1])
        xo[coords] = image.shape[1] - 1
        coords = np.where(xo < 0)
        xo[coords] = 0

        yo = y_pupil - yo
        yo = np.round(yo).astype(int)
        coords = np.where(yo >= image.shape[0])
        yo[coords] = image.shape[0] - 1
        coords = np.where(yo < 0)
        yo[coords] = 0

        # Extract intensity values into the normalised polar representation through
        # interpolation
        polar_array = image[yo, xo]
        polar_array = polar_array / 255
        # Create noise array with location of NaNs in polar_array
        polar_noise = np.zeros(polar_array.shape)
        coords = np.where(np.isnan(polar_array))
        polar_noise[coords] = 1
        # Get rid of outling points in order to write out the circular pattern
        image[yo, xo] = 255
        # Get pixel coords for circle around iris
        x, y = get_circle_coords([x_iris, y_iris], r_iris, image.shape)
        image[y, x] = 255
        # Get pixel coords for circle around pupil
        xp, yp = get_circle_coords([x_pupil, y_pupil], r_pupil, image.shape)
        image[yp, xp] = 255
        # Replace NaNs before performing feature encoding
        coords = np.where((np.isnan(polar_array)))
        polar_array2 = polar_array
        polar_array2[coords] = 0.5
        avg = np.sum(polar_array2) / \
            (polar_array.shape[0] * polar_array.shape[1])
        polar_array[coords] = avg

        return polar_array, polar_noise.astype(bool)

    @classmethod
    def encode(cls, polar_array, noise_array):
        # Convolve normalised region with Gabor filters
        filter_bank = gabor_convolve(
            polar_array,  18, 1, 0.5)

        length = polar_array.shape[1]
        template = np.zeros([polar_array.shape[0], 2 * length])
        h = np.arange(polar_array.shape[0])

        mask = np.zeros(template.shape)
        eleFilt = filter_bank[:, :]

        # Phase quantization
        H1 = np.real(eleFilt) > 0
        H2 = np.imag(eleFilt) > 0
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

    @classmethod
    def get_features(cls, image_dir):
        image = cv2.imread(image_dir, 0)

        iris_circle, pupil_circle, masked_image = cls.segment(
            image, 80)

        polar_array, noise_array = cls.normalize(masked_image, iris_circle[1], iris_circle[0], iris_circle[2],
                                                 pupil_circle[1], pupil_circle[0], pupil_circle[2])

        feature, feature_mask = cls.encode(polar_array, noise_array)

        return feature, feature_mask, image_dir

    @classmethod
    def cal_distance(cls, feature1, mask1, feature2, mask2):
        hamming_distance = np.nan
        for shifts in range(-8, 9):
            feature2s = shift_bits(feature1, shifts)
            mask1s = shift_bits(mask1, shifts)

            mask = np.logical_or(mask1s, mask2)
            mask_bits = np.sum(mask == 1)
            total_bits = feature2s.size - mask_bits

            C = np.logical_xor(feature2s, feature2)
            C = np.logical_and(C, np.logical_not(mask))
            bits_diff = np.sum(C == 1)

            if total_bits == 0:
                hamming_distance = np.nan
            else:
                hd1 = bits_diff / total_bits
                if hd1 < hamming_distance or np.isnan(hamming_distance):
                    hamming_distance = hd1

        return hamming_distance
