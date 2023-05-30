from utilities.functions import find_bottom_eyelid, find_top_eyelid, gabor_convolve, get_circle_coords, search_inner_bound, search_outer_bound, shift_bits
import numpy as np
import cv2
import PIL.Image
import os


class IrisRec:

    @classmethod
    def segment(cls, eye_image, eyelash_threshold):
        # Search for the inner boundary of the pupil
        pupil_center_row, pupil_center_col, pupil_radius = search_inner_bound(
            eye_image)

        # Search for the outer boundary of the iris
        iris_center_row, iris_center_col, iris_radius = search_outer_bound(
            eye_image, pupil_center_row, pupil_center_col, pupil_radius)

        # Package pupil and iris boundaries
        pupil_center_row = np.round(pupil_center_row).astype(int)
        pupil_center_col = np.round(pupil_center_col).astype(int)
        pupil_radius = np.round(pupil_radius).astype(int)
        iris_center_row = np.round(iris_center_row).astype(int)
        iris_center_col = np.round(iris_center_col).astype(int)
        iris_radius = np.round(iris_radius).astype(int)
        pupil_circle = [pupil_center_row, pupil_center_col, pupil_radius]
        iris_circle = [iris_center_row, iris_center_col, iris_radius]

        # Find top and bottom eyelid regions
        image_shape = eye_image.shape
        eyelid_region_top = np.round(iris_center_row - iris_radius).astype(int)
        eyelid_region_bottom = np.round(
            iris_center_row + iris_radius).astype(int)
        eyelid_region_left = np.round(
            iris_center_col - iris_radius).astype(int)
        eyelid_region_right = np.round(
            iris_center_col + iris_radius).astype(int)
        eyelid_region_top = max(0, eyelid_region_top)
        eyelid_region_left = max(0, eyelid_region_left)
        eyelid_region_bottom = min(image_shape[0] - 1, eyelid_region_bottom)
        eyelid_region_right = min(image_shape[1] - 1, eyelid_region_right)
        iris_region_image = eye_image[eyelid_region_top:eyelid_region_bottom +
                                      1, eyelid_region_left:eyelid_region_right + 1]

        # Find the top eyelid mask
        top_eyelid_mask = find_top_eyelid(
            image_shape, iris_region_image, eyelid_region_top, eyelid_region_left, pupil_center_row, pupil_radius)

        # Find the bottom eyelid mask
        bottom_eyelid_mask = find_bottom_eyelid(
            image_shape, iris_region_image, pupil_center_row, pupil_radius, eyelid_region_top, eyelid_region_left)

        # Mask the eye image
        noisy_image = eye_image.astype(float)
        noisy_image = eye_image + top_eyelid_mask + bottom_eyelid_mask

        # Eliminate eyelashes by thresholding
        eyelash_mask = eye_image < eyelash_threshold
        noisy_image[eyelash_mask] = np.nan

        return iris_circle, pupil_circle, noisy_image

    @classmethod
    def normalize(cls, image, x_iris, y_iris, r_iris, x_pupil, y_pupil, r_pupil):
        rad_res = 22
        ang_res = 239

        # Create the radial and angular grids
        r = np.arange(rad_res)
        theta = np.linspace(0, 2*np.pi, ang_res+1)

        # Calculate displacement of pupil center from the iris center
        dx = x_pupil - x_iris
        dy = y_pupil - y_iris

        if dx <= 0:
            sign_x = -1
        elif dx > 0:
            sign_x = 1

        if dx == 0 and dy > 0:
            sign_x = 1

        a = np.ones(ang_res+1) * (dx**2 + dy**2)

        if dx == 0:
            phi = np.pi/2
        else:
            phi = np.arctan(dy/dx)

        b = sign_x * np.cos(np.pi - phi - theta)

        # Calculate the radius around the iris as a function of the angle
        radius = np.sqrt(a) * b + np.sqrt(a * b**2 - (a - r_iris**2))
        radius = np.array([radius - r_pupil])

        r_matrix = np.dot(np.ones([rad_res, 1]), radius)
        r_matrix = r_matrix * np.dot(np.ones([ang_res+1, 1]),
                                     np.array([np.linspace(0, 1, rad_res)])).transpose()
        r_matrix = r_matrix + r_pupil

        # Exclude values at the boundary of the pupil-iris border and the iris-sclera border
        r_matrix = r_matrix[1: rad_res-1, :]

        # Calculate the Cartesian location of each data point around the circular iris region
        x_cos_matrix = np.dot(np.ones([rad_res-2, 1]),
                              np.array([np.cos(theta)]))
        y_sin_matrix = np.dot(np.ones([rad_res-2, 1]),
                              np.array([np.sin(theta)]))

        x_coordinates = r_matrix * x_cos_matrix
        y_coordinates = r_matrix * y_sin_matrix

        x_coordinates = x_pupil + x_coordinates
        x_coordinates = np.round(x_coordinates).astype(int)
        coords = np.where(x_coordinates >= image.shape[1])
        x_coordinates[coords] = image.shape[1] - 1
        coords = np.where(x_coordinates < 0)
        x_coordinates[coords] = 0

        y_coordinates = y_pupil - y_coordinates
        y_coordinates = np.round(y_coordinates).astype(int)
        coords = np.where(y_coordinates >= image.shape[0])
        y_coordinates[coords] = image.shape[0] - 1
        coords = np.where(y_coordinates < 0)
        y_coordinates[coords] = 0

        # Extract intensity values into the normalized polar representation
        polar_array = image[y_coordinates, x_coordinates]
        polar_array = polar_array / 255

        # Create a noise array with the location of NaNs in polar_array
        polar_noise = np.zeros(polar_array.shape)
        coords = np.where(np.isnan(polar_array))
        polar_noise[coords] = 1

        # Get rid of outlier points in order to write out the circular pattern
        image[y_coordinates, x_coordinates] = 255

        # Get pixel coordinates for the circle around the iris
        iris_x, iris_y = get_circle_coords(
            [x_iris, y_iris], r_iris, image.shape)
        image[iris_y, iris_x] = 255

        # Get pixel coordinates for the circle around the pupil
        pupil_x, pupil_y = get_circle_coords(
            [x_pupil, y_pupil], r_pupil, image.shape)
        image[pupil_y, pupil_x] = 255

        # Replace NaNs before performing feature encoding
        coords = np.where(np.isnan(polar_array))
        polar_array2 = polar_array
        polar_array2[coords] = 0.5
        avg = np.sum(polar_array2) / \
            (polar_array.shape[0] * polar_array.shape[1])
        polar_array[coords] = avg

        return polar_array, polar_noise.astype(bool)

    @classmethod
    def encode(cls, polar_array, noise_array):
        # Apply Gabor filtering to the polar array
        filter_bank = gabor_convolve(
            polar_array, 18, 1, 0.5)

        # Get the length of the polar array
        length = polar_array.shape[1]

        # Initialize the template array
        template = np.zeros([polar_array.shape[0], 2 * length])

        # Initialize the mask array
        mask = np.zeros(template.shape)

        # Get the filtered array from the filter bank
        filtered_array = filter_bank[:, :]

        # Phase quantization
        positive_real = np.real(filtered_array) > 0
        positive_imaginary = np.imag(filtered_array) > 0
        close_to_zero = np.abs(filtered_array) < 0.0001

        # Iterate over the columns of the template
        for i in range(length):
            # Calculate the column index for the template
            column_index = 2 * i

            # Construct the template
            template[:, column_index] = positive_real[:, i]
            template[:, column_index + 1] = positive_imaginary[:, i]

            # Create noise mask
            mask[:, column_index] = noise_array[:, i] | close_to_zero[:, i]
            mask[:, column_index + 1] = noise_array[:, i] | close_to_zero[:, i]

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
        # Initialize the hamming distance
        distance = np.nan

        # Iterate over different shifts to find the best match
        for shift in range(-8, 9):
            # Shift the bits of feature1 and corresponding mask
            feature1_shifted = shift_bits(feature1, shift)
            mask1_shifted = shift_bits(mask1, shift)

            # Compute the combined mask
            mask = np.logical_or(mask1_shifted, mask2)

            # Count the number of masked bits
            mask_bits = np.sum(mask == 1)

            # Calculate the total number of non-masked bits
            total_bits = feature2.size - mask_bits

            # Compute the Hamming distance between feature1_shifted and feature2
            C = np.logical_xor(feature1_shifted, feature2)
            C = np.logical_and(C, np.logical_not(mask))
            bits_diff = np.sum(C == 1)

            # Calculate the Hamming distance ratio
            if total_bits == 0:
                distance = np.nan
            else:
                hd = bits_diff / total_bits
                if hd < distance or np.isnan(distance):
                    distance = hd

        return distance
