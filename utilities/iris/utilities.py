import numpy as np
from scipy.ndimage import convolve


def fspecial_gaussian(shape=(3, 3), sigma=1):
    # Compute the center coordinates of the filter shape
    m, n = [(size - 1) / 2 for size in shape]

    # Generate the grid of x and y coordinates
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    # Compute the Gaussian values
    exponent = -(x * x + y * y) / (2 * sigma * sigma)
    gaussian = np.exp(exponent)

    # Set small values to zero for numerical stability
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0

    # Normalize the filter to ensure the sum of values is 1
    sum_gaussian = gaussian.sum()
    if sum_gaussian != 0:
        gaussian /= sum_gaussian

    return gaussian


def canny(image, sigma, vertical, horizontal):
    # Apply Gaussian filter to smooth the image
    filter_size = [6 * sigma + 1, 6 * sigma + 1]
    gaussian = fspecial_gaussian(filter_size, sigma)
    image = convolve(image, gaussian, mode='constant')
    rows, cols = image.shape

    # Compute horizontal and vertical gradients using finite differences
    h = np.concatenate([image[:, 1:cols], np.zeros([rows, 1])], axis=1) - \
        np.concatenate([np.zeros([rows, 1]), image[:, 0:cols - 1]], axis=1)

    v = np.concatenate([image[1:rows, :], np.zeros([1, cols])], axis=0) - \
        np.concatenate([np.zeros([1, cols]), image[0:rows - 1, :]], axis=0)

    # Compute diagonal gradients
    d11 = np.concatenate(
        [image[1:rows, 1:cols], np.zeros([rows - 1, 1])], axis=1)
    d11 = np.concatenate([d11, np.zeros([1, cols])], axis=0)
    d12 = np.concatenate(
        [np.zeros([rows-1, 1]), image[0:rows - 1, 0:cols - 1]], axis=1)
    d12 = np.concatenate([np.zeros([1, cols]), d12], axis=0)
    d1 = d11 - d12

    d21 = np.concatenate(
        [image[0:rows - 1, 1:cols], np.zeros([rows - 1, 1])], axis=1)
    d21 = np.concatenate([np.zeros([1, cols]), d21], axis=0)
    d22 = np.concatenate(
        [np.zeros([rows - 1, 1]), image[1:rows, 0:cols - 1]], axis=1)
    d22 = np.concatenate([d22, np.zeros([1, cols])], axis=0)
    d2 = d21 - d22

    # Compute X and Y gradients
    X = (h + (d1 + d2) / 2) * vertical
    Y = (v + (d1 - d2) / 2) * horizontal

    # Compute gradient magnitude
    gradient = np.sqrt(X * X + Y * Y)

    # Compute gradient orientation
    orientation = np.arctan2(-Y, X)
    negative = orientation < 0
    orientation = orientation * ~negative + (orientation + np.pi) * negative
    orientation = orientation * 180 / np.pi

    return gradient, orientation


def non_max_suppression(input_image, orientation, radius):
    rows, cols = input_image.shape
    output_image = np.zeros([rows, cols])
    integer_radius = np.ceil(radius).astype(int)

    # Pre-calculate x and y offsets relative to center pixel for each orientation angle
    angles = np.arange(181) * np.pi / 180
    x_offset = radius * np.cos(angles)
    y_offset = radius * np.sin(angles)

    # Fractional offset of x_offset relative to integer location
    x_fractional = x_offset - np.floor(x_offset)
    # Fractional offset of y_offset relative to integer location
    y_fractional = y_offset - np.floor(y_offset)

    orientation = np.fix(orientation)

    # Run through the image interpolating gray values on each side of the center pixel
    # to be used for non-maximal suppression
    col, row = np.meshgrid(np.arange(integer_radius, cols - integer_radius),
                           np.arange(integer_radius, rows - integer_radius))

    # Index into precomputed arrays
    orientation_arrays = orientation[row, col].astype(int)

    # x, y location on one side of the point in question
    x = col + x_offset[orientation_arrays]
    y = row - y_offset[orientation_arrays]

    # Get integer pixel locations that surround the location x, y
    floor_x = np.floor(x).astype(int)
    ceil_x = np.ceil(x).astype(int)
    floor_y = np.floor(y).astype(int)
    ceil_y = np.ceil(y).astype(int)

    # Values at integer pixel locations
    top_left = input_image[floor_y, floor_x]
    top_right = input_image[floor_y, ceil_x]
    bottom_left = input_image[ceil_y, floor_x]
    bottom_right = input_image[ceil_y, ceil_x]

    # Bi-linear interpolation to estimate value at x, y
    upper_avg = top_left + \
        x_fractional[orientation_arrays] * (top_right - top_left)
    lower_avg = bottom_left + \
        x_fractional[orientation_arrays] * (bottom_right - bottom_left)
    interpolated_values_1 = upper_avg + \
        y_fractional[orientation_arrays] * (lower_avg - upper_avg)

    # Check the value on the other side
    candidate_region_map = input_image[row, col] > interpolated_values_1

    x = col - x_offset[orientation_arrays]
    y = row + y_offset[orientation_arrays]

    floor_x = np.floor(x).astype(int)
    ceil_x = np.ceil(x).astype(int)
    floor_y = np.floor(y).astype(int)
    ceil_y = np.ceil(y).astype(int)

    top_left = input_image[floor_y, floor_x]
    top_right = input_image[floor_y, ceil_x]
    bottom_left = input_image[ceil_y, floor_x]
    bottom_right = input_image[ceil_y, ceil_x]

    upper_avg = top_left + \
        x_fractional[orientation_arrays] * (top_right - top_left)
    lower_avg = bottom_left + \
        x_fractional[orientation_arrays] * (bottom_right - bottom_left)
    interpolated_values_2 = upper_avg + \
        y_fractional[orientation_arrays] * (lower_avg - upper_avg)

    # Local maximum
    active_map = input_image[row, col] > interpolated_values_2
    active_map = active_map * candidate_region_map
    output_image[row, col] = input_image[row, col] * active_map

    return output_image


def h_threshold(image, threshold1, threshold2):
    rows, cols = image.shape
    total_pixels = rows * cols
    rc_minus_r = total_pixels - rows
    rows_plus_1 = rows + 1
    # Flattening the image into a 1D array
    bw_image = image.ravel()
    edge_pixels = np.where(bw_image > threshold1)
    edge_pixels = edge_pixels[0]
    num_edge_pixels = edge_pixels.size
    # Creating a stack array to store indices
    stack = np.zeros(total_pixels)
    stack[0:num_edge_pixels] = edge_pixels
    stack_pointer = num_edge_pixels

    for k in range(num_edge_pixels):
        bw_image[edge_pixels[k]] = -1

    offsets = np.array([-1, 1, -rows - 1, -rows, -rows +
                       1, rows - 1, rows, rows + 1])

    while stack_pointer != 0:
        v = int(stack[stack_pointer - 1])
        stack_pointer -= 1
        # Check if the index is within valid range
        if rows_plus_1 < v < rc_minus_r:
            indices = offsets + v
            for l in range(8):
                index = indices[l]
                if bw_image[index] > threshold2:
                    stack_pointer += 1
                    stack[stack_pointer - 1] = index
                    bw_image[index] = -1
    # Convert edge points to boolean values (True: edge, False: non-edge)
    bw_image = (bw_image == -1)
    # Reshape the image array back to 2D
    bw_image = np.reshape(bw_image, [rows, cols])

    return bw_image


def iris_preprocess(img):
    image, orient = canny(img, 2, 0, 1)

    # Adjusting the gamma of the image
    adjusted = image - np.min(image)
    adjusted = adjusted / np.max(adjusted)
    adjusted = adjusted ** (1 / 1.9)

    suppressed = non_max_suppression(adjusted, orient, 1.5)
    tresholded = h_threshold(suppressed, 0.2, 0.15)

    return tresholded
