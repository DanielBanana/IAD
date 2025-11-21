import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import argrelextrema

import numpy as np
import cv2
import math

class CircularMeasureArc:
    """
    A class to represent a circular measure arc for extracting pixel values along an arc.

    Attributes:
        center_row (float): Row coordinate of the center.
        center_col (float): Column coordinate of the center.
        radius (float): Radius of the circular arc.
        angle_start (float): Start angle of the arc in radians.
        angle_extent (float): Angular extent of the arc in radians.
        annulus_radius (float): Width of the annulus (ring) around the arc.
        width (int): Width of the image or ROI.
        height (int): Height of the image or ROI.
        interpolation (str): Interpolation method ('nearest' or 'linear').
        measure_handle (int): Optional handle for the measure object.
    """

    def __init__(
        self,
        center_row,
        center_col,
        radius,
        angle_start,
        angle_extent,
        annulus_radius,
        width,
        height,
        interpolation="nearest",
        measure_handle=None,
    ):
        self.center_row = center_row
        self.center_col = center_col
        self.radius = radius
        self.angle_start = angle_start
        self.angle_extent = angle_extent
        self.annulus_radius = annulus_radius
        self.width = width
        self.height = height
        self.interpolation = interpolation
        self.measure_handle = measure_handle

    def extract_arc_pixels(self, image):
        """
        Extract pixel values along the circular arc.

        Args:
            image (numpy.ndarray): Input grayscale image.

        Returns:
            numpy.ndarray: Array of pixel values along the arc.
        """
        # Generate points along the arc
        arc_points = []
        num_points = 100  # Number of points to sample along the arc
        for i in range(num_points + 1):
            theta = self.angle_start + (i / num_points) * self.angle_extent
            x = self.center_col + self.radius * math.cos(theta)
            y = self.center_row + self.radius * math.sin(theta)
            arc_points.append((x, y))

        # Extract pixel values along the arc
        pixel_values = []
        for x, y in arc_points:
            # Ensure coordinates are within image bounds
            if 0 <= x < self.width and 0 <= y < self.height:
                if self.interpolation == "nearest":
                    # Nearest-neighbor interpolation
                    pixel_value = image[int(round(y)), int(round(x))]
                else:
                    # Bilinear interpolation (approximation)
                    x1, y1 = int(math.floor(x)), int(math.floor(y))
                    x2, y2 = min(x1 + 1, self.width - 1), min(y1 + 1, self.height - 1)

                    # Calculate weights
                    dx = x - x1
                    dy = y - y1

                    # Interpolate
                    pixel_value = (
                        (1 - dx) * (1 - dy) * image[y1, x1] +
                        dx * (1 - dy) * image[y1, x2] +
                        (1 - dx) * dy * image[y2, x1] +
                        dx * dy * image[y2, x2]
                    )

                pixel_values.append(pixel_value)
            else:
                pixel_values.append(np.nan)  # Out of bounds

        return np.array(pixel_values)

    def visualize_arc(self, image):
        """
        Visualize the circular arc on the image.

        Args:
            image (numpy.ndarray): Input grayscale image.
        """
        # Create a color image for visualization
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Generate points along the arc
        arc_points = []
        num_points = 100
        for i in range(num_points + 1):
            theta = self.angle_start + (i / num_points) * self.angle_extent
            x = self.center_col + self.radius * math.cos(theta)
            y = self.center_row + self.radius * math.sin(theta)
            arc_points.append((int(round(x)), int(round(y))))

        # Draw the arc
        for i in range(len(arc_points) - 1):
            cv2.line(vis_image, arc_points[i], arc_points[i + 1], (0, 0, 255), 1)

        # Draw the annulus (ring)
        cv2.circle(vis_image, (int(round(self.center_col)), int(round(self.center_row))),
                   int(round(self.radius + self.annulus_radius)), (0, 255, 0), 1)
        cv2.circle(vis_image, (int(round(self.center_col)), int(round(self.center_row))),
                   int(round(self.radius - self.annulus_radius)), (0, 255, 0), 1)

        # Display the result
        cv2.imshow('Circular Measure Arc', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def find_local_extrema(gradient, order=1, threshold = 10):
    """
    Find all local maxima and minima in a 1D gradient vector.

    Args:
        gradient (numpy.ndarray): Input 1D gradient array.
        order (int): How many points on each side to use for the comparison.

    Returns:
        tuple: (maxima_indices, minima_indices)
            - maxima_indices (numpy.ndarray): Indices of local maxima.
            - minima_indices (numpy.ndarray): Indices of local minima.
    """
    # Find local maxima (peaks)
    _maxima_indices = argrelextrema(gradient, np.greater, order=order)[0]
    maxima_indices = []
    for idx in _maxima_indices:
        if np.abs(gradient[idx]) > threshold:
            maxima_indices.append(idx)
    # Find local minima (troughs)
    _minima_indices = argrelextrema(gradient, np.less, order=order)[0]
    minima_indices = []
    for idx in _minima_indices:
        if np.abs(gradient[idx]) > threshold:
            minima_indices.append(idx)
    return maxima_indices, minima_indices

def gaussian_smooth(array, sigma=1.0, mode='reflect', truncate=4.0):
    """
    Smooths a NumPy array using a Gaussian filter.

    Args:
        array (numpy.ndarray): Input array (1D, 2D, or n-dimensional).
        sigma (float): Standard deviation for the Gaussian kernel.
        mode (str): How to handle array borders ('reflect', 'constant', 'nearest', 'mirror', 'wrap').
        truncate (float): Truncate the filter at this many standard deviations.

    Returns:
        numpy.ndarray: Smoothed array.
    """
    if array.ndim == 1:
        # Use gaussian_filter1d for 1D arrays
        smoothed_array = gaussian_filter1d(array, sigma=sigma, mode=mode, truncate=truncate)
    else:
        # Use gaussian_filter for n-dimensional arrays
        smoothed_array = gaussian_filter(array, sigma=sigma, mode=mode, truncate=truncate)
    return smoothed_array

def measure_and_interpolate_profile(
    image,
    origin,
    direction,
    line_length,
    num_projections,
    projection_length,
    interpolation="nearest",
    interpolate_profile=True,
    num_interpolated_points=100,
    sigma=1.0,
    threshold=10
):
    """
    Measures the edge profile along a line with perpendicular projections and interpolates the profile.

    Args:
        image (numpy.ndarray): Input grayscale image.
        origin (tuple): (x, y) coordinate of the line's origin.
        direction (tuple): (dx, dy) vector defining the line's direction.
        line_length (float): Length of the main line.
        num_projections (int): Number of equidistant perpendicular projection lines.
        projection_length (float): Length of each perpendicular projection line.
        interpolation (str): Interpolation method for pixel values ("nearest" or "linear").
        interpolate_profile (bool): If True, interpolates the profile between projections.
        num_interpolated_points (int): Number of points to interpolate between projections.

    Returns:
        tuple: (profile, interpolated_profile, projection_lines, interpolated_positions)
            - profile (list): Mean gray values at each projection line.
            - interpolated_profile (numpy.ndarray): Interpolated profile values.
            - projection_lines (list): List of projection line coordinates.
            - interpolated_positions (numpy.ndarray): Positions along the main line for interpolated values.
    """
    # Normalize the direction vector
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        raise ValueError("Direction vector must not be zero.")
    direction_unit = np.array(direction, dtype=float) / direction_norm

    # Generate points along the main line
    main_line_points = []
    for i in range(num_projections + 1):
        t = (i / num_projections) * line_length
        point = origin + t * direction_unit
        main_line_points.append(point.astype(int))

    # Generate perpendicular projection lines and measure profile
    projection_lines = []
    profile = []
    positions = []

    for i, point in enumerate(main_line_points):
        # Define the start and end points of the projection line
        start = point + projection_length * np.array([-direction_unit[1], direction_unit[0]])
        end = point - projection_length * np.array([-direction_unit[1], direction_unit[0]])

        # Interpolate pixel values along the projection line
        line_points = []
        num_steps = int(np.linalg.norm(end - start)) + 1
        for j in range(num_steps):
            t = j / (num_steps - 1)
            x = int(start[0] + t * (end[0] - start[0]))
            y = int(start[1] + t * (end[1] - start[1]))

            # Ensure coordinates are within image bounds
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                line_points.append((x, y))

        # Calculate the mean gray value along the projection line
        if line_points:
            gray_values = [image[y, x] for (x, y) in line_points]
            mean_gray = np.mean(gray_values)
        else:
            mean_gray = 0

        profile.append(mean_gray)
        positions.append(i * line_length / num_projections)
        projection_lines.append((start.astype(int), end.astype(int)))

    # Interpolate the profile
    if interpolate_profile:
        # Create interpolation function
        interp_func = interp1d(positions, profile, kind='linear')

        # Generate interpolated positions
        interpolated_positions = np.linspace(0, line_length, num_interpolated_points)

        # Interpolate profile values
        interpolated_profile = interp_func(interpolated_positions)
        if sigma > 0.0:
            interpolated_profile = gaussian_smooth(interpolated_profile, sigma=sigma)
        gradient = np.gradient(interpolated_profile)
        maximaIndices, minimaIndices = find_local_extrema(gradient=gradient, threshold=threshold)
    else:
        interpolated_profile = np.asarray(profile)
        if sigma > 0.0:
            interpolated_profile = gaussian_smooth(interpolated_profile, sigma=sigma)
        interpolated_profile = gaussian_smooth(interpolated_profile, sigma=sigma, threshold=threshold)
        interpolated_positions = np.array(positions)
        gradient = np.gradient(interpolated_profile)
        maximaIndices, minimaIndices = find_local_extrema(gradient=gradient)

    return interpolated_profile, gradient, [maximaIndices, minimaIndices], projection_lines

def visualize_projections(image, projection_lines):
    """
    Visualizes the projection lines on the image.

    Args:
        image (numpy.ndarray): Input grayscale image.
        projection_lines (list): List of projection line coordinates.
    """
    # Convert to color for visualization
    vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw the projection lines
    for start, end in projection_lines:
        cv2.line(vis_image, tuple(start), tuple(end), (0, 0, 255), 1)

    # Display the result
    cv2.imshow('Projection Lines', vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_profiles(profile, gradient, extrema):

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot lines
    ax.plot(profile, label="Profile")
    ax.plot(gradient, label="Gradient")

    # Plot vertical lines at extrema
    for idx in extrema[0]:
        ax.axvline(x=idx, color='red', alpha=0.3)

    for idx in extrema[1]:
        ax.axvline(x=idx, color='blue', alpha=0.3)

    ax.set_title("Profile and gradient with local extrema")
    ax.legend()
    fig.savefig("images/profile.png")

def extrema2pixel(extrema, origin, direction, line_length, projection_length):
    # Normalize the direction vector
    direction_norm = np.linalg.norm(direction)
    if direction_norm == 0:
        raise ValueError("Direction vector must not be zero.")
    direction_unit = np.array(direction, dtype=float) / direction_norm

    # Generate points along the main line
    maxima_points = []
    minima_points = []
    max_projection_lines = []
    min_projection_lines = []
    for i in extrema[0]:
        point = origin + i * direction_unit
        maxima_points.append(point.astype(int))
    for i in extrema[1]:
        point = origin + i * direction_unit
        minima_points.append(point.astype(int))

    for i, point in enumerate(maxima_points):
        # Define the start and end points of the projection line
        start = point + projection_length * np.array([-direction_unit[1], direction_unit[0]])
        end = point - projection_length * np.array([-direction_unit[1], direction_unit[0]])
        max_projection_lines.append((start.astype(int), end.astype(int)))
    for i, point in enumerate(minima_points):
        # Define the start and end points of the projection line
        start = point + projection_length * np.array([-direction_unit[1], direction_unit[0]])
        end = point - projection_length * np.array([-direction_unit[1], direction_unit[0]])
        min_projection_lines.append((start.astype(int), end.astype(int)))
    return max_projection_lines, min_projection_lines

# Example usage
if __name__ == "__main__":
    # Load a grayscale image
    image = cv2.imread('images/object.png', cv2.IMREAD_GRAYSCALE)

    # Define the line parameters
    origin = (15,10)
    direction = (0,1)  # Direction vector (dx, dy)
    line_length = 70
    num_projections = 70
    projection_length = 10

    # Measure and interpolate the edge profile
    profile, gradient, extrema, projection_lines = measure_and_interpolate_profile(
        image, origin, direction, line_length, num_projections, projection_length,
        interpolate_profile=True, num_interpolated_points=line_length, sigma=0.0, threshold=10
    )

    # Plot the profiles
    plot_profiles(profile, gradient, extrema)


    # Visualize the projection lines
    visualize_projections(image, projection_lines)

    max_lines, min_lines = extrema2pixel(extrema, origin, direction, line_length, projection_length)

    visualize_projections(image, max_lines)
    visualize_projections(image, min_lines)

    ################
    # Measure circle
    ################

    # Example image (replace with your image)
    image = np.zeros((500, 500), dtype=np.uint8)
    cv2.circle(image, (250, 250), 100, 255, -1)  # Draw a white circle for demonstration

    # Define the circular measure arc
    circular_arc = CircularMeasureArc(
        center_row=250,
        center_col=250,
        radius=100,
        angle_start=0,
        angle_extent=math.pi,  # 180 degrees
        annulus_radius=5,
        width=image.shape[1],
        height=image.shape[0],
        interpolation="nearest",
    )

    # Extract pixel values along the arc
    pixel_values = circular_arc.extract_arc_pixels(image)
    print("Pixel values along the arc:", pixel_values)

    # Visualize the arc
    circular_arc.visualize_arc(image)
