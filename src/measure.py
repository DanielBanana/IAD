from abc import ABC, abstractmethod
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import argrelextrema

import numpy as np
import cv2
import math

class Measure(ABC):
    """
    Abstract base class for measurement tools, providing common features like Gaussian smoothing
    and finding local extrema.
    """

    @staticmethod
    def extrema2pixel(extrema, origin, direction, projection_length):
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

    @staticmethod
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

    @staticmethod
    def find_local_extrema(gradient, order=1, threshold=10):
        """
        Find all local maxima and minima in a 1D gradient vector.

        Args:
            gradient (numpy.ndarray): Input 1D gradient array.
            order (int): How many points on each side to use for the comparison.
            threshold (float): Threshold for extrema detection.

        Returns:
            tuple: (maxima_indices, minima_indices)
                - maxima_indices (list): Indices of local maxima.
                - minima_indices (list): Indices of local minima.
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

    @staticmethod
    def visualize_projections(image, projection_lines):
        """
        Visualizes the projection lines on the image.

        Args:
            image (numpy.ndarray): Input grayscale image.
            projection_lines (list): List of projection line coordinates.
        """
        # Convert to color for visualization
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw the projection lines
        for start, end in projection_lines:
            cv2.line(image, tuple(start), tuple(end), (0, 0, 255), 1)

        return image
    
    @abstractmethod
    def measure(self):
        """
        Abstract method to perform a measurement.
        This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def visualize(self):
        """
        Abstract method to visualize the measurement.
        This method should be implemented by subclasses.
        """
        pass
    
class CircularMeasureArc(Measure):
    """
    A class to measure and interpolate profiles along a circular arc with radial projections.
    """

    def __init__(
        self,
        image,
        center_row,
        center_col,
        radius,
        angle_start,
        angle_extent,
        annulus_radius,
        num_projections,
        projection_length,
        interpolation="nearest",
        interpolate_profile=True,
        num_interpolated_points=100,
        sigma=1.0,
        threshold=10,
    ):
        self.image = image
        self.center_row = center_row
        self.center_col = center_col
        self.radius = radius
        self.angle_start = angle_start
        self.angle_extent = angle_extent
        self.annulus_radius = annulus_radius
        self.num_projections = num_projections
        self.projection_length = projection_length
        self.interpolation = interpolation
        self.interpolate_profile = interpolate_profile
        self.num_interpolated_points = num_interpolated_points
        self.sigma = sigma
        self.threshold = threshold

    def measure(self):
        """
        Measures the edge profile along a circular arc with radial projections and interpolates the profile.

        Returns:
            tuple: (interpolated_profile, gradient, extrema, projection_lines)
                - interpolated_profile (numpy.ndarray): Interpolated profile values.
                - gradient (numpy.ndarray): Gradient of the interpolated profile.
                - extrema (list): List of local maxima and minima indices.
                - projection_lines (list): List of projection line coordinates.
        """
        # Generate points along the arc
        arc_points = []
        for i in range(self.num_projections + 1):
            theta = self.angle_start + (i / self.num_projections) * self.angle_extent
            x = self.center_col + self.radius * math.cos(theta)
            y = self.center_row + self.radius * math.sin(theta)
            arc_points.append((x, y, theta))

        # Generate radial projection lines and measure profile
        projection_lines = []
        profile = []
        positions = []

        for i, (arc_x, arc_y, theta) in enumerate(arc_points):
            # Calculate the angle for the radial projection line (perpendicular to the arc)
            radial_direction_x = math.cos(theta)
            radial_direction_y = math.sin(theta)

            # Define the start and end points of the radial projection line
            start_x = arc_x + self.projection_length * radial_direction_x
            start_y = arc_y + self.projection_length * radial_direction_y
            end_x = arc_x - self.projection_length * radial_direction_x
            end_y = arc_y - self.projection_length * radial_direction_y

            # Interpolate pixel values along the radial projection line
            line_points = []
            num_steps = int(np.linalg.norm([end_x - start_x, end_y - start_y])) + 1
            for j in range(num_steps):
                t = j / (num_steps - 1)
                x = int(start_x + t * (end_x - start_x))
                y = int(start_y + t * (end_y - start_y))

                # Ensure coordinates are within image bounds
                if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                    line_points.append((x, y))

            # Calculate the mean gray value along the radial projection line
            if line_points:
                gray_values = []
                for (x, y) in line_points:
                    if self.interpolation == "nearest":
                        gray_values.append(self.image[y, x])
                    else:
                        # Bilinear interpolation
                        x1, y1 = int(math.floor(x)), int(math.floor(y))
                        x2, y2 = min(x1 + 1, self.image.shape[1] - 1), min(y1 + 1, self.image.shape[0] - 1)
                        dx = x - x1
                        dy = y - y1
                        interpolated_value = (
                            (1 - dx) * (1 - dy) * self.image[y1, x1] +
                            dx * (1 - dy) * self.image[y1, x2] +
                            (1 - dx) * dy * self.image[y2, x1] +
                            dx * dy * self.image[y2, x2]
                        )
                        gray_values.append(interpolated_value)
                mean_gray = np.mean(gray_values)
            else:
                mean_gray = 0

            profile.append(mean_gray)
            positions.append(i * self.angle_extent / self.num_projections)
            projection_lines.append(((int(start_x), int(start_y)), (int(end_x), int(end_y))))

        # Interpolate the profile
        if self.interpolate_profile:
            # Create interpolation function
            interp_func = interp1d(positions, profile, kind='linear')

            # Generate interpolated positions
            interpolated_positions = np.linspace(0, self.angle_extent, self.num_interpolated_points)

            # Interpolate profile values
            interpolated_profile = interp_func(interpolated_positions)

            # Apply Gaussian smoothing
            if self.sigma > 0.0:
                interpolated_profile = self.gaussian_smooth(interpolated_profile, sigma=self.sigma)

            # Calculate gradient
            gradient = np.gradient(interpolated_profile)

            # Find local extrema
            maxima_indices, minima_indices = self.find_local_extrema(gradient, threshold=self.threshold)
            extrema = [maxima_indices, minima_indices]
        else:
            interpolated_profile = np.asarray(profile)
            if self.sigma > 0.0:
                interpolated_profile = self.gaussian_smooth(interpolated_profile, sigma=self.sigma)

            interpolated_positions = np.array(positions)
            gradient = np.gradient(interpolated_profile)

            # Find local extrema
            maxima_indices, minima_indices = self.find_local_extrema(gradient, threshold=self.threshold)
            extrema = [maxima_indices, minima_indices]

        self.profile = interpolated_profile
        self.gradient = gradient
        self.extrema = extrema
        self.projection_lines = projection_lines
        self.arcPoints = arc_points
        
        return interpolated_profile, gradient, extrema, projection_lines
    
    def visualize(self):
        """
        Visualizes the projection lines on the image.

        Args:
            image (numpy.ndarray): Input grayscale image.
            projection_lines (list): List of projection line coordinates.
        """
        visImage1 = self.visualize_projections(self.image, self.projection_lines)
        cv2.imwrite("images/projectionLines.png", visImage1)

        max_lines, min_lines = self.extrema2pixel(self.extrema, self.origin, self.direction, self.line_length)
        visImage2 = self.visualize_projections(self.image, max_lines)
        cv2.imwrite("images/maxLines.png", visImage2)
        visImage3 = self.visualize_projections(self.image, min_lines)
        cv2.imwrite("images/minLines.png", visImage3)

        # Create a color image for visualization
        vis_image4 = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        # Draw the arc points
        for point in self.arcPoints:
            cv2.circle(vis_image4, (int(round(point[0])), int(round(point[1]))), 3, (0, 0, 255), -1)

        # Draw the projection lines
        for start, end in projection_lines:
            cv2.line(vis_image4, start, end, (0, 255, 0), 1)
        cv2.imwrite("iamges/arcPoints.png", vis_image4)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(8, 4))

        # Plot lines
        ax.plot(self.profile, label="Profile")
        ax.plot(self.gradient, label="Gradient")

        # Plot vertical lines at extrema
        for idx in self.extrema[0]:
            ax.axvline(x=idx, color='red', alpha=0.3)

        for idx in self.extrema[1]:
            ax.axvline(x=idx, color='blue', alpha=0.3)

        ax.set_title("Profile and gradient with local extrema")
        ax.legend()
        fig.savefig("images/profile.png")


class MeasureProfile(Measure):
    """
    A class to measure and interpolate profiles along a line with perpendicular projections.

    Attributes:
        image (numpy.ndarray): Input grayscale image.
        origin (tuple): (x, y) coordinate of the line's origin.
        direction (tuple): (dx, dy) vector defining the line's direction.
        line_length (float): Length of the main line.
        num_projections (int): Number of equidistant perpendicular projection lines.
        projection_length (float): Length of each perpendicular projection line.
        interpolation (str): Interpolation method for pixel values ("nearest" or "linear").
        interpolate_profile (bool): If True, interpolates the profile between projections.
        num_interpolated_points (int): Number of points to interpolate between projections.
        sigma (float): Standard deviation for Gaussian smoothing.
        threshold (float): Threshold for extrema detection.
    """

    def __init__(
        self,
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
        threshold=10,
    ):
        self.image = image
        self.origin = np.array(origin, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.line_length = line_length
        self.num_projections = num_projections
        self.projection_length = projection_length
        self.interpolation = interpolation
        self.interpolate_profile = interpolate_profile
        self.num_interpolated_points = num_interpolated_points
        self.sigma = sigma
        self.threshold = threshold

        # Normalize the direction vector
        direction_norm = np.linalg.norm(self.direction)
        if direction_norm == 0:
            raise ValueError("Direction vector must not be zero.")
        self.direction_unit = self.direction / direction_norm

        # Calculate the perpendicular direction
        self.perpendicular_direction = np.array([-self.direction_unit[1], self.direction_unit[0]])

    def measure(self):
        """
        Measures the edge profile along a line with perpendicular projections and interpolates the profile.

        Returns:
            tuple: (interpolated_profile, gradient, extrema, projection_lines)
                - interpolated_profile (numpy.ndarray): Interpolated profile values.
                - gradient (numpy.ndarray): Gradient of the interpolated profile.
                - extrema (list): List of local maxima and minima indices.
                - projection_lines (list): List of projection line coordinates.
        """
        # Generate points along the main line
        main_line_points = []
        for i in range(self.num_projections + 1):
            t = (i / self.num_projections) * self.line_length
            point = self.origin + t * self.direction_unit
            main_line_points.append(point.astype(int))

        # Generate perpendicular projection lines and measure profile
        projection_lines = []
        profile = []
        positions = []

        for i, point in enumerate(main_line_points):
            # Define the start and end points of the projection line
            start = point + self.projection_length * self.perpendicular_direction
            end = point - self.projection_length * self.perpendicular_direction

            # Interpolate pixel values along the projection line
            line_points = []
            num_steps = int(np.linalg.norm(end - start)) + 1
            for j in range(num_steps):
                t = j / (num_steps - 1)
                x = int(start[0] + t * (end[0] - start[0]))
                y = int(start[1] + t * (end[1] - start[1]))

                # Ensure coordinates are within image bounds
                if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                    line_points.append((x, y))

            # Calculate the mean gray value along the projection line
            if line_points:
                gray_values = [self.image[y, x] for (x, y) in line_points]
                mean_gray = np.mean(gray_values)
            else:
                mean_gray = 0

            profile.append(mean_gray)
            positions.append(i * self.line_length / self.num_projections)
            projection_lines.append((start.astype(int), end.astype(int)))

        # Interpolate the profile
        if self.interpolate_profile:
            # Create interpolation function
            interp_func = interp1d(positions, profile, kind='linear')

            # Generate interpolated positions
            interpolated_positions = np.linspace(0, self.line_length, self.num_interpolated_points)

            # Interpolate profile values
            interpolated_profile = interp_func(interpolated_positions)

            # Apply Gaussian smoothing
            if self.sigma > 0.0:
                interpolated_profile = self.gaussian_smooth(interpolated_profile, sigma=self.sigma)

            # Calculate gradient
            gradient = np.gradient(interpolated_profile)

            # Find local extrema
            maxima_indices, minima_indices = self.find_local_extrema(gradient)
            extrema = [maxima_indices, minima_indices]
        else:
            interpolated_profile = np.asarray(profile)
            if self.sigma > 0.0:
                interpolated_profile = self.gaussian_smooth(interpolated_profile, self.sigma)

            interpolated_positions = np.array(positions)
            gradient = np.gradient(interpolated_profile)

            # Find local extrema
            maxima_indices, minima_indices = self.find_local_extrema(gradient)
            extrema = [maxima_indices, minima_indices]

        self.profile = interpolated_profile
        self.gradient = gradient
        self.extrema = extrema
        self.projection_lines = projection_lines

        return interpolated_profile, gradient, extrema, projection_lines
    
    def visualize(self, image):
        """
        Visualizes the projection lines on the image.

        Args:
            image (numpy.ndarray): Input grayscale image.
            projection_lines (list): List of projection line coordinates.
        """
        visImage1 = self.visualize_projections(image, self.projection_lines)
        cv2.imwrite("images/projectionLines.png", visImage1)

        max_lines, min_lines = self.extrema2pixel(self.extrema, self.origin, self.direction, self.line_length)
        visImage2 = self.visualize_projections(image, max_lines)
        cv2.imwrite("images/maxLines.png", visImage2)
        visImage3 = self.visualize_projections(image, min_lines)
        cv2.imwrite("images/minLines.png", visImage3)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(8, 4))

        # Plot lines
        ax.plot(self.profile, label="Profile")
        ax.plot(self.gradient, label="Gradient")

        # Plot vertical lines at extrema
        for idx in self.extrema[0]:
            ax.axvline(x=idx, color='red', alpha=0.3)

        for idx in self.extrema[1]:
            ax.axvline(x=idx, color='blue', alpha=0.3)

        ax.set_title("Profile and gradient with local extrema")
        ax.legend()
        fig.savefig("images/profile.png")
    
    

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

    # Define the measure profile
    measureProfile = MeasureProfile(
        image=image,
        origin=origin,
        direction=direction,
        line_length=line_length,
        num_projections=num_projections,
        projection_length=projection_length,
        interpolation="nearest",
        interpolate_profile=True,
        num_interpolated_points=line_length,
        sigma=1.0,
        threshold=10,
    )

    # Measure the profile
    interpolated_profile, gradient, extrema, projection_lines = measureProfile.measure()

    print("Interpolated Profile:", interpolated_profile)

    measureProfile.visualize(image)


    ################
    # Measure circle
    ################

    # # Example image (replace with your image)
    # image = np.zeros((500, 500), dtype=np.uint8)
    # cv2.circle(image, (250, 250), 100, 255, -1)  # Draw a white circle for demonstration

    circleImage = cv2.imread("images/gears.png")

    # Define the circular measure arc
    circular_arc = CircularMeasureArc(
        image=circleImage,
        center_row=250,
        center_col=250,
        radius=100,
        angle_start=0,
        angle_extent=math.pi,  # 180 degrees
        annulus_radius=5,
        num_projections=25,
        projection_length=10,
        interpolate_profile=True,
        num_interpolated_points=100,
        interpolation="nearest",
        sigma=1.0,
        threshold=10.0
    )

    # Extract pixel values along the arc
    pixel_values = circular_arc.measure()
    print("Pixel values along the arc:", pixel_values)

    # Visualize the arc
    circular_arc.visualize()
