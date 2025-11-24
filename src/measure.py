from abc import ABC, abstractmethod
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import argrelextrema

import numpy as np
import cv2
import math
from functools import partial

class Measure:
    """
    Abstract base class for measurement tools, providing common features like Gaussian smoothing
    and finding local extrema.
    """
    def __init__(self,
                 startPXL,
                 endPXL,
                 startCO,
                 endCO,
                 profilePoints,
                 profileValues,
                 measurePositions,
                 projectionLines,
                 projectionWidth,
                 sigma,
                 threshold):
        self.startPXL = startPXL
        self.endPXL = endPXL
        self.startCO = startCO
        self.endCO = endCO
        self.profilePoints = profilePoints
        self.profileValues = profileValues
        self.measurePositions = measurePositions
        self.projectionLines = projectionLines
        self.projectionWidth = projectionWidth
        self.sigma = sigma
        self.threshold = threshold

    @staticmethod
    def plotLines(image, lines):
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
        for start, end in lines:
            cv2.line(image, tuple(start), tuple(end), (0, 0, 255), 1)
        return image
    
    @staticmethod
    def plotValues(values, gradient=None) -> Figure:
        # Create a figure with two subplots (2 rows, 1 column)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # Plot the values on the top subplot
        ax1.plot(values, 'b-', label='Values')
        ax1.set_xlabel('Profile')
        ax1.set_ylabel('Values', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.legend(loc='upper left')

        # Plot the gradient on the bottom subplot
        if gradient is not None:
            ax2.plot(gradient, 'r--', label='Gradient')
            ax2.set_xlabel('Profile')
            ax2.set_ylabel('Gradient', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.legend(loc='upper left')

        # Set a title for the entire figure
        fig.suptitle('Values and Gradient')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        return fig

    @staticmethod
    def gaussianSmoothing(array, sigma=1.0, mode='reflect', truncate=4.0):
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
    def findLocalExtrema(gradient, order=1, threshold=10):
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
        # maxima_indices = np.where(gradient > threshold)[0]
        # minima_indices = np.where(gradient < -threshold)[0]

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

class LineMeasure(Measure):
    def __init__(self, 
                 image,
                 origin,
                 direction,
                 length,
                 projectionWidth,
                 nProjections,
                 sigma,
                 threshold):
        self.image = image
        self.originPXL = np.round(origin)
        self.originCO = origin
        self.direction = direction
        self.length = length
        self.projectionWidth = projectionWidth
        self.nProjections = nProjections
        if sigma <= 0.0:
            self.smooth = False
        # Normalize the direction vector
        direction_norm = np.linalg.norm(self.direction)
        if direction_norm == 0:
            raise ValueError("Direction vector must not be zero.")
        self.direction_unit = self.direction / direction_norm
        # Calculate the perpendicular direction
        self.perpendicular_direction = np.array([-self.direction_unit[1], self.direction_unit[0]])

        profilePoints = self.getProfilePoints(nProjections)

        projectionLines, profilePoints, profileValues, measurePositions = self.getProjectionLines(profilePoints, sigma, interpolate=True, smooth=True)

        super().__init__(
            startPXL=np.round(origin),
            endPXL=np.round(profilePoints[-1]),
            startCO=origin,
            endCO=profilePoints[-1],
            profilePoints=profilePoints,
            profileValues=profileValues,
            measurePositions=measurePositions,
            projectionLines=projectionLines,
            projectionWidth=projectionWidth,
            sigma=sigma,
            threshold=threshold)
        
    def getProfilePoints(self, nPoints):
        points = []
        for i in range(nPoints + 1):
            t = (i / nPoints) * self.length
            point = self.originCO + t * self.direction_unit
            points.append(point.astype(int))
        return np.asarray(points)
    
    def getProjectionLines(self, profilePoints, sigma=0.0, interpolate=False, smooth=False):
        projectionLines = []
        profileValues = []
        measurePositions = []
        for i, point in enumerate(profilePoints):
            # Define the start and end points of the projection line
            start = point + self.projectionWidth * self.perpendicular_direction # TODO: FUNCTION INSTEAD OF FIXED VARIABLE FOR PERPENDICULAR DIRECTION FOR CURVED PROFILE
            end = point - self.projectionWidth * self.perpendicular_direction

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

            profileValues.append(mean_gray)
            measurePositions.append(i * self.length / self.nProjections) # how far along the profile the points are in PXL
            projectionLines.append((start.astype(int), end.astype(int)))

        if interpolate:
            # Create interpolation function
            interp_func = interp1d(measurePositions, profileValues, kind='linear')
            # Generate interpolated positions
            interpolated_positions = np.linspace(0, self.length, self.length+1)

            # Interpolate profile values
            profileValues = interp_func(interpolated_positions)
            profilePoints = self.getProfilePoints(self.length)

        if smooth:
            profileValues = self.gaussianSmoothing(profileValues, sigma)

        return projectionLines, profilePoints, profileValues, measurePositions

    def getEdges(self):
        self.gradient = np.sqrt(2*np.pi) * np.gradient(self.profileValues)
        self.extrema = self.findLocalExtrema(self.gradient, order=1, threshold=self.threshold)
        self.lightEdges = []
        self.darkEdges = []
        maxis = self.extrema[0]
        self.lightEdges,_,_,_ = self.getProjectionLines(self.profilePoints[maxis], sigma=0.0)

        minis = self.extrema[1]
        self.darkEdges,_,_,_ = self.getProjectionLines(self.profilePoints[minis], sigma=0.0)

        return [self.darkEdges, self.lightEdges], self.gradient

    def visualize(self):
        fig1 = self.plotValues(self.profileValues, self.gradient)
        fig1.savefig("images/ValuesAndGradient.png")
        visImage1 = self.plotLines(self.image.copy(), self.projectionLines)
        cv2.imwrite("images/projectionLines.png", visImage1)
        visImage2 = self.plotLines(self.image.copy(), self.lightEdges)
        cv2.imwrite("images/lightEdges.png", visImage2)
        visImage3 = self.plotLines(self.image.copy(), self.darkEdges)
        cv2.imwrite("images/darkEdges.png", visImage3)
    

class CurveMeasure(Measure):
    def __init__(self, 
                 image,
                 center,
                 radius,
                 angleStart,
                 angleExtent,
                 projectionWidth,
                 nProjections,
                 sigma,
                 threshold):
        self.image = image
        self.centerPXL = np.round(center)
        self.centerCO = center
        self.radius = radius
        self.angleStart = angleStart
        self.angleExtent = angleExtent
        self.projectionWidth = projectionWidth
        self.nProjections = nProjections
        if sigma <= 0.0:
            self.smooth = False
        # Normalize the direction vector

        # Calculate the perpendicular direction
        profilePoints, profileAngles = self.getProfilePoints(nProjections)

        projectionLines, profilePoints, profileAngles, profileValues, measurePositions = self.getProjectionLines(profilePoints, profileAngles, sigma, interpolate=True, smooth=True)
        self.profileAngles = profileAngles

        super().__init__(
            startPXL=np.round(profilePoints[0]),
            endPXL=np.round(profilePoints[-1]),
            startCO=profilePoints[0],
            endCO=profilePoints[-1],
            profilePoints=profilePoints,
            profileValues=profileValues,
            measurePositions=measurePositions,
            projectionLines=projectionLines,
            projectionWidth=projectionWidth,
            sigma=sigma,
            threshold=threshold)
        
    def getRadiusVector(self, point):
        direction = point - self.centerCO
        direction_norm = np.linalg.norm(direction)
        direction_unit = direction / direction_norm
        return direction_unit
    
    def getPerpendicularDirections(self, directions):
        return np.stack([-directions[:, 1], directions[:, 0]], axis=1) 

    def getProfilePoints(self, nPoints):
        points = []
        angles = []
        for i in range(nPoints + 1):
            angle = i *  self.angleExtent/nPoints
            point = np.array(
                [self.centerCO[0] + self.radius * np.cos(self.angleStart + angle),
                 self.centerCO[1] + self.radius * np.sin(self.angleStart + angle)]
            )
            points.append(point.astype(int))  
            angles.append(angle)
        return np.asarray(points), np.asarray(angles)
    
    def getProjectionLines(self, profilePoints, profileAngles, sigma=0.0, interpolate=False, smooth=False):
        projectionLines = []
        profileValues = []
        measurePositions = []
        for i, point in enumerate(profilePoints):
            perpendicularDirection = self.getRadiusVector(point)
            # direction = np.array([-perpendicularDirection[1], perpendicularDirection[0]])
            # Define the start and end points of the projection line
            start = point + self.projectionWidth * perpendicularDirection # TODO: FUNCTION INSTEAD OF FIXED VARIABLE FOR PERPENDICULAR DIRECTION FOR CURVED PROFILE
            end = point - self.projectionWidth * perpendicularDirection

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

            profileValues.append(mean_gray)
            projectionLines.append((start.astype(int), end.astype(int)))

        if interpolate:
            # Create interpolation function
            interp_func = interp1d(profileAngles, profileValues, kind='linear')
            # Generate interpolated positions
            length = np.abs(profilePoints[-1]-profilePoints[0]).max() # Over how many pixel does the curve measure go -> Interpolate to that value
            interpolated_positions = np.linspace(profileAngles[0], profileAngles[-1], length+1)

            # Interpolate profile values
            profileValues = interp_func(interpolated_positions)
            profilePoints, profileAngles = self.getProfilePoints(length)

        if smooth:
            profileValues = self.gaussianSmoothing(profileValues, sigma)

        return projectionLines, profilePoints, profileAngles, profileValues, measurePositions

    def getEdges(self):
        self.gradient = np.sqrt(2*np.pi) * np.gradient(self.profileValues)
        self.extrema = self.findLocalExtrema(self.gradient, order=1, threshold=self.threshold)
        self.lightEdges = []
        self.darkEdges = []
        maxis = self.extrema[0]
        self.lightEdges,_,_,_,_ = self.getProjectionLines(self.profilePoints[maxis], self.profileAngles[maxis], sigma=0.0)

        minis = self.extrema[1]
        self.darkEdges,_,_,_,_ = self.getProjectionLines(self.profilePoints[minis], self.profileAngles[minis], sigma=0.0)

        return [self.darkEdges, self.lightEdges], self.gradient

    def visualize(self):
        fig1 = self.plotValues(self.profileValues, self.gradient)
        fig1.savefig("images/ValuesAndGradient.png")
        visImage1 = self.plotLines(self.image.copy(), self.projectionLines)
        cv2.imwrite("images/projectionLines.png", visImage1)
        visImage2 = self.plotLines(self.image.copy(), self.lightEdges)
        cv2.imwrite("images/lightEdges.png", visImage2)
        visImage3 = self.plotLines(self.image.copy(), self.darkEdges)
        cv2.imwrite("images/darkEdges.png", visImage3)

if __name__ == "__main__": 

    image = cv2.imread("images/gears2.jpg")
    # lineMeasure1 = LineMeasure(image,
    #                            [15,10],
    #                            direction=[0,1],
    #                            length=70,
    #                            projectionWidth=20,
    #                            nProjections=10,
    #                            sigma=2.0,
    #                            threshold=1)
    # lineMeasure1.getEdges()
    # lineMeasure1.visualize()
    curveMeasure = CurveMeasure(
        image=image, 
        center=[280,500],
        radius=150,
        angleStart=0,
        angleExtent=np.pi/2,
        projectionWidth=30,
        nProjections=30,
        sigma=1.1,
        threshold=15
    )
    curveMeasure.getEdges()
    curveMeasure.visualize()