from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
import yaml
from enum import Enum
from cv2.typing import MatLike


class ContourID(Enum):
    INTERNAL = "internal"
    EXTERNAL = "extenal"

class Colour(Enum):
    RED=0
    BLUE=1
    GREEN=2
    YELLOW=3

class ContourType(Enum):
    POINT="Point"
    LINE = "Line"
    RECTANGLE = "Rectangle"
    OVAL = "Oval"
    POLYGON = "Polygon"
    FREEHAND_REGION = "Freehand Region"
    ANNULUS = "Annulus"
    BROKEN_LINE = "Broken Line"
    FREEHAND_LINE = "Freehand Line"
    ROTATED_RECTANGLE = "Rotated Rectangle"

@dataclass
class Contour():
    """Contours are each of the individual shapes that define an ROI
    """
    ID:ContourID                                        # referst to whether the contour is the external or internal edge of an ROI
    contourType:ContourType                             # is the shape type of the counter
    coordinates:Tuple[int,int]                          # Indicates the relative position of the contour

@dataclass
class ROI_Descriptor:
    globalRectangle:Tuple[int,int,int,int]              # minimum rectangle required to contain all of the contours in the ROI [x1,y1,x2,y2]
    contour:List[Contour]                               # Contours are each of the individual shapes that define an ROI

@dataclass
class ROI_Graph:
    """ROI_Graph is a cluster that contains the ROI profile with an x-origin at 0 and an increment of 1. 
    The cluster contains the following elements
    """
    pixelsLine:NDArray[np.float32]                      # returns the ROI profile calculated in an array in which elements represent the pixel values belonging
                                                        # to the specified vector
    x0:float = 0.0                                      # always returns 0 (?)
    dx:float = 1.0                                      # always returns 1 (?)

@dataclass
class ROI_PixelStatistics:
    min:float                                           # min pxiel value along found along ROI boundary
    max:float                                           # max pixel value along found along ROI boundary
    mean:float                                          # mean pixel value along found along ROI boundary
    std:float                                           # standard deviation of pixel values along found along ROI boundary
    count:int                                           # number of pixel found along ROI profile

class LevelType(Enum):
    PEAKS="Peaks"
    VALLEYS="Valleys"

class ThresholdLevel(Enum):
    ABSOLUTE="absolute"
    RELATIVE="relative"

class HysteresisLevel(Enum):
    ABSOLUTE="absolute"
    RELATIVE="relative"

class SimpleEdgeProcess(Enum):
    FIRST=0
    FIRSTANDLAST=1
    ALL=2

@dataclass
class ThresholdParameters:
    """Used to determine whether a change in pixel values is considered an edge
    """
    levelType: LevelType                                # choses between looking for peaks (positive bump) or valleys (negative bumps)
    thresholdLevel:ThresholdLevel                       # either absolute or relative. Absolute: based on pixel values; Relative: expressed as percentage of pixel value range found along the path defined by the pixel coordinates
    threshold:int                                       # See thresholdLevel
    hysteresisLevel:HysteresisLevel                     # can be either absolute or relative. Determines the difference in threshold level betwen a rising and a falling edge; enabling accurate detection in noisy image
    hysteresis:int                                      # See hysteresisLevel

class FindSimpleEdge:
    """Finds step edges along an array of pixel coordinates. This class  can return the first, both the first and last, or all the edges found.
    """
    def __init__(self, thresholdParameters:ThresholdParameters, image:MatLike, pixelCoordinates:NDArray[np.int32], process:SimpleEdgeProcess) -> None:
        thresholdParameters = thresholdParameters
        self.image = image
        self.pixelCoordinates = pixelCoordinates
        self.process = process
        self.edgeCoordinates:List[NDArray[np.float32]]|None = None
        self.nEdges:int|None = None

    def determineEdges(self) -> Tuple[List[NDArray[np.float32]], int]:
        return [np.asarray([0,0,1,1]), np.asarray([2,2,3,3])], 2

class ROI_Profile():
    """Image Analysis (compare: IMAQ ROIProfile VI)
    """
    def __init__(self, image:NDArray[np.float32], descriptor: ROI_Descriptor) -> None:
        self.image = image
        self.descriptor = descriptor

class EdgeFindDirection(Enum):
    LtoR=0
    RtoL=1
    TtoB=2
    BtoT=3

class EdgePolarity(Enum):
    ALL=0
    RISING=1
    FALLING=2

class InterpolationOrder(Enum):
    ZERO=0
    ONE=1
    TWO=2

class DataProcessingMethod(Enum):
    AVERAGE=0
    MEDIAN=1

@dataclass
class EdgeOptions():
    edgePolarity:EdgePolarity
    kernelSize:int
    width:int
    minimumEdgeStrength:int
    interpolationOrder:InterpolationOrder
    dataProcessingMethod:DataProcessingMethod

@dataclass
class StraightEdge():
    point1:Tuple[float,float]           # start point of the edge in pixel units
    point2:Tuple[float,float]           # end point of the edge in pixels
    point1RL:Tuple[float,float]         # start point of the edge in real live calibrated units
    point2RL:Tuple[float,float]         # end point of the edge in real live calibrated units
    angle:float                         # angle the detected edge makes with the axis perpendicular to the search direction
    angleRL:float                       # angle the detected edge makes with the axis perpendicular to the search direction in calibrated units
    score:float                         # score of the deteced straight edge
    straightness:float
    averageSNR:float
    calibrationValid:bool


@dataclass
class CoordinateSystemSettings():
    roi1: ROI_Profile
    roi2: ROI_Profile
    overlayGroup:str
    searchDirection:EdgeFindDirection
    edgeOptions:EdgeOptions
    showSearchArea:bool
    searchAreaColour:Colour
    showSearchLine:bool
    searchLineColour:Colour
    showEdgeFound:bool
    edgeLocationsColour:Colour
    showResult:bool
    resultColour:Colour

class FindCoordinateSystem():
    """Build_CS_v15
    """
    def __init__(self, coordinateSystemSettings:CoordinateSystemSettings) -> None:
        self.coordinateSystemSettings = coordinateSystemSettings

    def findEdges(self):
        """
        Find two edges of the product that represent the leftmost and bottommost edge.
        They are assumed to be perpendicular. Based on these two edges a coordinate system is created and the image is realigned.

        Returns:
            
        """

class AxisType(Enum):
    """Refer to Axis Type in LabView
    ↑ Y          → → → → X   
    ↑           ↓
    ↑           ↓
    ↑           ↓
     → → → → X  ↓ Y
     DIRECT       INDIRECT

    """
    DIRECT=0
    INDIRECT=1

@dataclass
class CoordinateSystem():
    origin:Tuple[float,float]           # origin of the coordinate system
    angle:float                         # angle between the coordinate system and a potential image coordinate system
    axisReference:AxisType              # orientation of the axis

class BuildCoordinateSystem():
    """IMAQ Build CoordSys (Points) VI
    """
    def __init__(self, point1:Tuple[float, float], point2:Tuple[float, float], point3:Tuple[float, float]|None):
        """If two points are specified, these points are asssumed to lie on the x axis with the first one being the origin. 
        The y axis is perpendicular to the x axis
        
        If three points are specified, the first two are assumed to be along the x-axis and the thrid point is assumed to be on the y-axis.
        The y-axis is perpendicular to the specified x-axis and going through the third point. 
        The origin is given by the intersection point of the two axis.

        Arguments:
            point1 -- _description_
            point2 -- _description_
            point3 -- _description_
        """
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3

    def build(self) -> CoordinateSystem:
        # TODO: Implementation
        cs = CoordinateSystem((0,0), 0, AxisType.DIRECT)
        return cs
    
class StraightEdgeType(Enum):
    FIRST_EDGE_RAKE=0                       # Fits a straight edge on the first points detected using a rake (Default)
    BEST_EDGE_RAKE=1                        # Fits a straight edge on the best points detected using a rake. (Used in AOI)
    HOUGH_EDGE_RAKE=2                       # Find the strongest straight edge using all points detect on a rake.
    FIRST_EDGE_PROJECTION=3                 # Uses the location of the first projected edge as the straight edge.
    BEST_EDGE_PROJECTION=4                  # Finds the strongest projected edge location to determine the straight edge.

@dataclass
class LineFitOptions():
    nLines:int=1                            # number of lines to find; default is 1
    type:StraightEdgeType=StraightEdgeType.BEST_EDGE_RAKE
    minScore:int=10                         # minimum number of points as a percentage of the number of search lines that need to be included in the detected straight edge.
    maxScore:int=1000                       # maximum score of a detected straight edge
    orientation:int=0                       # angle at which the straight edge is expected to be found
    angleRange:float=45                     # positive and negative range around the orientation within which the straight edge is expected to be found; in degrees
    angleTolerance:float=1                  # expected angulare accuracy of the straight edge
    stepSize:float=7                        # gap in pixels between the search lines used with the rake-based methods
    minSNR:float=0                          # minimum signal to noise ration (SNR) of the edge points used to fit the straight edge; in db
    minPoints:float=25                      # minimum number of points as a percentage of the number of search lines that need to be included in the detected straight edge.; %
    houghIterations:int=5                   # number of iterations used in hough-based methods


class FindEdges():
    """IMAQ Find Edge VI
    Machine Vision/Locate Edges/Find Edge
    """
    def __init__(self, options:EdgeOptions, image:MatLike, roiDescriptor:ROI_Descriptor, CSYS:CoordinateSystem, lineFitOptions:LineFitOptions) -> None:
        pass