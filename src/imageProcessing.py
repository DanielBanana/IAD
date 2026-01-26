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