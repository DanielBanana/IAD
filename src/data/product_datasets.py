from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
import yaml
from enum import Enum
from cv2.typing import MatLike

@dataclass
class Image():
    """Class that holds an image to process
    """
    image: NDArray[np.float32]|None = None
    imageSegments: List[NDArray[np.float32]]|None = None

    def addImage(self, image:NDArray[np.float32]):
        self.image = image

    def addImageSegments(self, segments:NDArray[np.float32]):
        segments = np.asarray(segments)
        self.Image = np.concat(segments, axis=0)

@dataclass
class Product():
    """Product description
    """
    type:str
    dimensions: NDArray[np.float32]
    id:int
    image: Image

@dataclass
class Error():
    """Error Description"""
    type:str
    products:List[Product]
    size:Tuple[int, int]                                # in pixels
    location:Tuple[int,int]                             # in pixels
    angle:int                                           # rotational angle of the bounding box of the error; in degree; 0 is parallel to x axis to the right, 90 is parallel to y axis upwards








