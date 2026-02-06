import cv2
from cv2.typing import MatLike

import numpy as np
from numpy.typing import NDArray

from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path

class AnomalyBoxes:
    """Generates bounding boxes for anomalies based on anomaly heatmaps (anomaly score maps)
    """
    def __init__(self, threshold: float = 0.5, minArea: int = 100):
        """
        Initialize the anomaly detector.
        
        Args:
            threshold: Anomaly values above this threshold are considered anomalous
            min_area: Minimum area (in pixels) for a bounding box to be kept
        """
        self.threshold = threshold
        self.minArea = minArea

    def detectAnoamlies(self, heatmap:NDArray[np.float32]) -> List[Tuple[int, int, int, int]]:
        """
        Detect anomalies in a heatmap and return bounding boxes.
        
        Args:
            heatmap: 2D numpy array containing anomaly scores
            
        Returns:
            List of bounding boxes as (x, y, width, height) tuples
        """
        # Normalize heatmap to 0-1 range if not already
        if heatmap.max() > 1.0:
            heatmap = heatmap / heatmap.max()
        
        # Create binary mask of anomalous regions
        binaryMask = (heatmap > self.threshold).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        binaryMask = cv2.morphologyEx(binaryMask, cv2.MORPH_CLOSE, kernel, iterations=2)
        binaryMask = cv2.morphologyEx(binaryMask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(binaryMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract bounding boxes
        bboxes:List[Tuple[int,int,int,int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by minimum area
            if area >= self.minArea:
                bboxes.append((x, y, w, h))
        
        return bboxes

    def processImagePair(self, 
                        image: NDArray[np.float32], 
                        heatmap: NDArray[np.float32]) -> Tuple[MatLike, List[Tuple[int, int, int, int]]]:
        """
        Process an image-heatmap pair and draw bounding boxes.
        
        Args:
            image: Original image (HxW or HxWx3)
            heatmap: Anomaly heatmap (HxW)
            
        Returns:
            Tuple of (annotated_image, bboxes)
        """
        # Ensure image is in color
        if len(image.shape) == 2:
            cvimage:MatLike = cv2.Mat(image)
            cvimage = cv2.cvtColor(cvimage, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            cvimage:MatLike = cv2.Mat(image)
            cvimage = cv2.cvtColor(cvimage, cv2.COLOR_BGRA2BGR)
        else:
            print(f"Wrong image shape. Image should either have on channel or 3 channels. Image has {len(image.shape)-1} channels.")
            exit(1)
        
        # Detect anomalies
        bboxes = self.detectAnoamlies(heatmap)
        
        # Draw bounding boxes on image
        annotatedImage:MatLike = cvimage.copy()
        for (x, y, w, h) in bboxes:
            cv2.rectangle(annotatedImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add label with anomaly score
            max_score = heatmap[y:y+h, x:x+w].max()
            label = f"{max_score:.2f}"
            cv2.putText(annotatedImage, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotatedImage, bboxes
    
    def visualize_results(self, 
                          image: NDArray[np.float32], 
                          heatmap: NDArray[np.float32], 
                          bboxes: List[Tuple[int, int, int, int]],
                          savePath: Optional[str] = None):
        """
        Create a visualization showing original image, heatmap, and detected anomalies.
        
        Args:
            image: Original image
            heatmap: Anomaly heatmap
            bboxes: List of bounding boxes
            save_path: Optional path to save the visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if len(image.shape) == 3 and image.shape[2] == 3:
            cvImage = cv2.Mat(image)
            axes[0].imshow(cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB))
        else:
            cvImage = cv2.Mat(image)
            axes[0].imshow(cvImage, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title(f'Anomaly Heatmap (threshold={self.threshold})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Image with bounding boxes
        if len(image.shape) == 3 and image.shape[2] == 3:
            axes[2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            axes[2].imshow(image, cmap='gray')
        
        for (x, y, w, h) in bboxes:
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                    edgecolor='lime', facecolor='none')
            axes[2].add_patch(rect)
        
        axes[2].set_title(f'Detected Anomalies ({len(bboxes)} regions)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if savePath:
            plt.savefig(savePath, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {savePath}")
        else:
            plt.show()

def loadImageAndHeatmap(imagePath: Path, heatmapPath: Path) -> Tuple[MatLike, MatLike]:
    """
    Load an image and its corresponding heatmap.
    
    Args:
        image_path: Path to the image file
        heatmap_path: Path to the heatmap file (can be image or .npy)
        
    Returns:
        Tuple of (image, heatmap)
    """
    # Load image
    image:MatLike|None = cv2.imread(str(imagePath))
    if image is None:
        raise ValueError(f"Could not load image from {imagePath}")
    
    # Load heatmap
    if heatmapPath.suffix == '.npy':
        heatmap:NDArray[np.float32] = np.load(heatmapPath)
        cvheatmap:MatLike = cv2.Mat(heatmap)
    else:
        heatmap:MatLike|None = cv2.imread(str(heatmapPath), cv2.IMREAD_GRAYSCALE)
        if heatmap is None:
            raise ValueError(f"Could not load heatmap from {heatmapPath}")
        heatmap = heatmap.astype(np.float32) / 255.0
    
    # Ensure heatmap matches image dimensions
    if heatmap is not None:
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    return image, heatmap
