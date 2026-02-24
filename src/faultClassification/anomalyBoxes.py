import cv2
from cv2.typing import MatLike

import numpy as np
from numpy.typing import NDArray

from typing import List, Tuple, Optional, Dict
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path

class AnomalyBoxes:
    """Generates bounding boxes for anomalies based on anomaly heatmaps (anomaly score maps)
    """
    def __init__(self, minArea: int = 100):
        """
        Initialize the anomaly detector.
        
        Args:
            threshold: Anomaly values above this threshold are considered anomalous
            min_area: Minimum area (in pixels) for a bounding box to be kept
        """
        self.minArea = minArea

    def detectAnomalies(self, binaryMask:NDArray[np.uint8]) -> List[Tuple[int, int, int, int]]:
        """
        Detect anomalies in a heatmap and return bounding boxes.
        
        Args:
            heatmap: 2D numpy array containing anomaly scores
            
        Returns:
            List of bounding boxes as (x, y, width, height) tuples
        """

        # Create binary mask of anomalous regions
        # binaryMask = (heatmap > self.threshold).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up the mask
        kernel:MatLike = cv2.Mat(np.ones((3, 3), np.uint8))
        binaryMaskCV:MatLike = cv2.Mat(binaryMask)
        binaryMask = cv2.morphologyEx(binaryMaskCV, cv2.MORPH_CLOSE, kernel, iterations=2)
        binaryMask = cv2.morphologyEx(binaryMaskCV, cv2.MORPH_OPEN, kernel, iterations=1)
        
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
                        binaryMask: NDArray[np.uint8],
                        label:str) -> Tuple[MatLike, List[Tuple[int, int, int, int]]]:
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
        bboxes = self.detectAnomalies(binaryMask)
        
        # Draw bounding boxes on image
        annotatedImage:MatLike = cvimage.copy()
        for (x, y, w, h) in bboxes:
            cv2.rectangle(annotatedImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add label with anomaly score
            # max_score = binaryMask[y:y+h, x:x+w].max()
            # label = f"{max_score:.2f}"
            cv2.putText(annotatedImage, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return annotatedImage, bboxes
    
    def visualize_results(self, 
                          image: NDArray[np.float32], 
                          binaryMask: NDArray[np.uint8], 
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
        im = axes[1].imshow(binaryMask, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title(f'Anomaly binaryMask')
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

    # ------------------------------------------------------------------ #
    #  Folder / IO helpers                                                 #
    # ------------------------------------------------------------------ #

    MASK_EXTENSIONS: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy')

    def loadMasksFromFolder(self, folderPath: Path) -> Dict[str, NDArray[np.uint8]]:
        """
        Load all binary mask files from a folder and its subfolders.
        Only files with "_predMask" at the end of the filename (before the suffix) are considered.

        Supported formats: PNG, JPG, BMP, TIFF (loaded as grayscale) and NumPy .npy files.

        Args:
            folderPath: Path to the directory containing mask files.

        Returns:
            Dictionary mapping filename stem (without extension) to the loaded
            mask as a uint8 NumPy array.

        Raises:
            ValueError: If folderPath does not exist or is not a directory.
        """
        folderPath = Path(folderPath)
        if not folderPath.is_dir():
            raise ValueError(f"Folder not found: {folderPath}")

        masks: Dict[str, NDArray[np.uint8]] = {}

        for filePath in sorted(folderPath.rglob('*')):
            if filePath.is_dir():
                continue

            if filePath.suffix.lower() not in self.MASK_EXTENSIONS:
                continue

            # Check if "_predMask" is at the end of the stem (before the suffix)
            if not filePath.stem.endswith("_predMask"):
                continue

            try:
                if filePath.suffix.lower() == '.npy':
                    mask: NDArray[np.uint8] = np.load(filePath).astype(np.uint8)
                else:
                    maskCV: MatLike | None = cv2.imread(str(filePath), cv2.IMREAD_GRAYSCALE)
                    if maskCV is None:
                        print(f"Warning: could not load mask '{filePath}', skipping.")
                        continue
                    mask = maskCV.astype(np.uint8)


                # Get the immediate parent folder name
                parent_folder = filePath.parent.name
                # Construct the key as "parent_folder/filename_stem"
                key = f"{parent_folder}/{filePath.stem}"

                masks[key] = mask
            except Exception as e:
                print(f"Warning: error loading mask '{filePath}': {e}, skipping.")

        return masks

    def saveBBoxes(self,
                   bboxes: Dict[str, List[Tuple[int, int, int, int]]],
                   savePath: Path,
                   sourcePaths: Optional[Dict[str, Path]] = None) -> None:
        """
        Save bounding boxes to a JSON file.

        The JSON structure is::

            {
                "<stem>": {
                    "source_path": "/absolute/path/to/stem_predMask.png",
                    "format": "xywh",
                    "boxes": [[x, y, w, h], ...]
                },
                ...
            }

        Args:
            bboxes:       Dictionary mapping a name/stem to a list of
                          (x, y, width, height) bounding-box tuples.
            savePath:     Destination file path (should end in ``.json``).
            sourcePaths:  Optional dictionary mapping the same stems to the
                          absolute Path of the original binary mask file.
                          When provided, each entry gains a ``"source_path"`` field.
        """
        savePath = Path(savePath)
        savePath.parent.mkdir(parents=True, exist_ok=True)

        serialisable = {}
        for stem, boxList in bboxes.items():
            entry: dict = {
                "format": "xywh",
                "boxes": [list(box) for box in boxList],
            }
            if sourcePaths and stem in sourcePaths:
                entry["source_path"] = str(Path(sourcePaths[stem]).resolve())
            serialisable[stem] = entry

        with savePath.open('w') as f:
            json.dump(serialisable, f, indent=2)

        print(f"Bounding boxes saved to {savePath}")

    def saveIndividualBBoxes(self,
                   crops:List[NDArray[np.uint8]],
                   maskPath:Path) -> None:
        
        # savePath = Path(savePath)
        # savePath.parent.mkdir(parents=True, exist_ok=True)
        folder:Path = maskPath.parent

        for i, crop in enumerate(crops):
            cropName:Path = folder / (str(maskPath.stem) + f"_{i}.png")
            cv2.imwrite(str(cropName), crop)

        # serialisable = {}
        # for stem, boxList in bboxes.items():
        #     entry: dict = {
        #         "format": "xywh",
        #         "boxes": [list(box) for box in boxList],
        #     }
        #     if sourcePaths and stem in sourcePaths:
        #         entry["source_path"] = str(Path(sourcePaths[stem]).resolve())
        #     serialisable[stem] = entry

        # with savePath.open('w') as f:
        #     json.dump(serialisable, f, indent=2)

        print(f"Bounding boxes saved for {maskPath}")

    def loadBBoxes(self, loadPath: Path) -> Dict[str, List[Tuple[int, int, int, int]]]:
        """
        Load bounding boxes previously saved by :meth:`saveBBoxes`.

        Args:
            loadPath: Path to the JSON file written by :meth:`saveBBoxes`.

        Returns:
            Dictionary mapping name/stem to a list of
            (x, y, width, height) bounding-box tuples.

        Raises:
            ValueError: If the file cannot be found or is not valid JSON.
        """
        loadPath = Path(loadPath)
        if not loadPath.is_file():
            raise ValueError(f"BBox file not found: {loadPath}")

        with loadPath.open('r') as f:
            data: dict = json.load(f)

        bboxes: Dict[str, List[Tuple[int, int, int, int]]] = {}
        for stem, entry in data.items():
            bboxes[stem] = [tuple(box) for box in entry["boxes"]]  # type: ignore[misc]

        return bboxes

    def cropMaskToBBoxes(self, maskPath: Path, imagePath:Path) -> List[NDArray[np.uint8]]:
        """
        Load a binary mask from disk, detect anomaly bounding boxes, and return
        a cropped sub-image of the mask for each bounding box.

        Each returned crop contains exactly the region of the mask that falls
        inside one bounding box, preserving the original pixel values (0 / 255).

        Args:
            maskPath: Path to the binary mask file.
                      Supported formats: PNG, JPG, BMP, TIFF, or ``.npy``.

        Returns:
            List of uint8 arrays, one per detected bounding box, each of shape
            (h, w) and containing the mask content within that box.
            Returns an empty list if no anomalies are found.

        Raises:
            ValueError: If the mask file cannot be loaded.
        """
        maskPath = Path(maskPath)


        # --- load mask ---
        if maskPath.suffix.lower() == '.npy':
            mask: NDArray[np.uint8] = np.load(maskPath).astype(np.uint8)
        else:
            maskCV: MatLike | None = cv2.imread(str(maskPath), cv2.IMREAD_GRAYSCALE)
            if maskCV is None:
                raise ValueError(f"Could not load mask from {maskPath}")
            mask = maskCV.astype(np.uint8)

        # --- load image ---
        imageCV: MatLike | None = cv2.imread(str(imagePath), cv2.IMREAD_COLOR_RGB)
        if imageCV is None:
            raise ValueError(f"Could not load mask from {imagePath}")
        image = imageCV.astype(np.uint8)

        # --- detect bounding boxes ---
        bboxes = self.detectAnomalies(mask)

        # --- crop ---
        crops: List[NDArray[np.uint8]] = []
        for (x, y, w, h) in bboxes:
            crop: NDArray[np.uint8] = image[y:y + h, x:x + w]
            crops.append(crop)

        return crops

    def drawBBoxesOnMask(self,
                         binaryMask: NDArray[np.uint8],
                         bboxes: List[Tuple[int, int, int, int]],
                         savePath: Optional[Path] = None,
                         label: str = "") -> MatLike:
        """
        Overlay bounding boxes onto a binary mask and optionally save the result.

        The mask is converted to a 3-channel BGR image so the green rectangles
        (and optional text labels) stand out clearly against the white/black mask.

        Args:
            binaryMask: 2-D uint8 mask array (pixel values 0 or 255).
            bboxes:     List of (x, y, width, height) tuples to draw.
            savePath:   If provided, the annotated image is written to this path.
            label:      Optional text label drawn above each bounding box.

        Returns:
            Annotated BGR image as a MatLike (numpy array).
        """
        # Convert grayscale mask to BGR so we can draw coloured boxes
        maskBGR: MatLike = cv2.cvtColor(cv2.Mat(binaryMask), cv2.COLOR_GRAY2BGR)
        annotated: MatLike = maskBGR.copy()

        for (x, y, w, h) in bboxes:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if label:
                # Position the label just above the box; clamp so it stays in frame
                textY = max(y - 5, 12)
                cv2.putText(annotated, label, (x, textY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if savePath is not None:
            savePath = Path(savePath)
            savePath.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(savePath), annotated)
            print(f"Annotated mask saved to {savePath}")

        return annotated

def loadImageAndHeatmap(imagePath: Path, binaryMaskPath: Path) -> Tuple[MatLike, MatLike]:
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
    if binaryMaskPath.suffix == '.npy':
        binaryMask:NDArray[np.uint8] = np.load(binaryMaskPath)
    else:
        binaryMaskCV:MatLike|None = cv2.imread(str(binaryMaskPath), cv2.IMREAD_GRAYSCALE)
        if binaryMaskCV is None:
            raise ValueError(f"Could not load heatmap from {binaryMaskPath}")
        binaryMask = binaryMaskCV.astype(np.uint8)
    
    # Ensure heatmap matches image dimensions
    # if binaryMask is not None:
    if binaryMask.shape[:2] != image.shape[:2]:
        binaryMask = cv2.resize(binaryMask, (image.shape[1], image.shape[0])).astype(np.uint8)
    
    return image, binaryMask

if __name__ == "__main__":
    detector = AnomalyBoxes(minArea=100)

    maskParent = "results/MVTecAD/cable/padim/tiled/images/cable/test/"
    imageParent = Path("datasets/MVTecAD/cable/test/")


    masks = detector.loadMasksFromFolder(Path(maskParent))
    sourcePaths = {stem: Path(maskParent) / f"{stem}.png" for stem in masks}

    results = {stem: detector.detectAnomalies(mask) for stem, mask in masks.items()}

    # Save JSON with embedded source paths
    detector.saveBBoxes(results, Path(f"{maskParent}bboxes.json"), sourcePaths=sourcePaths)

    # Save one overlay image per mask
    for stem, mask in masks.items():
        detector.drawBBoxesOnMask(mask, results[stem],
                                savePath=Path(f"{maskParent}{stem}_overlay.png"),
                                label="anomaly")
        crops = detector.cropMaskToBBoxes(sourcePaths[stem], imagePath=imageParent/(stem))
        detector.saveIndividualBBoxes(crops, sourcePaths[stem])

        # for i, crop in enumerate(crops):
        #     cv2.imwrite(f"crops/anomaly_{i}.png", crop)