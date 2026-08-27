from pathlib import Path
import sys
import os
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2
from src.manager import AnomalyDetectionManager
from src.faultClassification.anomalyBoxes import AnomalyBoxes

# Add parent directory to path for absolute import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestIAD(unittest.TestCase):
    """Test suite for the anomaly detection class

    Arguments:
        unittest -- _description_
    """
    def setUp(self) -> None:
        # os.environ["FIFTYONE_DATABASE_URI"] = "mongodb://localhost"
        # wandb.login()
        self.adm = AnomalyDetectionManager()
        self.modelName = "padim.yaml"


    def test_generateModel(self):
        self.adm.generateModel(self.modelName)

    def test_loadFromDisk(self):
        datasetPath1 = Path(os.path.join("datasets", "traintest"))
        self.adm.loadDatasetFromDisk(datasetPath1, "traintest", overwrite=False, merge=False)

    def test_loadFromDB(self):
        # iad.launchSession()
        datasetPath2 = Path(os.path.join("datasets", "traintest"))
        self.adm.loadDatasetFromDisk(datasetPath2, "traintest", overwrite=False, merge=False)
        self.adm.loadDatasetFromDatabase("traintest")

    def test_overwriteDate(self):
        datasetPath2 = Path(os.path.join("datasets", "traintest"))
        self.adm.loadDatasetFromDisk(datasetPath2, "traintest", overwrite=False, merge=False)
        self.adm.loadDatasetFromDisk(datasetPath2, "traintest", overwrite=True, merge=False)

    def test_mergeData(self):
        datasetPath2 = Path(os.path.join("datasets", "traintest"))
        self.adm.loadDatasetFromDisk(datasetPath2, "traintest", overwrite=False, merge=False)
        datasetPath2 = Path(os.path.join("datasets", "MVTecADShort"))
        self.adm.loadDatasetFromDisk(datasetPath2, "MVTecADShort", overwrite=False, merge=True)

    def test_copyFiles(self):
        self.adm.adjustOutputPath()
        self.adm.copyFilesToOutputPath()
        assert (self.iad.outputPath/self.modelName).exists()

    def test_Tiling(self):
        self.adm.setupTiling(Path("configs/TiledEnsemble.yaml"))

    def test_loadCheckpoint(self):
        self.adm.loadCheckpoint(Path("test/MVTecADShort/bottle/padim/checkpoints/best.ckpt"))


        #iad.launchSession()
        # iad.loadCheckpoint(Path("results/MVTecADShort/bottle/padim/checkpoints/best.ckpt"))
        self.adm.setupTiling(Path("configs/TiledEnsemble.yaml"))
        self.adm.train(Path("configs/padim_Training.yaml"), tiling=True)


class TestAnomalyBoxes(unittest.TestCase):
    """Test suite for AnomalyBoxes class"""

    def setUp(self):
        """Set up test fixtures"""
        self.anomaly_detector = AnomalyBoxes(threshold=0.5, minArea=100)
        self.test_heatmap = np.random.rand(100, 100).astype(np.float32)
        self.test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    def test_anomaly_boxes_initialization(self):
        """Test AnomalyBoxes initialization"""
        detector = AnomalyBoxes(threshold=0.6, minArea=50)
        self.assertEqual(detector.threshold, 0.6)
        self.assertEqual(detector.minArea, 50)

    def test_detect_anomalies_normalized_heatmap(self):
        """Test detectAnoamlies with normalized heatmap"""
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[20:40, 20:40] = 0.8
        heatmap[60:80, 60:80] = 0.9
        
        bboxes = self.anomaly_detector.detectAnoamlies(heatmap)
        self.assertIsInstance(bboxes, list)
        self.assertTrue(all(isinstance(bbox, tuple) and len(bbox) == 4 for bbox in bboxes))

    def test_detect_anomalies_non_normalized_heatmap(self):
        """Test detectAnoamlies with non-normalized heatmap"""
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[20:40, 20:40] = 150
        heatmap[60:80, 60:80] = 200
        
        bboxes = self.anomaly_detector.detectAnoamlies(heatmap)
        self.assertIsInstance(bboxes, list)

    def test_detect_anomalies_min_area_filtering(self):
        """Test that bboxes smaller than minArea are filtered"""
        detector = AnomalyBoxes(threshold=0.5, minArea=1000)
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[20:25, 20:25] = 0.8
        
        bboxes = detector.detectAnoamlies(heatmap)
        self.assertEqual(len(bboxes), 0)

    def test_process_image_pair_grayscale(self):
        """Test processImagePair with grayscale image"""
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        heatmap = np.random.rand(100, 100).astype(np.float32)
        
        annotated_image, bboxes = self.anomaly_detector.processImagePair(gray_image, heatmap)
        self.assertEqual(annotated_image.shape, (100, 100, 3))
        self.assertIsInstance(bboxes, list)

    def test_process_image_pair_color(self):
        """Test processImagePair with color image"""
        color_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        heatmap = np.random.rand(100, 100).astype(np.float32)
        
        annotated_image, bboxes = self.anomaly_detector.processImagePair(color_image, heatmap)
        self.assertEqual(annotated_image.shape, (100, 100, 3))
        self.assertIsInstance(bboxes, list)

    def test_process_image_pair_rgba(self):
        """Test processImagePair with RGBA image"""
        rgba_image = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        heatmap = np.random.rand(100, 100).astype(np.float32)
        
        annotated_image, bboxes = self.anomaly_detector.processImagePair(rgba_image, heatmap)
        self.assertEqual(annotated_image.shape, (100, 100, 3))
        self.assertIsInstance(bboxes, list)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_visualize_results_without_save(self, mock_show, mock_savefig):
        """Test visualize_results without saving"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        heatmap = np.random.rand(100, 100).astype(np.float32)
        bboxes = [(10, 10, 30, 30), (50, 50, 20, 20)]
        
        self.anomaly_detector.visualize_results(image, heatmap, bboxes)
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    def test_visualize_results_with_save(self, mock_savefig):
        """Test visualize_results with saving"""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        heatmap = np.random.rand(100, 100).astype(np.float32)
        bboxes = [(10, 10, 30, 30)]
        
        self.anomaly_detector.visualize_results(image, heatmap, bboxes, savePath="/tmp/test.png")
        mock_savefig.assert_called_once()

    def test_visualize_results_grayscale(self):
        """Test visualize_results with grayscale image"""
        with patch('matplotlib.pyplot.show'):
            image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
            heatmap = np.random.rand(100, 100).astype(np.float32)
            bboxes = []
            
            self.anomaly_detector.visualize_results(image, heatmap, bboxes)


class TestLoadImageAndHeatmap(unittest.TestCase):
    """Test suite for loadImageAndHeatmap function"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)

    def test_load_image_and_heatmap_npy(self):
        """Test loading image and .npy heatmap"""
        image_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        heatmap_array = np.random.rand(100, 100).astype(np.float32)
        
        image_path = Path(self.temp_dir) / "test_image.png"
        heatmap_path = Path(self.temp_dir) / "test_heatmap.npy"
        
        cv2.imwrite(str(image_path), image_array)
        np.save(str(heatmap_path), heatmap_array)
        
        image, heatmap = loadImageAndHeatmap(image_path, heatmap_path)
        
        self.assertIsNotNone(image)
        self.assertIsNotNone(heatmap)
        self.assertEqual(heatmap.shape, (100, 100))

    def test_load_image_and_heatmap_image_file(self):
        """Test loading image and image heatmap"""
        image_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        heatmap_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        image_path = Path(self.temp_dir) / "test_image.png"
        heatmap_path = Path(self.temp_dir) / "test_heatmap.png"
        
        cv2.imwrite(str(image_path), image_array)
        cv2.imwrite(str(heatmap_path), heatmap_array)
        
        image, heatmap = loadImageAndHeatmap(image_path, heatmap_path)
        
        self.assertIsNotNone(image)
        self.assertIsNotNone(heatmap)

    def test_load_image_and_heatmap_resize_heatmap(self):
        """Test that heatmap is resized to match image dimensions"""
        image_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        heatmap_array = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        image_path = Path(self.temp_dir) / "test_image.png"
        heatmap_path = Path(self.temp_dir) / "test_heatmap.png"
        
        cv2.imwrite(str(image_path), image_array)
        cv2.imwrite(str(heatmap_path), heatmap_array)
        
        image, heatmap = loadImageAndHeatmap(image_path, heatmap_path)
        
        self.assertEqual(heatmap.shape[:2], image.shape[:2])

    def test_load_image_not_found(self):
        """Test error when image file not found"""
        image_path = Path(self.temp_dir) / "nonexistent_image.png"
        heatmap_path = Path(self.temp_dir) / "test_heatmap.npy"
        
        np.save(str(heatmap_path), np.random.rand(100, 100).astype(np.float32))
        
        with self.assertRaises(ValueError):
            loadImageAndHeatmap(image_path, heatmap_path)

    def test_load_heatmap_not_found(self):
        """Test error when heatmap file not found"""
        image_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        image_path = Path(self.temp_dir) / "test_image.png"
        heatmap_path = Path(self.temp_dir) / "nonexistent_heatmap.png"
        
        cv2.imwrite(str(image_path), image_array)
        
        with self.assertRaises(ValueError):
            loadImageAndHeatmap(image_path, heatmap_path)


if __name__ == "__main__":
    unittest.main()