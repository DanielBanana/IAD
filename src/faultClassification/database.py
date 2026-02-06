#
# Created on Mon Jan 26 2026
#
# Copyright (c) 2026 TH Nuernberg - Daniel Pommer
#
"""
Manufacturing Fault Image Database using FiftyOne
Manages image sections of manufacturing faults with comprehensive metadata
"""

import fiftyone as fo
import fiftyone.zoo as foz
from datetime import datetime
from typing import Optional, List, Tuple
import os
from pathlib import Path


class ManufacturingFaultDatabase:
    """
    Manages a database of manufacturing fault images using FiftyOne.
    Each image contains metadata about the fault, timing, location, and validation.
    """
    
    def __init__(self, dataset_name: str = "manufacturing_faults"):
        """
        Initialize or load the manufacturing fault database.
        
        Args:
            dataset_name: Name of the FiftyOne dataset
        """
        self.dataset_name = dataset_name
        
        # Load existing dataset or create new one
        if dataset_name in fo.list_datasets():
            self.dataset = fo.load_dataset(dataset_name)
            print(f"Loaded existing dataset: {dataset_name}")
        else:
            self.dataset = fo.Dataset(dataset_name)
            self._setup_dataset_schema()
            print(f"Created new dataset: {dataset_name}")
    
    def _setup_dataset_schema(self):
        """Configure the dataset schema with custom fields for manufacturing faults."""
        # FiftyOne allows dynamic fields, but we'll document the expected schema
        print("Dataset schema configured for manufacturing fault tracking")
    
    def add_fault_image(
        self,
        image_path: str,
        fault_type: str,
        product_name: str,
        image_section_pixels: Tuple[int, int],  # (width, height)
        image_section_mm: Tuple[float, float],  # (width, height)
        section_location: Tuple[int, int, int, int],  # (x, y, width, height) on original
        original_image_path: Optional[str] = None,
        human_confirmed: bool = False,
        timestamp: Optional[datetime] = None,
        additional_metadata: Optional[dict] = None
    ) -> fo.Sample:
        """
        Add a fault image to the database with all required metadata.
        
        Args:
            image_path: Path to the fault image section
            fault_type: Type/category of the fault (e.g., "scratch", "dent", "discoloration")
            product_name: Name/ID of the product where fault occurred
            image_section_pixels: Size of image section in pixels (width, height)
            image_section_mm: Physical size of image section in mm (width, height)
            section_location: Location on original image (x, y, width, height)
            original_image_path: Path to the original full product image
            human_confirmed: Whether fault was confirmed by human (True) or just model (False)
            timestamp: When the image was taken (defaults to now)
            additional_metadata: Any additional metadata as dictionary
        
        Returns:
            FiftyOne Sample object
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create sample
        sample = fo.Sample(filepath=image_path)
        
        # Add fault-specific metadata
        sample["fault_type"] = fault_type
        sample["timestamp"] = timestamp
        sample["product_name"] = product_name
        
        # Image section information
        sample["section_width_px"] = image_section_pixels[0]
        sample["section_height_px"] = image_section_pixels[1]
        sample["section_width_mm"] = image_section_mm[0]
        sample["section_height_mm"] = image_section_mm[1]
        
        # Location on original image
        sample["section_x"] = section_location[0]
        sample["section_y"] = section_location[1]
        sample["section_box_width"] = section_location[2]
        sample["section_box_height"] = section_location[3]
        
        # Original image reference
        sample["original_image_path"] = original_image_path
        
        # Validation information
        sample["human_confirmed"] = human_confirmed
        sample["validation_source"] = "human" if human_confirmed else "model"
        
        # Additional metadata
        if additional_metadata:
            for key, value in additional_metadata.items():
                sample[key] = value
        
        # Add to dataset
        self.dataset.add_sample(sample)
        self.dataset.save()
        
        print(f"Added fault image: {fault_type} for {product_name}")
        return sample
    
    def query_faults(
        self,
        fault_type: Optional[str] = None,
        product_name: Optional[str] = None,
        human_confirmed: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> fo.DatasetView:
        """
        Query fault images based on various criteria.
        
        Args:
            fault_type: Filter by fault type
            product_name: Filter by product name
            human_confirmed: Filter by validation source
            start_date: Filter images after this date
            end_date: Filter images before this date
        
        Returns:
            FiftyOne DatasetView with filtered results
        """
        view = self.dataset.view()
        
        if fault_type:
            view = view.match(fo.ViewField("fault_type") == fault_type)
        
        if product_name:
            view = view.match(fo.ViewField("product_name") == product_name)
        
        if human_confirmed is not None:
            view = view.match(fo.ViewField("human_confirmed") == human_confirmed)
        
        if start_date:
            view = view.match(fo.ViewField("timestamp") >= start_date)
        
        if end_date:
            view = view.match(fo.ViewField("timestamp") <= end_date)
        
        return view
    
    def get_fault_statistics(self) -> dict:
        """
        Get statistics about faults in the database.
        
        Returns:
            Dictionary with fault statistics
        """
        stats = {
            "total_samples": len(self.dataset),
            "fault_types": self.dataset.distinct("fault_type"),
            "products": self.dataset.distinct("product_name"),
            "human_confirmed_count": len(self.dataset.match(
                fo.ViewField("human_confirmed") == True
            )),
            "model_only_count": len(self.dataset.match(
                fo.ViewField("human_confirmed") == False
            ))
        }
        
        # Count by fault type
        stats["fault_type_distribution"] = {}
        for fault_type in stats["fault_types"]:
            count = len(self.dataset.match(
                fo.ViewField("fault_type") == fault_type
            ))
            stats["fault_type_distribution"][fault_type] = count
        
        return stats
    
    def export_for_training(
        self,
        export_dir: str,
        fault_types: Optional[List[str]] = None,
        human_confirmed_only: bool = True,
        export_format: str = "image-classification-directory-tree"
    ):
        """
        Export dataset in a format suitable for training image classification models.
        
        Args:
            export_dir: Directory to export the dataset
            fault_types: Specific fault types to export (None = all)
            human_confirmed_only: Only export human-confirmed faults
            export_format: FiftyOne export format type
        """
        view = self.dataset.view()
        
        if fault_types:
            view = view.match(fo.ViewField("fault_type").is_in(fault_types))
        
        if human_confirmed_only:
            view = view.match(fo.ViewField("human_confirmed") == True)
        
        # Export dataset
        view.export(
            export_dir=export_dir,
            dataset_type=fo.types.ImageClassificationDirectoryTree,
            label_field="fault_type"
        )
        
        print(f"Exported {len(view)} samples to {export_dir}")
    
    def launch_app(self):
        """Launch the FiftyOne app to visualize the dataset."""
        session = fo.launch_app(self.dataset)
        return session
    
    def update_validation_status(
        self,
        sample_id: str,
        human_confirmed: bool
    ):
        """
        Update the validation status of a sample.
        
        Args:
            sample_id: ID of the sample to update
            human_confirmed: New validation status
        """
        sample = self.dataset[sample_id]
        sample["human_confirmed"] = human_confirmed
        sample["validation_source"] = "human" if human_confirmed else "model"
        sample.save()
        
        print(f"Updated validation status for sample {sample_id}")
    
    def delete_dataset(self):
        """Delete the entire dataset."""
        self.dataset.delete()
        print(f"Deleted dataset: {self.dataset_name}")


# Example usage
if __name__ == "__main__":
    # Initialize database
    db = ManufacturingFaultDatabase("manufacturing_faults_demo")
    
    # Example: Add fault images
    # Note: Replace with actual image paths
    
    db.add_fault_image(
        image_path="path/to/fault_image_001.jpg",
        fault_type="scratch",
        product_name="ProductA_Serial123",
        image_section_pixels=(640, 480),
        image_section_mm=(12.8, 9.6),
        section_location=(100, 200, 640, 480),  # x, y, width, height
        original_image_path="path/to/original_image_001.jpg",
        human_confirmed=True,
        timestamp=datetime(2024, 1, 15, 14, 30),
        additional_metadata={
            "production_line": "Line_A",
            "shift": "morning",
            "severity": "minor"
        }
    )
    
    db.add_fault_image(
        image_path="path/to/fault_image_002.jpg",
        fault_type="dent",
        product_name="ProductB_Serial456",
        image_section_pixels=(800, 600),
        image_section_mm=(16.0, 12.0),
        section_location=(50, 100, 800, 600),
        original_image_path="path/to/original_image_002.jpg",
        human_confirmed=False,
        additional_metadata={
            "production_line": "Line_B",
            "confidence_score": 0.87
        }
    )
    """
    
    # Query examples
    print("\n=== Statistics ===")
    stats = db.get_fault_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Query scratches only
    print("\n=== Querying scratches ===")
    scratches = db.query_faults(fault_type="scratch")
    print(f"Found {len(scratches)} scratch faults")
    
    # Query human-confirmed faults
    print("\n=== Querying human-confirmed faults ===")
    confirmed = db.query_faults(human_confirmed=True)
    print(f"Found {len(confirmed)} human-confirmed faults")
    
    # Export for training (uncomment to use)
    # db.export_for_training(
    #     export_dir="./training_data",
    #     human_confirmed_only=True
    # )
    
    # Launch FiftyOne app for visualization (uncomment to use)
    # session = db.launch_app()
    # session.wait()
    
    print("\n=== Database ready for use ===")