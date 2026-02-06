"""
PyTorch-based Fault Classifier for Manufacturing Defects
Trains and evaluates image classification models on fault data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict
import fiftyone as fo
from database import ManufacturingFaultDatabase


class FaultDataset(Dataset):
    """PyTorch Dataset for manufacturing fault images."""
    
    def __init__(self, samples: List[fo.Sample], transform=None):
        """
        Args:
            samples: List of FiftyOne samples
            transform: Optional torchvision transforms
        """
        self.samples = samples
        self.transform = transform
        
        # Create label mapping
        self.fault_types = sorted(list(set(s["fault_type"] for s in samples)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.fault_types)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample.filepath).convert('RGB')
        
        # Get label
        label = self.label_to_idx[sample["fault_type"]]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced datasets."""
        label_counts = {}
        for sample in self.samples:
            label = sample["fault_type"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        total = len(self.samples)
        weights = torch.zeros(len(self.fault_types))
        for label, count in label_counts.items():
            idx = self.label_to_idx[label]
            weights[idx] = total / (len(self.fault_types) * count)
        
        return weights


class FaultClassifier:
    """PyTorch model trainer and evaluator for manufacturing fault classification."""
    
    def __init__(
        self,
        db: ManufacturingFaultDatabase,
        model_name: str = "resnet18",
        pretrained: bool = True,
        device: str = None
    ):
        """
        Initialize the fault classifier.
        
        Args:
            db: ManufacturingFaultDatabase instance
            model_name: Model architecture (resnet18, resnet50, efficientnet_b0, etc.)
            pretrained: Use pretrained weights
            device: Device to use (cuda/cpu), auto-detected if None
        """
        self.db = db
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.fault_types = None
        
    def prepare_data(
        self,
        human_confirmed_only: bool = True,
        test_size: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        Prepare data loaders for training.
        
        Args:
            human_confirmed_only: Use only human-confirmed samples
            test_size: Fraction of data for validation
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
        """
        # Get samples
        view = self.db.dataset.view()
        if human_confirmed_only:
            view = view.match(fo.ViewField("human_confirmed") == True)
        
        samples = list(view)
        
        if len(samples) == 0:
            raise ValueError("No samples found in dataset")
        
        print(f"Total samples: {len(samples)}")
        
        # Split into train/val
        train_samples, val_samples = train_test_split(
            samples, test_size=test_size, random_state=42,
            stratify=[s["fault_type"] for s in samples]
        )
        
        print(f"Training samples: {len(train_samples)}")
        print(f"Validation samples: {len(val_samples)}")
        
        # Create datasets
        train_dataset = FaultDataset(train_samples, transform=self.train_transform)
        val_dataset = FaultDataset(val_samples, transform=self.val_transform)
        
        self.fault_types = train_dataset.fault_types
        print(f"Fault types: {self.fault_types}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        # Calculate class weights for imbalanced datasets
        self.class_weights = train_dataset.get_class_weights().to(self.device)
        print(f"Class weights: {self.class_weights}")
        
        return train_dataset, val_dataset
    
    def build_model(self, pretrained: bool = True):
        """
        Build the model architecture.
        
        Args:
            pretrained: Use pretrained ImageNet weights
        """
        num_classes = len(self.fault_types)
        
        # Load pretrained model
        if self.model_name == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif self.model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif self.model_name == "efficientnet_b0":
            self.model = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Model {self.model_name} not supported")
        
        self.model = self.model.to(self.device)
        print(f"Built {self.model_name} model with {num_classes} classes")
        
        return self.model
    
    def train(
        self,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 5
    ) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
        
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(self.train_loader, desc="Training"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            val_loss, val_acc = self.evaluate()
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), 'best_fault_model.pth')
                print(f"Saved best model with validation accuracy: {val_acc:.4f}")
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_fault_model.pth'))
        print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
        
        return history
    
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate model on validation set.
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        return val_loss, val_acc
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """
        Predict fault type for a single image.
        
        Args:
            image_path: Path to image
        
        Returns:
            Tuple of (predicted_fault_type, confidence)
        """
        self.model.eval()
        
        image = Image.open(image_path).convert('RGB')
        image = self.val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        fault_type = self.fault_types[predicted.item()]
        confidence = confidence.item()
        
        return fault_type, confidence
    
    def add_predictions_to_dataset(self, confidence_threshold: float = 0.8):
        """
        Add model predictions to all samples in the dataset.
        
        Args:
            confidence_threshold: Minimum confidence for predictions
        """
        self.model.eval()
        
        print("Adding predictions to dataset...")
        for sample in tqdm(self.db.dataset):
            fault_type, confidence = self.predict(sample.filepath)
            
            sample["predicted_fault_type"] = fault_type
            sample["prediction_confidence"] = confidence
            sample["high_confidence"] = confidence >= confidence_threshold
            sample.save()
        
        print("Predictions added to all samples")
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'fault_types': self.fault_types,
            'model_name': self.model_name
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.fault_types = checkpoint['fault_types']
        self.model_name = checkpoint['model_name']
        
        self.build_model(pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    # Initialize database
    db = ManufacturingFaultDatabase("manufacturing_faults_demo")
    
    # Check if we have data
    stats = db.get_fault_statistics()
    print("\n=== Database Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    if stats["total_samples"] == 0:
        print("\n⚠️  No samples in database!")
        print("Please add fault images using manufacturing_fault_db.py first")
    else:
        print("\n=== Starting Training Pipeline ===")
        
        # Initialize classifier
        classifier = FaultClassifier(db, model_name="resnet18", pretrained=True)
        
        # Prepare data
        try:
            classifier.prepare_data(
                human_confirmed_only=True,
                test_size=0.2,
                batch_size=32,
                num_workers=0  # Set to 0 for Windows compatibility
            )
            
            # Train model
            print("\nStarting training...")
            history = classifier.train(
                num_epochs=20,
                learning_rate=0.001,
                patience=5
            )
            
            # Save model
            classifier.save_model("fault_classifier.pth")
            print("\n✓ Model saved as fault_classifier.pth")
            
            # Example prediction
            # fault_type, confidence = classifier.predict("path/to/test_image.jpg")
            # print(f"\nPrediction: {fault_type} (confidence: {confidence:.2%})")
            
            # Add predictions to dataset
            print("\nAdding predictions to entire dataset...")
            classifier.add_predictions_to_dataset(confidence_threshold=0.8)
            
            print("\n✓ Training complete!")
            print("Launch FiftyOne app to visualize results: db.launch_app()")
            
        except ValueError as e:
            print(f"\n⚠️  Error: {e}")
            print("Make sure you have human-confirmed samples in your database")