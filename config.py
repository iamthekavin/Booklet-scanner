"""
Configuration module for the VEE Scanner project.
"""
from dataclasses import dataclass, field
from pathlib import Path
import torch

@dataclass
class DetectorConfig:
    # Model settings
    model_path: str = 'yolo11n.pt'  # default pretrained model
    custom_model_path: str | None = 'models/booklet_detector.pt'  # custom fine-tuned model
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'cuda:0'
    
    # Detection thresholds
    confidence_threshold: float = 0.35
    high_confidence: float = 0.6  # above this: trust YOLO directly
    medium_confidence: float = 0.3  # 0.3-0.6: YOLO + CV refinement
    # below 0.3: flag for operator review
    
    iou_threshold: float = 0.5
    
    # Image settings
    input_size: int = 640  # YOLO input resolution
    
    # Target output size for warped booklet
    warp_width: int = 1200
    warp_height: int = 1700  # roughly A4 proportions
    
    # Corner refinement settings
    roi_padding: float = 0.05  # 5% padding around YOLO bbox for corner refinement
    canny_low: int = 50
    canny_high: int = 150
    corner_quality: float = 0.01
    min_corner_distance: int = 30
    
    # Class mappings (COCO pretrained)
    COCO_BOOKLET_CLASSES: tuple = (73,)  # 'book' in COCO
    COCO_HAND_CLASS: int = 0  # 'person' as proxy (no hand class in COCO)
    
    # Custom class names (Phase 2)
    CUSTOM_CLASS_NAMES: dict = field(default_factory=lambda: {
        0: 'booklet',
        1: 'hand', 
        2: 'background_clutter'
    })
    
    # Geometric validation
    min_booklet_area_ratio: float = 0.05  # booklet must be >5% of frame area
    max_booklet_area_ratio: float = 0.95  # booklet must be <95% of frame area
    min_quad_angle: float = 30.0  # minimum interior angle (degrees)
    max_aspect_ratio: float = 3.0  # max width/height ratio
    
    # Latency budget
    max_latency_ms: float = 800.0
    target_latency_ms: float = 150.0
    
    # ONNX export settings
    onnx_export_path: str = 'models/booklet_detector.onnx'


def resolve_device(device_str: str) -> str:
    """
    Resolve the device string to a valid device, checking CUDA availability.
    
    Args:
        device_str: The requested device string ('auto', 'cpu', 'cuda', etc.)
        
    Returns:
        A valid device string for PyTorch/Ultralytics.
    """
    if device_str == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_str.startswith('cuda') and not torch.cuda.is_available():
        import logging
        logging.warning(f"Requested device '{device_str}' but CUDA is not available. Falling back to CPU.")
        return 'cpu'
    return device_str


@dataclass
class PathConfig:
    project_root: Path
    models_dir: Path 
    output_dir: Path
    test_images_dir: Path
    training_dir: Path

    @classmethod
    def from_project_root(cls, root_path: str | Path) -> "PathConfig":
        """
        Create a PathConfig by deriving standard subdirectories from the project root.
        
        Args:
            root_path: The root directory of the project.
            
        Returns:
            An instance of PathConfig.
        """
        root = Path(root_path).resolve()
        return cls(
            project_root=root,
            models_dir=root / "models",
            output_dir=root / "output",
            test_images_dir=root / "test_images",
            training_dir=root / "training"
        )
