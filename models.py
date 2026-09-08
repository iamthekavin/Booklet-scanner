"""
Data models for the detection pipeline of the VEE Scanner project.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import cv2


class DetectionMethod(Enum):
    YOLO_DIRECT = 'yolo_direct'           # high-confidence YOLO, corners from bbox
    YOLO_CV_REFINED = 'yolo_cv_refined'   # YOLO bbox + CV corner refinement
    CV_FALLBACK = 'cv_fallback'           # pure CV (YOLO failed)
    NONE = 'none'                          # nothing detected


class ReviewFlag(Enum):
    OK = 'ok'
    LOW_CONFIDENCE = 'low_confidence'
    HAND_OCCLUSION = 'hand_occlusion'
    MULTIPLE_BOOKLETS = 'multiple_booklets'
    GEOMETRIC_INVALID = 'geometric_invalid'
    NO_DETECTION = 'no_detection'


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
        
    @property
    def height(self) -> float:
        return self.y2 - self.y1
        
    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.width / 2, self.y1 + self.height / 2)
        
    @property
    def area(self) -> float:
        return self.width * self.height
        
    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2])
        
    @property
    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)
        
    @property
    def to_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def __repr__(self) -> str:
        return f"BBox({self.class_name}: {self.confidence:.2f}, [{self.x1:.1f}, {self.y1:.1f}, {self.x2:.1f}, {self.y2:.1f}])"


@dataclass  
class QuadCorners:
    """Ordered quad corners: top-left, top-right, bottom-right, bottom-left."""
    points: np.ndarray  # shape (4, 2)
    
    def __post_init__(self):
        if self.points.shape != (4, 2):
            raise ValueError(f"Expected shape (4, 2) for corners, got {self.points.shape}")

    @property
    def top_left(self) -> np.ndarray:
        return self.points[0]
        
    @property
    def top_right(self) -> np.ndarray:
        return self.points[1]
        
    @property
    def bottom_right(self) -> np.ndarray:
        return self.points[2]
        
    @property
    def bottom_left(self) -> np.ndarray:
        return self.points[3]
        
    def area(self) -> float:
        """Calculate the area of the polygon using the Shoelace formula."""
        x = self.points[:, 0]
        y = self.points[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
    def is_convex(self) -> bool:
        """Check if the quadrilateral is convex."""
        return cv2.isContourConvex(self.points.astype(np.float32).reshape(-1, 1, 2))
        
    def as_float32(self) -> np.ndarray:
        return self.points.astype(np.float32)

    def __repr__(self) -> str:
        pts = ", ".join([f"({x:.1f}, {y:.1f})" for x, y in self.points])
        return f"QuadCorners([{pts}])"


@dataclass
class DetectionResult:
    bbox: BoundingBox | None
    corners: QuadCorners | None
    confidence: float
    detection_method: DetectionMethod
    review_flags: list[ReviewFlag]
    needs_review: bool
    hands_detected: list[BoundingBox]
    clutter_detected: list[BoundingBox]
    latency_ms: float
    raw_detections: list[BoundingBox]
    frame_shape: tuple[int, int]  # (height, width)
    
    def to_dict(self) -> dict:
        """Serialize the detection result to a dictionary."""
        def serialize_item(item):
            if isinstance(item, Enum):
                return item.value
            elif hasattr(item, 'to_dict'):
                return item.to_dict()
            elif isinstance(item, BoundingBox):
                return asdict(item)
            elif isinstance(item, QuadCorners):
                return item.points.tolist()
            elif isinstance(item, list):
                return [serialize_item(x) for x in item]
            return item

        result = {}
        for key, value in self.__dict__.items():
            result[key] = serialize_item(value)
        return result

    def __repr__(self) -> str:
        flags = [f.value for f in self.review_flags]
        return (f"DetectionResult(method={self.detection_method.value}, "
                f"conf={self.confidence:.2f}, needs_review={self.needs_review}, "
                f"flags={flags}, latency={self.latency_ms:.1f}ms)")
