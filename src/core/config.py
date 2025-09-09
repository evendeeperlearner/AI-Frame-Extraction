"""
Configuration management for adaptive frame extraction system.
Optimized for visual detail detection and SfM reconstruction quality.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Union
from pathlib import Path
import torch
from enum import Enum


class DINOModelType(Enum):
    """Enum for supported DINO model types"""
    DINOV2_BASE = "facebook/dinov2-base"
    DINOV2_SMALL = "facebook/dinov2-small" 
    DINOV2_LARGE = "facebook/dinov2-large"
    DINOV3_VITS16 = "facebook/dinov3-vits16-pretrain-lvd1689m"
    DINOV3_VITB16 = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    DINOV3_VIT7B16 = "facebook/dinov3-vit7b16-pretrain-lvd1689m"


@dataclass
class ProcessingConfig:
    """Main configuration for the adaptive extraction system"""
    
    # Video Processing
    input_video_path: Path
    output_frames_dir: Path
    temp_dir: Path = field(default_factory=lambda: Path("/tmp/adaptive_extraction"))
    
    # FFmpeg Settings
    ffmpeg_path: str = "ffmpeg"
    target_resolution: Tuple[int, int] = (1920, 1080)
    frame_format: str = "png"
    fps_extract: Optional[int] = None  # None = extract all frames
    
    # DINO Model Settings - Support both DINOv2 and DINOv3
    model_name: str = "facebook/dinov2-base"  # Default to DINOv2 for compatibility
    model_version: str = "dinov2"  # "dinov2" or "dinov3"
    device: str = "auto"  # auto-detect: MPS > CUDA > CPU
    batch_size: int = 8
    patch_size: int = 14
    
    # Visual Detail Detection Settings
    saliency_threshold: float = 0.75
    min_temporal_spacing: int = 30  # minimum frames between selections
    max_frames: Optional[int] = 1000
    feature_cache_enabled: bool = True
    
    # Dynamic Frame Extraction Rates
    feature_rich_threshold: float = 0.7  # Threshold for feature-rich regions
    feature_rich_fps: float = 4.0        # 4 fps in feature-rich areas
    feature_poor_fps: float = 0.5        # 1 frame every 2 seconds in feature-poor areas
    
    # SfM Optimization Settings
    overlap_ratio: float = 0.7
    track_length_threshold: int = 3
    viewpoint_diversity_weight: float = 0.3
    motion_blur_threshold: float = 0.1
    
    # Multi-criteria Scoring Weights
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        'spatial_complexity': 0.35,      # DINO dense feature analysis
        'semantic_richness': 0.25,       # Clustering-based semantic diversity
        'geometric_information': 0.25,   # Edge density and structure
        'texture_complexity': 0.15       # Local texture variation
    })
    
    # Performance Settings
    max_memory_gb: float = 8.0
    num_workers: int = 4
    enable_amp: bool = True  # Automatic Mixed Precision
    
    def __post_init__(self):
        """Initialize directories and auto-detect optimal device"""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect optimal device if not specified
        if self.device == "auto":
            self.device = self._detect_optimal_device()
        
        # Validate scoring weights sum to 1.0
        total_weight = sum(self.scoring_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total_weight}")
    
    def _detect_optimal_device(self) -> str:
        """Auto-detect the best available device: MPS > CUDA > CPU"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information"""
        device_info = {
            'device': self.device,
            'device_count': 1
        }
        
        if self.device == "mps":
            device_info.update({
                'type': 'Metal Performance Shaders',
                'available_memory_gb': 'Shared with system RAM'
            })
        elif self.device == "cuda":
            device_info.update({
                'type': f'NVIDIA {torch.cuda.get_device_name()}',
                'device_count': torch.cuda.device_count(),
                'available_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3
            })
        else:
            device_info.update({
                'type': 'CPU',
                'threads': torch.get_num_threads()
            })
        
        return device_info
    
    def adjust_batch_size_for_device(self) -> int:
        """Adjust batch size based on device capabilities"""
        if self.device == "mps":
            # MPS typically has less memory available than CUDA
            return min(self.batch_size, 6)
        elif self.device == "cuda":
            # Adjust based on GPU memory
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                max_batch_size = int(gpu_memory_gb * 1.5)  # Rough estimate
                return min(self.batch_size, max_batch_size)
        
        # CPU fallback - conservative batch size
        return min(self.batch_size, 4)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        is_dinov3 = self.model_version == "dinov3" or "dinov3" in self.model_name.lower()
        
        model_info = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "is_dinov3": is_dinov3,
            "estimated_params": self._estimate_model_parameters(),
            "recommended_batch_size": self._get_recommended_batch_size()
        }
        
        return model_info
    
    def _estimate_model_parameters(self) -> str:
        """Estimate model parameters based on model name"""
        model_name_lower = self.model_name.lower()
        
        if "vit7b16" in model_name_lower:
            return "~7B parameters"
        elif "vitb16" in model_name_lower:
            return "~86M parameters" 
        elif "vits16" in model_name_lower:
            return "~21M parameters"
        elif "dinov2-large" in model_name_lower:
            return "~300M parameters"
        elif "dinov2-base" in model_name_lower:
            return "~86M parameters"
        elif "dinov2-small" in model_name_lower:
            return "~21M parameters"
        else:
            return "Unknown"
    
    def _get_recommended_batch_size(self) -> int:
        """Get recommended batch size based on model size"""
        model_name_lower = self.model_name.lower()
        
        if "vit7b16" in model_name_lower:
            return 2  # Large model, small batch
        elif "vitb16" in model_name_lower or "dinov2-base" in model_name_lower:
            return 4  # Medium model, medium batch
        else:
            return 8  # Small model, larger batch
    
    def set_model(self, model_type: DINOModelType):
        """Set the model to use with proper configuration"""
        self.model_name = model_type.value
        
        # Set version based on model name
        if "dinov3" in model_type.value:
            self.model_version = "dinov3"
        else:
            self.model_version = "dinov2"
        
        # Adjust batch size for model
        self.batch_size = self._get_recommended_batch_size()


@dataclass
class QualityMetrics:
    """Quality metrics for frame assessment"""
    sharpness_score: float = 0.0
    exposure_score: float = 0.0
    motion_blur_score: float = 0.0
    noise_level: float = 0.0
    feature_density: float = 0.0
    composite_quality: float = 0.0
    
    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """Check if frame meets quality threshold"""
        return self.composite_quality >= threshold


@dataclass 
class SfMMetrics:
    """SfM-specific metrics for frame assessment"""
    geometric_information: float = 0.0
    viewpoint_uniqueness: float = 0.0
    tracking_potential: float = 0.0
    bundle_adjustment_weight: float = 0.0
    reconstruction_contribution: float = 0.0
    
    def is_sfm_valuable(self, threshold: float = 0.6) -> bool:
        """Check if frame contributes significantly to SfM reconstruction"""
        return self.reconstruction_contribution >= threshold