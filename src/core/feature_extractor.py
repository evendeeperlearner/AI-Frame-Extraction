"""
DINO-based dense feature extraction optimized for visual detail detection.
Supports both DINOv2 and DINOv3 models with MPS, CUDA, and CPU optimization.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image
import logging
from transformers import Dinov2Model, AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader, Dataset
import hashlib

from .config import ProcessingConfig
from .exceptions import FeatureExtractionError, DeviceError


class FrameDataset(Dataset):
    """Custom dataset for batch processing frames through DINOv3"""
    
    def __init__(self, frames: List[np.ndarray], processor, device: str):
        self.frames = frames
        self.processor = processor
        self.device = device
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        # Convert numpy array to PIL Image
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Ensure RGB format
        if frame.shape[2] == 3:
            pil_image = Image.fromarray(frame, mode='RGB')
        else:
            pil_image = Image.fromarray(frame[:, :, :3], mode='RGB')
        
        # Process image for DINOv3
        try:
            inputs = self.processor(images=pil_image, return_tensors="pt")
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'frame_idx': idx
            }
        except Exception as e:
            # Return dummy data if preprocessing fails
            return {
                'pixel_values': torch.zeros(3, 224, 224),
                'frame_idx': idx
            }


class DINOFeatureExtractor:
    """
    DINOv3-based dense feature extraction with memory optimization
    and device-specific optimizations for visual detail detection.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(config.device)
        
        # Initialize model and processor
        self._load_model()
        
        # Feature cache for repeated frames
        self.feature_cache = {} if config.feature_cache_enabled else None
        
        # Device-specific optimizations
        self._configure_device_optimizations()
        
        # Performance tracking
        self.stats = {
            'frames_processed': 0,
            'cache_hits': 0,
            'oom_recoveries': 0,
            'processing_time': 0.0
        }
    
    def _load_model(self):
        """Load DINO model (v2 or v3) and image processor with error handling"""
        try:
            model_info = self.config.get_model_info()
            is_dinov3 = model_info["is_dinov3"]
            
            self.logger.info(f"Loading {self.config.model_version.upper()} model: {self.config.model_name}")
            self.logger.info(f"Model info: {model_info['estimated_params']}")
            
            # Load model based on version
            if is_dinov3:
                # DINOv3 uses AutoModel instead of Dinov2Model
                self.model = AutoModel.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=True  # DINOv3 may require this
                )
            else:
                # DINOv2 uses specific Dinov2Model
                self.model = Dinov2Model.from_pretrained(
                    self.config.model_name
                )
            
            # Load processor (same for both versions)
            self.processor = AutoImageProcessor.from_pretrained(self.config.model_name)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Enable compilation if supported
            if hasattr(torch, 'compile') and self.device.type in ['cuda', 'mps']:
                try:
                    self.model = torch.compile(self.model)
                    self.logger.info("Model compilation enabled")
                except Exception as e:
                    self.logger.warning(f"Model compilation failed: {e}")
            
            version_name = "DINOv3" if is_dinov3 else "DINOv2"
            self.logger.info(f"{version_name} model loaded successfully on {self.device}")
        
        except Exception as e:
            version_name = "DINOv3" if self.config.get_model_info()["is_dinov3"] else "DINOv2"
            raise FeatureExtractionError(f"Failed to load {version_name} model: {e}")
    
    def _configure_device_optimizations(self):
        """Configure device-specific optimizations"""
        try:
            if self.device.type == 'mps':
                # MPS-specific optimizations (no cache management needed)
                self.logger.info("Configured MPS optimizations")
                
            elif self.device.type == 'cuda':
                # CUDA-specific optimizations
                torch.backends.cudnn.benchmark = True
                torch.cuda.empty_cache()
                
                # Get GPU info
                gpu_name = torch.cuda.get_device_name()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"Configured CUDA optimizations for {gpu_name} ({gpu_memory:.1f}GB)")
                
            else:
                # CPU optimizations
                torch.set_num_threads(min(self.config.num_workers, torch.get_num_threads()))
                self.logger.info(f"Configured CPU optimizations ({torch.get_num_threads()} threads)")
        
        except Exception as e:
            self.logger.warning(f"Device optimization failed: {e}")
    
    def extract_features_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Extract dense features from a batch of frames with error recovery"""
        if not frames:
            return torch.empty(0, 0, 768)  # Empty tensor with correct dimensions
        
        import time
        start_time = time.time()
        
        try:
            # Adjust batch size if necessary
            batch_size = min(len(frames), self.config.adjust_batch_size_for_device())
            
            # Create dataset and dataloader
            dataset = FrameDataset(frames, self.processor, self.device.type)
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # Avoid multiprocessing issues with transformers
                pin_memory=self.device.type in ['cuda', 'mps']
            )
            
            all_features = []
            
            with torch.no_grad():
                if self.config.enable_amp and self.device.type != 'cpu':
                    with torch.autocast(device_type=self.device.type):
                        all_features = self._process_batches(dataloader)
                else:
                    all_features = self._process_batches(dataloader)
            
            # Concatenate results
            if all_features:
                result = torch.cat(all_features, dim=0)
            else:
                result = torch.empty(0, 0, 768)
            
            # Update stats
            self.stats['frames_processed'] += len(frames)
            self.stats['processing_time'] += time.time() - start_time
            
            return result
        
        except Exception as e:
            self.logger.error(f"Batch feature extraction failed: {e}")
            # Return empty tensor to allow pipeline to continue
            return torch.empty(0, 0, 768)
    
    def _process_batches(self, dataloader: DataLoader) -> List[torch.Tensor]:
        """Process batches with OOM recovery"""
        all_features = []
        
        for batch in dataloader:
            try:
                pixel_values = batch['pixel_values'].to(self.device)
                
                # Forward pass
                outputs = self.model(pixel_values)
                
                # Get dense features (patch embeddings)
                patch_embeddings = outputs.last_hidden_state  # [batch_size, num_patches, hidden_size]
                
                # Move to CPU to free GPU memory
                all_features.append(patch_embeddings.cpu())
                
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower():
                    self.logger.warning(f"OOM detected, attempting recovery")
                    self.stats['oom_recoveries'] += 1
                    
                    # Clear cache and try single frame processing
                    self._clear_device_cache()
                    
                    # Process frames one by one
                    for i in range(pixel_values.shape[0]):
                        try:
                            single_frame = pixel_values[i:i+1]
                            outputs = self.model(single_frame)
                            all_features.append(outputs.last_hidden_state.cpu())
                        except Exception as single_e:
                            self.logger.error(f"Single frame processing failed: {single_e}")
                            # Add empty tensor as placeholder
                            all_features.append(torch.zeros(1, 257, 768))
                else:
                    raise
        
        return all_features
    
    def extract_single_frame_features(self, frame: np.ndarray) -> torch.Tensor:
        """Extract features from a single frame with caching"""
        
        # Check cache if enabled
        if self.feature_cache is not None:
            frame_hash = self._compute_frame_hash(frame)
            if frame_hash in self.feature_cache:
                self.stats['cache_hits'] += 1
                return self.feature_cache[frame_hash]
        
        # Extract features
        features = self.extract_features_batch([frame])
        
        if features.numel() == 0:
            # Return dummy features if extraction failed
            return torch.zeros(257, 768)
        
        result = features[0] if len(features) > 0 else torch.zeros(257, 768)
        
        # Cache result
        if self.feature_cache is not None and frame_hash is not None:
            self.feature_cache[frame_hash] = result
            
            # Limit cache size
            if len(self.feature_cache) > 1000:
                # Remove oldest entries
                keys_to_remove = list(self.feature_cache.keys())[:100]
                for key in keys_to_remove:
                    del self.feature_cache[key]
        
        return result
    
    def _compute_frame_hash(self, frame: np.ndarray) -> Optional[str]:
        """Compute hash for frame caching"""
        try:
            # Use a subset of pixels for faster hashing
            frame_subset = frame[::8, ::8, :].tobytes()
            return hashlib.md5(frame_subset).hexdigest()
        except:
            return None
    
    def _clear_device_cache(self):
        """Clear device-specific cache"""
        try:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            # Note: MPS doesn't have empty_cache() method
        except Exception as e:
            self.logger.warning(f"Cache clearing failed: {e}")
    
    def compute_spatial_complexity(self, features: torch.Tensor) -> float:
        """Compute spatial complexity from DINO dense features"""
        try:
            if features.numel() == 0:
                return 0.0
            
            feat_np = features.numpy()
            
            # Skip CLS token (first token) and use spatial tokens
            spatial_features = feat_np[1:] if feat_np.shape[0] > 1 else feat_np
            
            # Reshape to spatial grid if possible
            num_patches = spatial_features.shape[0]
            grid_size = int(np.sqrt(num_patches))
            
            if grid_size * grid_size == num_patches and grid_size > 1:
                # Reshape to spatial grid
                spatial_grid = spatial_features.reshape(grid_size, grid_size, -1)
                
                # Compute spatial gradients
                grad_x = np.diff(spatial_grid, axis=1)
                grad_y = np.diff(spatial_grid, axis=0)
                
                # Compute gradient magnitude
                grad_magnitude_x = np.sqrt(np.sum(grad_x**2, axis=2))
                grad_magnitude_y = np.sqrt(np.sum(grad_y**2, axis=2))
                
                # Average gradient magnitude
                complexity = (np.mean(grad_magnitude_x) + np.mean(grad_magnitude_y)) / 2
            else:
                # Fallback: use standard deviation across patches
                complexity = np.mean(np.std(spatial_features, axis=0))
            
            # Normalize to [0, 1] range
            return float(np.clip(complexity / 10.0, 0, 1))
        
        except Exception as e:
            self.logger.warning(f"Spatial complexity computation failed: {e}")
            return 0.0
    
    def compute_semantic_richness(self, features: torch.Tensor) -> float:
        """Compute semantic richness using feature clustering"""
        try:
            if features.numel() == 0:
                return 0.0
            
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score
            
            feat_np = features.numpy()
            
            # Skip CLS token if present
            if feat_np.shape[0] > 1:
                feat_np = feat_np[1:]
            
            # Determine number of clusters
            n_clusters = min(8, max(2, feat_np.shape[0] // 4))
            
            if n_clusters < 2:
                return 0.5
            
            # Standardize features
            scaler = StandardScaler()
            feat_scaled = scaler.fit_transform(feat_np)
            
            # K-means clustering
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(feat_scaled)
                
                # Compute silhouette score
                silhouette_avg = silhouette_score(feat_scaled, cluster_labels)
                richness_score = (silhouette_avg + 1) / 2  # Convert to [0, 1]
                
            except ValueError:
                # Fallback if clustering fails
                richness_score = np.std(feat_np.mean(axis=1)) / (np.mean(feat_np.mean(axis=1)) + 1e-6)
                richness_score = min(richness_score, 1.0)
            
            return float(np.clip(richness_score, 0, 1))
        
        except Exception as e:
            self.logger.warning(f"Semantic richness computation failed: {e}")
            return 0.5
    
    def compute_feature_statistics(self, features: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive statistical measures for feature analysis"""
        try:
            if features.numel() == 0:
                return {
                    'mean_activation': 0.0,
                    'std_activation': 0.0,
                    'max_activation': 0.0,
                    'feature_density': 0.0,
                    'spatial_variance': 0.0
                }
            
            feat_np = features.numpy()
            
            stats = {
                'mean_activation': float(np.mean(feat_np)),
                'std_activation': float(np.std(feat_np)),
                'max_activation': float(np.max(feat_np)),
                'feature_density': float(np.mean(np.std(feat_np, axis=-1))),
                'spatial_variance': float(np.var(np.mean(feat_np, axis=-1)))
            }
            
            return stats
        
        except Exception as e:
            self.logger.warning(f"Feature statistics computation failed: {e}")
            return {
                'mean_activation': 0.0,
                'std_activation': 0.0,
                'max_activation': 0.0,
                'feature_density': 0.0,
                'spatial_variance': 0.0
            }
    
    def get_processing_stats(self) -> Dict[str, Union[int, float]]:
        """Get processing statistics"""
        stats = self.stats.copy()
        if stats['frames_processed'] > 0:
            stats['avg_processing_time'] = stats['processing_time'] / stats['frames_processed']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['frames_processed']
        else:
            stats['avg_processing_time'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'feature_cache') and self.feature_cache:
                self.feature_cache.clear()
            
            self._clear_device_cache()
            
            if hasattr(self, 'stats'):
                self.logger.info(f"Cleanup complete. Final stats: {self.get_processing_stats()}")
            else:
                self.logger.info("Cleanup complete.")
        
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass