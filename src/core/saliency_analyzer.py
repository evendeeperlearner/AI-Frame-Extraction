"""
Advanced saliency analysis using DINO features combined with traditional CV methods.
Supports both DINOv2 and DINOv3 models for identifying frames with rich visual details for SfM reconstruction.
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import logging

from .config import ProcessingConfig, QualityMetrics, SfMMetrics
from .feature_extractor import DINOFeatureExtractor
from .exceptions import SaliencyAnalysisError


class SaliencyAnalyzer:
    """
    Advanced saliency analysis combining DINOv3 dense features with traditional
    computer vision techniques for identifying visually detailed regions.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_extractor = DINOFeatureExtractor(config)
        
        # Performance tracking
        self.stats = {
            'frames_analyzed': 0,
            'high_saliency_frames': 0,
            'processing_time': 0.0
        }
    
    def compute_comprehensive_saliency_score(self, frame: np.ndarray, quality_metrics: QualityMetrics) -> Dict[str, float]:
        """
        Compute comprehensive saliency score combining multiple analysis methods.
        
        Args:
            frame: RGB frame as numpy array
            quality_metrics: Pre-computed quality metrics
            
        Returns:
            Dictionary containing individual scores and composite score
        """
        import time
        start_time = time.time()
        
        try:
            # Extract DINOv3 dense features
            features = self.feature_extractor.extract_single_frame_features(frame)
            
            # Initialize scores dictionary
            scores = {}
            
            # 1. Spatial complexity from DINOv3 dense features (35% weight)
            scores['spatial_complexity'] = self.feature_extractor.compute_spatial_complexity(features)
            
            # 2. Semantic richness via clustering analysis (25% weight)
            scores['semantic_richness'] = self.feature_extractor.compute_semantic_richness(features)
            
            # 3. Geometric information density (25% weight)
            scores['geometric_information'] = self._compute_geometric_information(frame)
            
            # 4. Texture complexity analysis (15% weight)
            scores['texture_complexity'] = self._compute_texture_complexity(frame)
            
            # 5. Additional metrics from quality assessment
            scores['edge_density'] = self._compute_edge_density(frame)
            scores['local_variance'] = self._compute_local_variance(frame)
            scores['structural_similarity'] = self._compute_structural_complexity(frame)
            
            # Compute weighted composite score using config weights
            composite_score = sum(
                scores.get(key, 0.0) * weight 
                for key, weight in self.config.scoring_weights.items()
            )
            
            scores['composite_score'] = np.clip(composite_score, 0.0, 1.0)
            
            # Update statistics
            self.stats['frames_analyzed'] += 1
            self.stats['processing_time'] += time.time() - start_time
            
            if scores['composite_score'] >= self.config.saliency_threshold:
                self.stats['high_saliency_frames'] += 1
            
            return scores
        
        except Exception as e:
            self.logger.error(f"Saliency computation failed: {e}")
            raise SaliencyAnalysisError(f"Failed to compute saliency scores: {e}")
    
    def _compute_geometric_information(self, frame: np.ndarray) -> float:
        """Compute geometric information content using multiple detectors"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # 1. Corner detection (Harris)
            corners = cv2.cornerHarris(gray, 2, 3, 0.04)
            corner_density = (corners > 0.01 * corners.max()).sum() / gray.size
            
            # 2. Edge detection (Canny)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / gray.size
            
            # 3. FAST keypoints
            fast = cv2.FastFeatureDetector_create()
            keypoints = fast.detect(gray, None)
            keypoint_density = len(keypoints) / gray.size * 10000  # Scale for visibility
            
            # 4. Gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            avg_gradient = np.mean(grad_magnitude) / 255.0
            
            # Combine geometric measures
            geometric_score = (
                0.3 * min(corner_density * 100, 1.0) +
                0.25 * min(edge_density * 20, 1.0) +
                0.25 * min(keypoint_density, 1.0) +
                0.2 * min(avg_gradient * 4, 1.0)
            )
            
            return float(np.clip(geometric_score, 0, 1))
        
        except Exception as e:
            self.logger.warning(f"Geometric information computation failed: {e}")
            return 0.0
    
    def _compute_texture_complexity(self, frame: np.ndarray) -> float:
        """Compute texture complexity using Local Binary Patterns and GLCM"""
        try:
            from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
            
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # 1. Local Binary Pattern entropy
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Compute LBP histogram entropy
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, density=True)
            hist = hist[hist > 0]
            lbp_entropy = -np.sum(hist * np.log2(hist + 1e-10))
            max_entropy = np.log2(n_points + 2)
            lbp_score = lbp_entropy / max_entropy
            
            # 2. Gray-Level Co-occurrence Matrix properties
            # Downsample for faster computation
            gray_small = cv2.resize(gray, (64, 64))
            
            # Compute GLCM
            glcm = graycomatrix(
                gray_small, distances=[1], angles=[0, 45, 90, 135], 
                levels=16, symmetric=True, normed=True
            )
            
            # Extract GLCM properties
            contrast = graycoprops(glcm, 'contrast').mean()
            energy = graycoprops(glcm, 'energy').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            
            # Combine texture measures
            texture_score = (
                0.4 * lbp_score +
                0.3 * min(contrast / 50.0, 1.0) +
                0.15 * (1.0 - energy) +  # Lower energy = more complex
                0.15 * (1.0 - homogeneity)  # Lower homogeneity = more complex
            )
            
            return float(np.clip(texture_score, 0, 1))
        
        except Exception as e:
            self.logger.warning(f"Texture complexity computation failed: {e}")
            return 0.0
    
    def _compute_edge_density(self, frame: np.ndarray) -> float:
        """Compute edge density using multiple edge detectors"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # 1. Canny edges
            canny = cv2.Canny(gray, 50, 150)
            canny_density = np.sum(canny > 0) / gray.size
            
            # 2. Sobel edges
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_edges = sobel_magnitude > np.percentile(sobel_magnitude, 90)
            sobel_density = np.sum(sobel_edges) / gray.size
            
            # 3. Laplacian edges
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_edges = np.abs(laplacian) > np.percentile(np.abs(laplacian), 90)
            laplacian_density = np.sum(laplacian_edges) / gray.size
            
            # Combine edge densities
            edge_score = (
                0.5 * min(canny_density * 20, 1.0) +
                0.3 * min(sobel_density * 20, 1.0) +
                0.2 * min(laplacian_density * 20, 1.0)
            )
            
            return float(np.clip(edge_score, 0, 1))
        
        except Exception as e:
            self.logger.warning(f"Edge density computation failed: {e}")
            return 0.0
    
    def _compute_local_variance(self, frame: np.ndarray) -> float:
        """Compute local variance to measure fine-grained detail"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
            
            # Compute local variance using sliding window
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray, -1, kernel)
            local_variance = cv2.filter2D(gray**2, -1, kernel) - local_mean**2
            
            # Normalize and compute score
            avg_variance = np.mean(local_variance)
            variance_score = min(avg_variance / 1000.0, 1.0)  # Normalize
            
            return float(variance_score)
        
        except Exception as e:
            self.logger.warning(f"Local variance computation failed: {e}")
            return 0.0
    
    def _compute_structural_complexity(self, frame: np.ndarray) -> float:
        """Compute structural complexity using frequency domain analysis"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
            
            # Apply FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Compute energy in different frequency bands
            h, w = magnitude_spectrum.shape
            center_x, center_y = h // 2, w // 2
            
            # High frequency energy (detail)
            high_freq_mask = np.zeros((h, w))
            radius_outer = min(h, w) // 3
            radius_inner = min(h, w) // 6
            
            y, x = np.ogrid[:h, :w]
            mask_outer = (x - center_x)**2 + (y - center_y)**2 <= radius_outer**2
            mask_inner = (x - center_x)**2 + (y - center_y)**2 <= radius_inner**2
            high_freq_mask = mask_outer & ~mask_inner
            
            high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
            total_energy = np.sum(magnitude_spectrum)
            
            structural_score = high_freq_energy / (total_energy + 1e-6)
            
            return float(np.clip(structural_score, 0, 1))
        
        except Exception as e:
            self.logger.warning(f"Structural complexity computation failed: {e}")
            return 0.0
    
    def assess_sfm_potential(self, frame: np.ndarray, saliency_scores: Dict[str, float]) -> SfMMetrics:
        """Assess frame's potential contribution to SfM reconstruction"""
        try:
            # Geometric information weight
            geometric_info = saliency_scores.get('geometric_information', 0.0)
            
            # Viewpoint uniqueness (simplified - would need motion analysis for full implementation)
            viewpoint_uniqueness = saliency_scores.get('spatial_complexity', 0.0)
            
            # Tracking potential based on feature density
            tracking_potential = (
                0.4 * saliency_scores.get('geometric_information', 0.0) +
                0.3 * saliency_scores.get('edge_density', 0.0) +
                0.3 * saliency_scores.get('texture_complexity', 0.0)
            )
            
            # Bundle adjustment weight (features that are stable for optimization)
            bundle_adjustment_weight = (
                0.5 * saliency_scores.get('structural_similarity', 0.0) +
                0.3 * (1.0 - saliency_scores.get('local_variance', 0.0)) +  # Less noise
                0.2 * saliency_scores.get('geometric_information', 0.0)
            )
            
            # Overall reconstruction contribution
            reconstruction_contribution = (
                0.3 * geometric_info +
                0.25 * viewpoint_uniqueness +
                0.25 * tracking_potential +
                0.2 * bundle_adjustment_weight
            )
            
            return SfMMetrics(
                geometric_information=geometric_info,
                viewpoint_uniqueness=viewpoint_uniqueness,
                tracking_potential=tracking_potential,
                bundle_adjustment_weight=bundle_adjustment_weight,
                reconstruction_contribution=reconstruction_contribution
            )
        
        except Exception as e:
            self.logger.warning(f"SfM potential assessment failed: {e}")
            return SfMMetrics()
    
    def get_processing_stats(self) -> Dict[str, float]:
        """Get processing statistics"""
        stats = self.stats.copy()
        if stats['frames_analyzed'] > 0:
            stats['avg_processing_time'] = stats['processing_time'] / stats['frames_analyzed']
            stats['high_saliency_rate'] = stats['high_saliency_frames'] / stats['frames_analyzed']
        else:
            stats['avg_processing_time'] = 0.0
            stats['high_saliency_rate'] = 0.0
        
        # Add feature extractor stats
        feature_stats = self.feature_extractor.get_processing_stats()
        stats.update(feature_stats)
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.feature_extractor.cleanup()
            self.logger.info(f"Saliency analyzer cleanup complete. Stats: {self.get_processing_stats()}")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")


class AdaptiveFrameSampler:
    """
    Adaptive frame sampling based on comprehensive saliency analysis
    optimized for SfM reconstruction quality.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.saliency_analyzer = SaliencyAnalyzer(config)
        
        # Temporal distribution tracking
        self.selected_frames_info = []
        self.motion_history = []
        
        # Performance stats
        self.stats = {
            'total_frames_processed': 0,
            'frames_selected': 0,
            'temporal_rejections': 0,
            'quality_rejections': 0,
            'processing_time': 0.0
        }
    
    def select_frames_adaptive(self, video_processor) -> List[Tuple[int, np.ndarray, Dict[str, float], SfMMetrics]]:
        """
        Select frames adaptively based on visual detail analysis and SfM optimization.
        
        Returns:
            List of (frame_index, frame_array, saliency_scores, sfm_metrics)
        """
        import time
        start_time = time.time()
        
        selected_frames = []
        last_selected_frame = -self.config.min_temporal_spacing
        
        self.logger.info("Starting adaptive frame selection...")
        
        try:
            # Stream through all frames
            for frame_idx, frame, quality_metrics in video_processor.extract_frames_streaming():
                self.stats['total_frames_processed'] += 1
                
                # Log progress every 100 frames
                if frame_idx % 100 == 0:
                    self.logger.info(f"Processing frame {frame_idx}, selected: {len(selected_frames)}")
                
                # Compute comprehensive saliency scores
                try:
                    saliency_scores = self.saliency_analyzer.compute_comprehensive_saliency_score(
                        frame, quality_metrics
                    )
                except Exception as e:
                    self.logger.warning(f"Saliency analysis failed for frame {frame_idx}: {e}")
                    continue
                
                # Assess SfM potential
                sfm_metrics = self.saliency_analyzer.assess_sfm_potential(frame, saliency_scores)
                
                # Check selection criteria
                if self._should_select_frame(
                    frame_idx, frame, saliency_scores, sfm_metrics, 
                    quality_metrics, last_selected_frame, len(selected_frames)
                ):
                    selected_frames.append((frame_idx, frame, saliency_scores, sfm_metrics))
                    last_selected_frame = frame_idx
                    self.stats['frames_selected'] += 1
                    
                    self.logger.debug(f"Selected frame {frame_idx} (composite: {saliency_scores['composite_score']:.3f})")
                    
                    # Check if we've reached the maximum frame limit
                    if self.config.max_frames and len(selected_frames) >= self.config.max_frames:
                        self.logger.info(f"Reached maximum frame limit: {self.config.max_frames}")
                        break
                
                # Early termination if video processing fails
                if frame_idx > 10000:  # Safety limit
                    self.logger.warning("Reached safety limit for frame processing")
                    break
            
            # Post-process selected frames for optimal SfM distribution
            optimized_frames = self._optimize_temporal_distribution(selected_frames)
            
            self.stats['processing_time'] = time.time() - start_time
            
            self.logger.info(
                f"Frame selection complete: {len(optimized_frames)}/{self.stats['total_frames_processed']} "
                f"frames selected ({len(optimized_frames)/max(self.stats['total_frames_processed'], 1)*100:.1f}%)"
            )
            
            return optimized_frames
        
        except Exception as e:
            self.logger.error(f"Adaptive frame selection failed: {e}")
            return selected_frames  # Return what we have so far
    
    def _should_select_frame(
        self, 
        frame_idx: int,
        frame: np.ndarray,
        saliency_scores: Dict[str, float],
        sfm_metrics: SfMMetrics,
        quality_metrics: QualityMetrics,
        last_selected: int,
        num_selected: int
    ) -> bool:
        """Determine if a frame should be selected based on multiple criteria"""
        
        # 1. Check temporal spacing constraint
        if frame_idx - last_selected < self.config.min_temporal_spacing:
            self.stats['temporal_rejections'] += 1
            return False
        
        # 2. Check minimum quality threshold
        if not quality_metrics.is_high_quality(threshold=0.5):  # Lower threshold for quality
            self.stats['quality_rejections'] += 1
            return False
        
        # 3. Check saliency threshold
        if saliency_scores['composite_score'] < self.config.saliency_threshold:
            return False
        
        # 4. Check SfM contribution potential
        if not sfm_metrics.is_sfm_valuable(threshold=0.5):  # Moderate threshold
            return False
        
        # 5. Always select if we don't have enough frames yet (minimum for SfM)
        if num_selected < 10:
            return True
        
        # 6. Additional selection criteria for diversity
        if num_selected < 50:  # Be more permissive for initial frames
            return saliency_scores['composite_score'] > self.config.saliency_threshold * 0.8
        
        # 7. Stricter criteria for later frames to ensure high quality
        return saliency_scores['composite_score'] > self.config.saliency_threshold * 1.1
    
    def _optimize_temporal_distribution(self, selected_frames: List) -> List:
        """Optimize temporal distribution of selected frames for better SfM reconstruction"""
        if len(selected_frames) <= 10:
            return selected_frames
        
        try:
            # Sort by frame index
            selected_frames.sort(key=lambda x: x[0])
            
            # Analyze temporal distribution
            frame_indices = [f[0] for f in selected_frames]
            gaps = np.diff(frame_indices)
            
            # Identify large gaps that might hurt tracking
            median_gap = np.median(gaps)
            large_gaps = gaps > median_gap * 3
            
            if np.any(large_gaps):
                self.logger.info(f"Found {np.sum(large_gaps)} large temporal gaps, considering optimization")
                
                # Could implement gap filling logic here
                # For now, just log the information
                large_gap_positions = np.where(large_gaps)[0]
                for pos in large_gap_positions:
                    gap_size = gaps[pos]
                    self.logger.debug(f"Large gap at position {pos}: {gap_size} frames")
            
            return selected_frames
        
        except Exception as e:
            self.logger.warning(f"Temporal distribution optimization failed: {e}")
            return selected_frames
    
    def get_selection_stats(self) -> Dict[str, float]:
        """Get comprehensive selection statistics"""
        stats = self.stats.copy()
        
        # Calculate derived statistics
        if stats['total_frames_processed'] > 0:
            stats['selection_rate'] = stats['frames_selected'] / stats['total_frames_processed']
            stats['temporal_rejection_rate'] = stats['temporal_rejections'] / stats['total_frames_processed']
            stats['quality_rejection_rate'] = stats['quality_rejections'] / stats['total_frames_processed']
            stats['avg_processing_time'] = stats['processing_time'] / stats['total_frames_processed']
        else:
            stats['selection_rate'] = 0.0
            stats['temporal_rejection_rate'] = 0.0
            stats['quality_rejection_rate'] = 0.0
            stats['avg_processing_time'] = 0.0
        
        # Add saliency analyzer stats
        saliency_stats = self.saliency_analyzer.get_processing_stats()
        stats.update({f'saliency_{k}': v for k, v in saliency_stats.items()})
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.saliency_analyzer.cleanup()
            self.logger.info(f"Frame sampler cleanup complete. Final stats: {self.get_selection_stats()}")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")