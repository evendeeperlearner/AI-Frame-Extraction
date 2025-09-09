"""
Dynamic frame sampler with variable extraction rates based on visual detail density.
Extracts 1 frame per 2 seconds in feature-poor areas and 4 fps in feature-rich areas.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import time

from .config import ProcessingConfig, QualityMetrics, SfMMetrics
from .saliency_analyzer import SaliencyAnalyzer
from .exceptions import SaliencyAnalysisError


class DynamicFrameSampler:
    """
    Dynamic frame sampling with adaptive extraction rates based on visual detail density.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.saliency_analyzer = SaliencyAnalyzer(config)
        
        # Dynamic sampling parameters
        self.current_region_type = "unknown"  # "rich" or "poor"
        self.region_start_frame = 0
        self.frames_in_current_region = []
        
        # Frame timing tracking
        self.last_extracted_frame = -999
        self.video_fps = 30.0  # Will be updated from video info
        
        # Statistics
        self.stats = {
            'total_frames_processed': 0,
            'frames_selected': 0,
            'feature_rich_regions': 0,
            'feature_poor_regions': 0,
            'feature_rich_frames_selected': 0,
            'feature_poor_frames_selected': 0,
            'processing_time': 0.0
        }
    
    def select_frames_dynamic(self, video_processor) -> List[Tuple[int, np.ndarray, Dict[str, float], SfMMetrics]]:
        """
        Select frames with dynamic extraction rates based on visual detail analysis.
        
        Returns:
            List of (frame_index, frame_array, saliency_scores, sfm_metrics)
        """
        start_time = time.time()
        self.logger.info("Starting dynamic frame selection with adaptive rates...")
        
        # Get video info to determine FPS
        video_info = video_processor.get_video_info()
        self.video_fps = video_info.get('fps', 30.0)
        
        self.logger.info(f"Video FPS: {self.video_fps:.2f}")
        self.logger.info(f"Feature-rich extraction: {self.config.feature_rich_fps} fps")
        self.logger.info(f"Feature-poor extraction: {self.config.feature_poor_fps} fps (1 frame every {1/self.config.feature_poor_fps:.1f}s)")
        
        selected_frames = []
        analysis_window = []  # Buffer for analyzing region characteristics
        window_size = int(self.video_fps * 2)  # 2-second analysis window
        
        try:
            # First pass: Analyze all frames to identify regions
            self.logger.info("Pass 1: Analyzing video regions...")
            region_map = self._analyze_video_regions(video_processor)
            
            # Second pass: Extract frames based on region analysis
            self.logger.info("Pass 2: Dynamic frame extraction...")
            selected_frames = self._extract_frames_by_regions(video_processor, region_map)
            
            self.stats['processing_time'] = time.time() - start_time
            
            self.logger.info(
                f"Dynamic extraction complete:\n"
                f"  Total frames processed: {self.stats['total_frames_processed']}\n"
                f"  Feature-rich regions: {self.stats['feature_rich_regions']}\n"
                f"  Feature-poor regions: {self.stats['feature_poor_regions']}\n"
                f"  Frames selected from rich regions: {self.stats['feature_rich_frames_selected']}\n"
                f"  Frames selected from poor regions: {self.stats['feature_poor_frames_selected']}\n"
                f"  Total selected: {len(selected_frames)}"
            )
            
            return selected_frames
        
        except Exception as e:
            self.logger.error(f"Dynamic frame selection failed: {e}")
            return selected_frames  # Return partial results
    
    def _analyze_video_regions(self, video_processor) -> Dict[int, str]:
        """
        Analyze video to identify feature-rich and feature-poor regions.
        
        Returns:
            Dictionary mapping frame_index to region_type ("rich" or "poor")
        """
        region_map = {}
        frame_scores = []
        
        # Sample frames at regular intervals for analysis
        sample_interval = max(1, int(self.video_fps / 2))  # Sample every 0.5 seconds
        
        self.logger.info(f"Analyzing video regions (sampling every {sample_interval} frames)...")
        
        frame_count = 0
        for frame_idx, frame, quality_metrics in video_processor.extract_frames_streaming():
            frame_count += 1
            
            # Sample frames for region analysis
            if frame_idx % sample_interval == 0:
                try:
                    # Compute saliency score
                    saliency_scores = self.saliency_analyzer.compute_comprehensive_saliency_score(
                        frame, quality_metrics
                    )
                    
                    frame_scores.append((frame_idx, saliency_scores['composite_score']))
                    
                    if len(frame_scores) % 50 == 0:
                        self.logger.info(f"Analyzed {len(frame_scores)} sample frames...")
                
                except Exception as e:
                    self.logger.warning(f"Failed to analyze frame {frame_idx}: {e}")
                    frame_scores.append((frame_idx, 0.0))
            
            # Safety limit to prevent excessive processing
            if frame_count > 10000:
                self.logger.warning("Reached frame limit during analysis")
                break
        
        if not frame_scores:
            self.logger.warning("No frames analyzed, defaulting to poor quality")
            return {}
        
        # Apply smoothing and region detection
        region_map = self._detect_regions(frame_scores)
        
        self.logger.info(f"Region analysis complete: {len(region_map)} regions identified")
        return region_map
    
    def _detect_regions(self, frame_scores: List[Tuple[int, float]]) -> Dict[int, str]:
        """
        Detect feature-rich and feature-poor regions using sliding window analysis.
        """
        if not frame_scores:
            return {}
        
        region_map = {}
        window_size = 10  # Analyze regions in 10-frame windows
        
        # Smooth scores using moving average
        scores = [score for _, score in frame_scores]
        smoothed_scores = self._smooth_scores(scores, window=5)
        
        # Classify each sampled frame
        for i, (frame_idx, original_score) in enumerate(frame_scores):
            smoothed_score = smoothed_scores[i]
            
            # Determine region type based on smoothed score
            if smoothed_score >= self.config.feature_rich_threshold:
                region_type = "rich"
                self.stats['feature_rich_regions'] += 1
            else:
                region_type = "poor" 
                self.stats['feature_poor_regions'] += 1
            
            region_map[frame_idx] = region_type
        
        # Interpolate regions for all frames
        all_frames_map = self._interpolate_regions(region_map, max(frame_scores)[0] if frame_scores else 0)
        
        return all_frames_map
    
    def _smooth_scores(self, scores: List[float], window: int = 5) -> List[float]:
        """Apply moving average smoothing to scores."""
        if len(scores) < window:
            return scores
        
        smoothed = []
        for i in range(len(scores)):
            start_idx = max(0, i - window//2)
            end_idx = min(len(scores), i + window//2 + 1)
            avg_score = np.mean(scores[start_idx:end_idx])
            smoothed.append(avg_score)
        
        return smoothed
    
    def _interpolate_regions(self, region_map: Dict[int, str], max_frame: int) -> Dict[int, str]:
        """
        Interpolate region types for all frames based on analyzed samples.
        """
        if not region_map:
            return {}
        
        all_frames_map = {}
        analyzed_frames = sorted(region_map.keys())
        
        for frame_idx in range(max_frame + 1):
            # Find nearest analyzed frames
            region_type = "poor"  # Default
            
            for analyzed_frame in analyzed_frames:
                if analyzed_frame <= frame_idx:
                    region_type = region_map[analyzed_frame]
                else:
                    break
            
            all_frames_map[frame_idx] = region_type
        
        return all_frames_map
    
    def _extract_frames_by_regions(self, video_processor, region_map: Dict[int, str]) -> List[Tuple[int, np.ndarray, Dict[str, float], SfMMetrics]]:
        """
        Extract frames based on region analysis with dynamic rates.
        """
        selected_frames = []
        last_rich_extraction = -999
        last_poor_extraction = -999
        
        # Calculate frame intervals
        rich_interval = max(1, int(self.video_fps / self.config.feature_rich_fps))
        poor_interval = max(1, int(self.video_fps / self.config.feature_poor_fps))
        
        self.logger.info(f"Frame intervals - Rich: every {rich_interval} frames, Poor: every {poor_interval} frames")
        
        frame_count = 0
        for frame_idx, frame, quality_metrics in video_processor.extract_frames_streaming():
            self.stats['total_frames_processed'] += 1
            frame_count += 1
            
            # Determine region type for this frame
            region_type = region_map.get(frame_idx, "poor")
            
            # Check if we should extract this frame
            should_extract = False
            
            if region_type == "rich":
                if frame_idx - last_rich_extraction >= rich_interval:
                    should_extract = True
                    last_rich_extraction = frame_idx
                    self.stats['feature_rich_frames_selected'] += 1
            else:  # poor region
                if frame_idx - last_poor_extraction >= poor_interval:
                    should_extract = True
                    last_poor_extraction = frame_idx
                    self.stats['feature_poor_frames_selected'] += 1
            
            if should_extract:
                try:
                    # Compute full saliency analysis for selected frames
                    saliency_scores = self.saliency_analyzer.compute_comprehensive_saliency_score(
                        frame, quality_metrics
                    )
                    
                    # Assess SfM potential
                    sfm_metrics = self.saliency_analyzer.assess_sfm_potential(frame, saliency_scores)
                    
                    # Add region type to scores for tracking
                    saliency_scores['region_type'] = region_type
                    saliency_scores['extraction_reason'] = f"{region_type}_region"
                    
                    selected_frames.append((frame_idx, frame, saliency_scores, sfm_metrics))
                    
                    if len(selected_frames) % 10 == 0:
                        self.logger.info(f"Selected {len(selected_frames)} frames (current: {region_type} region)")
                    
                    # Check maximum frame limit
                    if self.config.max_frames and len(selected_frames) >= self.config.max_frames:
                        self.logger.info(f"Reached maximum frame limit: {self.config.max_frames}")
                        break
                
                except Exception as e:
                    self.logger.warning(f"Failed to process frame {frame_idx}: {e}")
            
            # Progress logging
            if frame_count % 100 == 0:
                self.logger.debug(f"Processed {frame_count} frames, selected {len(selected_frames)}")
            
            # Safety limit
            if frame_count > 20000:
                self.logger.warning("Reached safety limit during extraction")
                break
        
        return selected_frames
    
    def get_selection_stats(self) -> Dict[str, float]:
        """Get comprehensive selection statistics"""
        stats = self.stats.copy()
        
        if stats['total_frames_processed'] > 0:
            stats['overall_selection_rate'] = stats['frames_selected'] / stats['total_frames_processed']
            
            # Rich region stats
            if stats['feature_rich_regions'] > 0:
                stats['rich_region_efficiency'] = stats['feature_rich_frames_selected'] / stats['feature_rich_regions']
            
            # Poor region stats
            if stats['feature_poor_regions'] > 0:
                stats['poor_region_efficiency'] = stats['feature_poor_frames_selected'] / stats['feature_poor_regions']
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.saliency_analyzer.cleanup()
            self.logger.info(f"Dynamic sampler cleanup complete. Final stats: {self.get_selection_stats()}")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")