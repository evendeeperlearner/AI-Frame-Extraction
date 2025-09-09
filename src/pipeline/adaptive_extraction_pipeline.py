"""
Main pipeline for adaptive frame extraction with comprehensive monitoring and SfM optimization.
Production-ready implementation with error handling and performance tracking.
"""

import logging
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
import psutil
import torch

from ..core.config import ProcessingConfig, QualityMetrics, SfMMetrics
from ..core.video_processor import VideoProcessor
from ..core.dynamic_sampler import DynamicFrameSampler
from ..core.exceptions import *


class AdaptiveExtractionPipeline:
    """
    Complete pipeline for adaptive frame extraction optimized for SfM reconstruction.
    Includes comprehensive monitoring, error recovery, and performance optimization.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.video_processor = VideoProcessor(config)
        self.frame_sampler = DynamicFrameSampler(config)
        
        # Performance and monitoring
        self.metrics = {
            'pipeline_start_time': None,
            'pipeline_end_time': None,
            'total_processing_time': 0.0,
            'video_analysis_time': 0.0,
            'frame_selection_time': 0.0,
            'frame_saving_time': 0.0,
            'memory_usage': {},
            'device_info': config.get_device_info(),
            'frame_statistics': {},
            'error_count': 0,
            'warnings_count': 0
        }
        
        # Results storage
        self.results = {
            'selected_frames': [],
            'video_info': {},
            'processing_config': config.__dict__.copy(),
            'performance_metrics': {},
            'quality_summary': {}
        }
        
        self.logger.info(f"Pipeline initialized with device: {self.metrics['device_info']}")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete adaptive extraction pipeline with comprehensive monitoring.
        
        Returns:
            Dictionary containing results, metrics, and metadata
        """
        self.metrics['pipeline_start_time'] = time.time()
        self.logger.info("=== Starting Adaptive Frame Extraction Pipeline ===")
        
        try:
            # Step 1: Video Analysis and Validation
            self._execute_video_analysis()
            
            # Step 2: Adaptive Frame Selection
            self._execute_frame_selection()
            
            # Step 3: Save Results and Generate Reports
            self._save_results_and_metadata()
            
            # Step 4: Generate Quality Summary
            self._generate_quality_summary()
            
            self.metrics['pipeline_end_time'] = time.time()
            self.metrics['total_processing_time'] = (
                self.metrics['pipeline_end_time'] - self.metrics['pipeline_start_time']
            )
            
            self.logger.info(
                f"=== Pipeline Complete ===\n"
                f"Selected: {len(self.results['selected_frames'])} frames\n"
                f"Total time: {self.metrics['total_processing_time']:.2f}s\n"
                f"Memory peak: {self.metrics['memory_usage'].get('peak_mb', 0):.1f}MB"
            )
            
            return self.results
        
        except Exception as e:
            self.metrics['error_count'] += 1
            self.logger.error(f"Pipeline execution failed: {e}")
            
            # Save partial results
            self._save_partial_results(str(e))
            
            raise AdaptiveExtractionError(f"Pipeline execution failed: {e}")
        
        finally:
            # Cleanup resources
            self._cleanup_resources()
    
    def _execute_video_analysis(self):
        """Execute video analysis and validation phase"""
        self.logger.info("Phase 1: Video Analysis and Validation")
        start_time = time.time()
        
        try:
            # Validate video file
            self.video_processor.validate_video_file()
            
            # Extract video metadata
            self.results['video_info'] = self.video_processor.get_video_info()
            
            # Log video information
            video_info = self.results['video_info']
            self.logger.info(
                f"Video: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.2f}fps\n"
                f"Duration: {video_info['duration']:.1f}s ({video_info['total_frames']} frames)\n"
                f"Size: {video_info['size_bytes'] / 1024**2:.1f}MB"
            )
            
            # Estimate processing requirements
            estimated_memory = self._estimate_memory_requirements(video_info)
            self.logger.info(f"Estimated memory requirement: {estimated_memory:.1f}MB")
            
            # Check system resources
            self._check_system_resources(estimated_memory)
            
        except Exception as e:
            raise VideoProcessingError(f"Video analysis failed: {e}")
        
        finally:
            self.metrics['video_analysis_time'] = time.time() - start_time
    
    def _execute_frame_selection(self):
        """Execute adaptive frame selection phase"""
        self.logger.info("Phase 2: Adaptive Frame Selection")
        start_time = time.time()
        
        try:
            # Monitor memory usage during processing
            self._start_memory_monitoring()
            
            # Execute dynamic frame selection with adaptive rates
            selected_frames = self.frame_sampler.select_frames_dynamic(self.video_processor)
            
            if not selected_frames:
                raise SaliencyAnalysisError("No frames were selected - check thresholds and input quality")
            
            self.results['selected_frames'] = []
            
            # Process and store selected frames
            for i, (frame_idx, frame, saliency_scores, sfm_metrics) in enumerate(selected_frames):
                
                # Save frame to disk
                frame_filename = f"frame_{i:06d}_idx_{frame_idx}.{self.config.frame_format}"
                frame_path = self.config.output_frames_dir / frame_filename
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(str(frame_path), frame_bgr)
                
                if not success:
                    self.logger.warning(f"Failed to save frame {i} to {frame_path}")
                    continue
                
                # Store frame metadata
                frame_metadata = {
                    'frame_index': i,
                    'original_frame_idx': frame_idx,
                    'file_path': str(frame_path),
                    'file_size_bytes': frame_path.stat().st_size if frame_path.exists() else 0,
                    'saliency_scores': saliency_scores,
                    'sfm_metrics': {
                        'geometric_information': sfm_metrics.geometric_information,
                        'viewpoint_uniqueness': sfm_metrics.viewpoint_uniqueness,
                        'tracking_potential': sfm_metrics.tracking_potential,
                        'bundle_adjustment_weight': sfm_metrics.bundle_adjustment_weight,
                        'reconstruction_contribution': sfm_metrics.reconstruction_contribution
                    },
                    'timestamp_seconds': frame_idx / self.results['video_info']['fps']
                }
                
                self.results['selected_frames'].append(frame_metadata)
            
            self.logger.info(f"Saved {len(self.results['selected_frames'])} frames to {self.config.output_frames_dir}")
            
        except Exception as e:
            raise SaliencyAnalysisError(f"Frame selection failed: {e}")
        
        finally:
            self.metrics['frame_selection_time'] = time.time() - start_time
            self._stop_memory_monitoring()
    
    def _save_results_and_metadata(self):
        """Save comprehensive results and metadata"""
        self.logger.info("Phase 3: Saving Results and Metadata")
        start_time = time.time()
        
        try:
            # Get comprehensive statistics
            selection_stats = self.frame_sampler.get_selection_stats()
            self.metrics['frame_statistics'] = selection_stats
            
            # Prepare complete results
            complete_results = {
                'pipeline_version': '1.0.0',
                'execution_timestamp': time.time(),
                'processing_config': {
                    k: str(v) if isinstance(v, Path) else v 
                    for k, v in self.config.__dict__.items()
                },
                'video_info': self.results['video_info'],
                'selected_frames': self.results['selected_frames'],
                'performance_metrics': self.metrics,
                'selection_statistics': selection_stats,
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                    'torch_version': torch.__version__,
                    'device_info': self.metrics['device_info']
                }
            }
            
            # Save main results file
            results_path = self.config.output_frames_dir / "extraction_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, indent=2, ensure_ascii=False, default=str)
            
            # Save frame list for easy import
            frame_list_path = self.config.output_frames_dir / "frame_list.txt"
            with open(frame_list_path, 'w') as f:
                for frame_meta in self.results['selected_frames']:
                    f.write(f"{frame_meta['file_path']}\n")
            
            # Save SfM-optimized metadata
            sfm_metadata_path = self.config.output_frames_dir / "sfm_metadata.json"
            sfm_data = {
                'frames': [
                    {
                        'path': frame['file_path'],
                        'timestamp': frame['timestamp_seconds'],
                        'reconstruction_contribution': frame['sfm_metrics']['reconstruction_contribution'],
                        'geometric_information': frame['sfm_metrics']['geometric_information']
                    }
                    for frame in self.results['selected_frames']
                ],
                'total_frames': len(self.results['selected_frames']),
                'video_duration': self.results['video_info']['duration'],
                'fps': self.results['video_info']['fps']
            }
            
            with open(sfm_metadata_path, 'w') as f:
                json.dump(sfm_data, f, indent=2)
            
            self.logger.info(f"Results saved to {self.config.output_frames_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise
        
        finally:
            self.metrics['frame_saving_time'] = time.time() - start_time
    
    def _generate_quality_summary(self):
        """Generate quality assessment summary"""
        try:
            if not self.results['selected_frames']:
                return
            
            # Aggregate saliency scores
            all_scores = [frame['saliency_scores'] for frame in self.results['selected_frames']]
            
            score_summary = {}
            for score_type in ['composite_score', 'spatial_complexity', 'semantic_richness', 
                              'geometric_information', 'texture_complexity']:
                scores = [s.get(score_type, 0) for s in all_scores]
                if scores:
                    score_summary[score_type] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores),
                        'median': np.median(scores)
                    }
            
            # Aggregate SfM metrics
            sfm_scores = [frame['sfm_metrics']['reconstruction_contribution'] 
                         for frame in self.results['selected_frames']]
            
            quality_summary = {
                'total_frames_selected': len(self.results['selected_frames']),
                'selection_rate': len(self.results['selected_frames']) / max(self.results['video_info']['total_frames'], 1),
                'average_composite_score': np.mean([s['composite_score'] for s in all_scores]),
                'average_reconstruction_contribution': np.mean(sfm_scores) if sfm_scores else 0,
                'score_distributions': score_summary,
                'quality_assessment': self._assess_overall_quality(score_summary, sfm_scores)
            }
            
            self.results['quality_summary'] = quality_summary
            
            self.logger.info(
                f"Quality Summary:\n"
                f"  Selection rate: {quality_summary['selection_rate']*100:.1f}%\n"
                f"  Avg composite score: {quality_summary['average_composite_score']:.3f}\n"
                f"  Avg SfM contribution: {quality_summary['average_reconstruction_contribution']:.3f}\n"
                f"  Overall assessment: {quality_summary['quality_assessment']}"
            )
            
        except Exception as e:
            self.logger.warning(f"Quality summary generation failed: {e}")
    
    def _assess_overall_quality(self, score_summary: Dict, sfm_scores: List[float]) -> str:
        """Assess overall quality of selected frames"""
        try:
            avg_composite = score_summary.get('composite_score', {}).get('mean', 0)
            avg_sfm = np.mean(sfm_scores) if sfm_scores else 0
            
            if avg_composite >= 0.8 and avg_sfm >= 0.7:
                return "Excellent - High quality frames with strong SfM potential"
            elif avg_composite >= 0.7 and avg_sfm >= 0.6:
                return "Good - Quality frames suitable for SfM reconstruction"
            elif avg_composite >= 0.6 and avg_sfm >= 0.5:
                return "Moderate - Acceptable quality, may need parameter tuning"
            else:
                return "Poor - Consider adjusting thresholds or checking input video quality"
        except:
            return "Unknown - Assessment failed"
    
    def _estimate_memory_requirements(self, video_info: Dict) -> float:
        """Estimate memory requirements in MB"""
        try:
            # Base memory for DINOv3 model
            base_memory = 2000  # MB
            
            # Frame processing memory
            frame_size_mb = (self.config.target_resolution[0] * self.config.target_resolution[1] * 3) / (1024**2)
            batch_memory = frame_size_mb * self.config.batch_size * 4  # 4x for processing overhead
            
            # Feature storage memory
            num_patches = (224 // 14) ** 2  # Approximate for DINOv3
            feature_memory = num_patches * 768 * 4 / (1024**2)  # 4 bytes per float32
            
            total_memory = base_memory + batch_memory + feature_memory
            
            return total_memory
        except:
            return 4000  # Conservative estimate
    
    def _check_system_resources(self, estimated_memory: float):
        """Check if system has sufficient resources"""
        try:
            available_memory = psutil.virtual_memory().available / (1024**2)  # MB
            
            if estimated_memory > available_memory * 0.8:
                self.logger.warning(
                    f"Estimated memory requirement ({estimated_memory:.0f}MB) exceeds "
                    f"80% of available memory ({available_memory:.0f}MB)"
                )
                
                # Suggest batch size reduction
                if self.config.batch_size > 2:
                    suggested_batch_size = max(1, self.config.batch_size // 2)
                    self.logger.info(f"Consider reducing batch_size to {suggested_batch_size}")
            
        except Exception as e:
            self.logger.warning(f"Resource check failed: {e}")
    
    def _start_memory_monitoring(self):
        """Start monitoring memory usage"""
        try:
            process = psutil.Process()
            self.metrics['memory_usage']['initial_mb'] = process.memory_info().rss / (1024**2)
            self.metrics['memory_usage']['peak_mb'] = self.metrics['memory_usage']['initial_mb']
        except:
            pass
    
    def _stop_memory_monitoring(self):
        """Stop monitoring memory usage and record peak"""
        try:
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024**2)
            self.metrics['memory_usage']['final_mb'] = current_memory
            self.metrics['memory_usage']['peak_mb'] = max(
                self.metrics['memory_usage'].get('peak_mb', 0),
                current_memory
            )
        except:
            pass
    
    def _save_partial_results(self, error_message: str):
        """Save partial results in case of failure"""
        try:
            partial_results = {
                'status': 'FAILED',
                'error_message': error_message,
                'partial_results': self.results,
                'metrics_at_failure': self.metrics,
                'timestamp': time.time()
            }
            
            error_path = self.config.output_frames_dir / "extraction_error.json"
            with open(error_path, 'w') as f:
                json.dump(partial_results, f, indent=2, default=str)
            
            self.logger.info(f"Partial results saved to {error_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save partial results: {e}")
    
    def _cleanup_resources(self):
        """Clean up pipeline resources"""
        try:
            # Cleanup frame sampler and its components
            if hasattr(self, 'frame_sampler'):
                self.frame_sampler.cleanup()
            
            # Clear any remaining GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Note: MPS doesn't have empty_cache() method
            
            self.logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        try:
            stats = {
                'pipeline_metrics': self.metrics,
                'selection_stats': self.frame_sampler.get_selection_stats() if hasattr(self, 'frame_sampler') else {},
                'quality_summary': self.results.get('quality_summary', {}),
                'system_resources': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'available_memory_gb': psutil.virtual_memory().available / (1024**3)
                }
            }
            
            return stats
        
        except Exception as e:
            self.logger.warning(f"Stats collection failed: {e}")
            return {'error': str(e)}