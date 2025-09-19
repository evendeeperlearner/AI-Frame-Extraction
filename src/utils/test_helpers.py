#!/usr/bin/env python3
"""
Common test utilities for the panoramic video preparation toolkit.
Consolidates duplicate test helper functions.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from ..core.config import ProcessingConfig, DINOModelType
from ..core.feature_extractor import DINOFeatureExtractor


def validate_video_path(video_path: Path) -> bool:
    """
    Validate that a video file exists.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if video exists, False otherwise
    """
    return video_path.exists() and video_path.is_file()


def create_test_config(
    video_path: Path,
    output_dir: Path,
    model_type: DINOModelType = DINOModelType.DINOV2_BASE,
    **kwargs
) -> ProcessingConfig:
    """
    Create a standardized test configuration.
    
    Args:
        video_path: Input video path
        output_dir: Output directory
        model_type: DINO model to use
        **kwargs: Additional config parameters
        
    Returns:
        Configured ProcessingConfig instance
    """
    config = ProcessingConfig(
        input_video_path=video_path,
        output_frames_dir=output_dir,
        device="auto",
        batch_size=kwargs.get('batch_size', 2),
        **kwargs
    )
    
    config.set_model(model_type)
    return config


def test_model_loading(
    model_type: DINOModelType, 
    test_name: str,
    config_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Test loading and basic functionality of a DINO model.
    
    Args:
        model_type: Model type to test
        test_name: Human-readable test name
        config_kwargs: Additional configuration parameters
        
    Returns:
        Tuple of (success, results_dict)
    """
    if config_kwargs is None:
        config_kwargs = {}
    
    try:
        # Create minimal config for testing
        config = create_test_config(
            video_path=Path("/tmp/dummy.mp4"),  # Won't be used
            output_dir=Path("/tmp/test_output"),
            model_type=model_type,
            **config_kwargs
        )
        
        # Test model loading
        start_time = time.time()
        feature_extractor = DINOFeatureExtractor(config)
        load_time = time.time() - start_time
        
        # Get model info
        model_info = config.get_model_info()
        
        # Test basic functionality
        stats = feature_extractor.get_processing_stats()
        
        # Cleanup
        feature_extractor.cleanup()
        
        results = {
            'model': test_name,
            'model_type': model_type.value,
            'success': True,
            'load_time': load_time,
            'model_info': model_info,
            'stats': stats
        }
        
        return True, results
        
    except Exception as e:
        results = {
            'model': test_name,
            'model_type': model_type.value,
            'success': False,
            'error': str(e)
        }
        return False, results


def analyze_extraction_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze extraction results and compute statistics.
    
    Args:
        results: Pipeline extraction results
        
    Returns:
        Analysis dictionary with computed statistics
    """
    if not results.get('selected_frames'):
        return {'error': 'No frames selected'}
    
    selected_frames = results['selected_frames']
    video_info = results['video_info']
    
    # Basic statistics
    total_frames = video_info['total_frames']
    selected_count = len(selected_frames)
    selection_rate = (selected_count / total_frames) * 100
    
    # Analyze frame regions
    rich_frames = [f for f in selected_frames 
                   if f.get('saliency_scores', {}).get('region_type') == 'rich']
    poor_frames = [f for f in selected_frames 
                   if f.get('saliency_scores', {}).get('region_type') == 'poor']
    
    # Calculate time intervals and FPS
    frame_times = [(f['original_frame_idx'] / video_info['fps']) 
                   for f in selected_frames]
    time_intervals = [frame_times[i+1] - frame_times[i] 
                     for i in range(len(frame_times)-1)]
    
    avg_interval = sum(time_intervals) / len(time_intervals) if time_intervals else 0
    actual_avg_fps = 1.0 / avg_interval if avg_interval > 0 else 0
    
    # Analyze rich vs poor intervals
    rich_intervals = []
    poor_intervals = []
    
    for i, frame_meta in enumerate(selected_frames[:-1]):
        region_type = frame_meta.get('saliency_scores', {}).get('region_type', 'unknown')
        if region_type == 'rich' and i < len(time_intervals):
            rich_intervals.append(time_intervals[i])
        elif region_type == 'poor' and i < len(time_intervals):
            poor_intervals.append(time_intervals[i])
    
    analysis = {
        'total_frames': total_frames,
        'selected_frames': selected_count,
        'selection_rate': selection_rate,
        'rich_frames': len(rich_frames),
        'poor_frames': len(poor_frames),
        'rich_poor_ratio': len(rich_frames) / (len(poor_frames) + 1e-6),
        'avg_time_interval': avg_interval,
        'actual_avg_fps': actual_avg_fps
    }
    
    # Add rich/poor specific FPS if available
    if rich_intervals:
        rich_avg_fps = 1.0 / (sum(rich_intervals) / len(rich_intervals))
        analysis['rich_actual_fps'] = rich_avg_fps
    
    if poor_intervals:
        poor_avg_fps = 1.0 / (sum(poor_intervals) / len(poor_intervals))
        analysis['poor_actual_fps'] = poor_avg_fps
    
    return analysis


def print_extraction_summary(
    results: Dict[str, Any], 
    analysis: Dict[str, Any],
    video_path: Path,
    processing_time: float,
    model_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Print a standardized extraction results summary.
    
    Args:
        results: Pipeline extraction results
        analysis: Results analysis from analyze_extraction_results()
        video_path: Path to processed video
        processing_time: Total processing time
        model_info: Optional model information
    """
    print("\n" + "=" * 80)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Video: {video_path.name}")
    
    if model_info:
        print(f"Model: {model_info.get('model_version', 'Unknown')} ({model_info.get('estimated_params', 'Unknown')})")
    
    video_info = results['video_info']
    print(f"Duration: {video_info['duration']:.1f} seconds")
    print(f"Total frames in video: {analysis['total_frames']}")
    print(f"Video FPS: {video_info['fps']:.2f}")
    print(f"Selected frames: {analysis['selected_frames']}")
    print(f"Selection rate: {analysis['selection_rate']:.1f}%")
    print(f"Processing time: {processing_time:.1f} seconds")
    
    print(f"\n📊 Extraction Pattern Analysis:")
    print(f"Feature-rich frames: {analysis['rich_frames']}")
    print(f"Feature-poor frames: {analysis['poor_frames']}")
    print(f"Rich/Poor ratio: {analysis['rich_poor_ratio']:.2f}")
    print(f"Average time interval: {analysis['avg_time_interval']:.2f} seconds")
    print(f"Actual average FPS: {analysis['actual_avg_fps']:.2f}")
    
    # Rich/Poor specific FPS
    if 'rich_actual_fps' in analysis:
        print(f"Rich regions - Actual FPS: {analysis['rich_actual_fps']:.2f}")
    if 'poor_actual_fps' in analysis:
        print(f"Poor regions - Actual FPS: {analysis['poor_actual_fps']:.2f}")
    
    # Quality summary
    if 'quality_summary' in results:
        quality = results['quality_summary']
        print(f"\n🎯 Quality Assessment: {quality.get('quality_assessment', 'N/A')}")
        print(f"Average composite score: {quality.get('average_composite_score', 0):.3f}")
        print(f"Average SfM contribution: {quality.get('average_reconstruction_contribution', 0):.3f}")


def run_model_comparison_test(
    models_to_test: List[Tuple[DINOModelType, str]],
    logger
) -> List[Dict[str, Any]]:
    """
    Run a comparison test across multiple models.
    
    Args:
        models_to_test: List of (model_type, test_name) tuples
        logger: Logger instance
        
    Returns:
        List of test results
    """
    results = []
    
    for model_type, test_name in models_to_test:
        logger.info(f"Testing {test_name}")
        success, result = test_model_loading(model_type, test_name)
        results.append(result)
        
        if success:
            logger.info(f"✅ {test_name} - SUCCESS ({result['load_time']:.2f}s)")
        else:
            logger.error(f"❌ {test_name} - FAILED: {result['error']}")
        
        # Brief pause between tests
        time.sleep(1)
    
    return results