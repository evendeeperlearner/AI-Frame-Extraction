#!/usr/bin/env python3
"""
Complete pipeline test script for dynamic frame extraction.
Consolidates multiple test scenarios with configurable parameters.

Usage:
    python test_complete_pipeline.py [OPTIONS]

Examples:
    # Test with default video (IMG_0029.mp4)
    python test_complete_pipeline.py
    
    # Test with custom video and parameters
    python test_complete_pipeline.py --video /path/to/video.mp4 --rich-fps 4.0 --poor-fps 0.5
    
    # Test with specific model
    python test_complete_pipeline.py --model dinov3_small --video IMG_5200.MOV
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.core.config import ProcessingConfig, DINOModelType
from src.pipeline.adaptive_extraction_pipeline import AdaptiveExtractionPipeline
from src.utils.logging_setup import setup_extraction_logging
from src.utils.test_helpers import (
    validate_video_path, create_test_config, analyze_extraction_results,
    print_extraction_summary
)


def get_model_type(model_name: str) -> DINOModelType:
    """Convert model name string to DINOModelType enum"""
    model_map = {
        'dinov2_base': DINOModelType.DINOV2_BASE,
        'dinov3_small': DINOModelType.DINOV3_VITS16,
        'dinov3_base': DINOModelType.DINOV3_VITB16,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(model_map.keys())}")
    
    return model_map[model_name]


def parse_resolution(resolution_str: str) -> tuple:
    """Parse resolution string like '1920x1080' to (1920, 1080)"""
    try:
        width, height = resolution_str.split('x')
        return (int(width), int(height))
    except:
        raise ValueError(f"Invalid resolution format: {resolution_str}. Use format: WIDTHxHEIGHT")


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(
        description="Complete pipeline test for dynamic frame extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Video and model selection
    parser.add_argument('--video', 
                       help='Input video path (default: IMG_0029.mp4 or IMG_5200.MOV)')
    parser.add_argument('--model', 
                       choices=['dinov2_base', 'dinov3_small', 'dinov3_base'],
                       default='dinov2_base',
                       help='Model to use for extraction (default: dinov2_base)')
    
    # Dynamic extraction parameters
    parser.add_argument('--rich-fps', type=float, default=4.0,
                       help='FPS for feature-rich areas (default: 4.0)')
    parser.add_argument('--poor-fps', type=float, default=0.5,
                       help='FPS for feature-poor areas (default: 0.5)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Feature-rich threshold 0.0-1.0 (default: 0.7)')
    
    # Quality settings
    parser.add_argument('--max-frames', type=int, default=1000,
                       help='Maximum frames to extract (default: 1000)')
    parser.add_argument('--resolution', default='1920x1080',
                       help='Target resolution WxH (default: 1920x1080)')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size (auto-detected based on model if not specified)')
    
    # Output settings
    parser.add_argument('--output-dir',
                       help='Output directory (default: auto-generated)')
    parser.add_argument('--log-level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_extraction_logging('pipeline_test')
    
    # Determine video path
    if args.video:
        video_path = Path(args.video)
    else:
        # Try default videos in order of preference
        default_videos = [
            Path("/Users/jigarzonalva/Proyecto_GaussianSplating/IMG_0029.mp4"),
            Path("/Users/jigarzonalva/Downloads/IMG_5200.MOV")
        ]
        
        video_path = None
        for vid_path in default_videos:
            if validate_video_path(vid_path):
                video_path = vid_path
                break
        
        if not video_path:
            logger.error("No valid video found. Please specify --video path")
            print("\nAvailable options:")
            print("  --video /path/to/your/video.mp4")
            print("  --video /Users/jigarzonalva/Proyecto_GaussianSplating/IMG_0029.mp4")
            print("  --video /Users/jigarzonalva/Downloads/IMG_5200.MOV")
            sys.exit(1)
    
    if not validate_video_path(video_path):
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    
    # Parse resolution
    try:
        target_resolution = parse_resolution(args.resolution)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Validate threshold
    if not (0.0 <= args.threshold <= 1.0):
        logger.error(f"Threshold must be between 0.0 and 1.0, got {args.threshold}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path("results") / args.output_dir
    else:
        output_dir = Path("results") / f"pipeline_test_{args.model}_{args.rich_fps}fps_{args.poor_fps}fps"
    
    # Get model type
    try:
        model_type = get_model_type(args.model)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Determine batch size based on model if not specified
    if args.batch_size is None:
        batch_size = 8 if 'small' in args.model else 4
    else:
        batch_size = args.batch_size
    
    # Print test configuration
    print("🚀 COMPLETE PIPELINE TEST")
    print("=" * 60)
    print(f"Video: {video_path.name}")
    print(f"Model: {args.model}")
    print(f"Feature-rich FPS: {args.rich_fps}")
    print(f"Feature-poor FPS: {args.poor_fps}")
    print(f"Feature threshold: {args.threshold}")
    print(f"Max frames: {args.max_frames}")
    print(f"Resolution: {args.resolution}")
    print(f"Batch size: {batch_size}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    try:
        # Create configuration
        config = ProcessingConfig(
            input_video_path=video_path,
            output_frames_dir=output_dir,
            device="auto",
            
            # Dynamic extraction settings
            feature_rich_threshold=args.threshold,
            feature_rich_fps=args.rich_fps,
            feature_poor_fps=args.poor_fps,
            
            # Quality settings
            saliency_threshold=0.6,
            max_frames=args.max_frames,
            target_resolution=target_resolution,
            batch_size=batch_size,
            
            # Scoring weights optimized for detail detection
            scoring_weights={
                'spatial_complexity': 0.4,
                'semantic_richness': 0.3,
                'geometric_information': 0.2,
                'texture_complexity': 0.1
            }
        )
        
        # Set model
        config.set_model(model_type)
        model_info = config.get_model_info()
        
        logger.info(f"Configuration: {model_info}")
        logger.info(f"Expected intervals: Rich={1/args.rich_fps:.2f}s, Poor={1/args.poor_fps:.2f}s")
        
        # Run pipeline
        start_time = time.time()
        pipeline = AdaptiveExtractionPipeline(config)
        results = pipeline.run_full_pipeline()
        total_time = time.time() - start_time
        
        # Analyze results
        analysis = analyze_extraction_results(results)
        
        # Print results
        print_extraction_summary(results, analysis, video_path, total_time, model_info)
        
        # Rate accuracy check
        if 'rich_actual_fps' in analysis and analysis['rich_frames'] > 0:
            rich_accuracy = (analysis['rich_actual_fps'] / args.rich_fps) * 100
            print(f"Rich regions accuracy: {rich_accuracy:.1f}% (actual: {analysis['rich_actual_fps']:.2f} fps)")
        
        if 'poor_actual_fps' in analysis and analysis['poor_frames'] > 0:
            poor_accuracy = (analysis['poor_actual_fps'] / args.poor_fps) * 100
            print(f"Poor regions accuracy: {poor_accuracy:.1f}% (actual: {analysis['poor_actual_fps']:.2f} fps)")
        
        # Dynamic sampler statistics
        if hasattr(pipeline.frame_sampler, 'get_selection_stats'):
            sampler_stats = pipeline.frame_sampler.get_selection_stats()
            print(f"\n🔍 Dynamic Sampler Statistics:")
            print(f"Feature-rich regions detected: {sampler_stats.get('feature_rich_regions', 0)}")
            print(f"Feature-poor regions detected: {sampler_stats.get('feature_poor_regions', 0)}")
        
        print(f"\n📁 Output Files:")
        print(f"• Frames: {output_dir}/")
        print(f"• Results: extraction_results.json")
        print(f"• Metadata: sfm_metadata.json")
        print(f"• Log: pipeline_test_extraction.log")
        
        logger.info(f"✅ Pipeline test completed successfully using {args.model}")
        print(f"\n🎉 SUCCESS! {analysis['selected_frames']} frames extracted using {args.model.upper()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        print("Check the log file 'pipeline_test_extraction.log' for detailed error information.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)