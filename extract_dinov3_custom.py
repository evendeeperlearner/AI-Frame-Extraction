#!/usr/bin/env python3
"""
DINOv3 Dynamic Frame Extraction Script
Extract frames from IMG_5200.MOV using DINOv3 with custom parameters.

Usage:
    python extract_dinov3_custom.py [OPTIONS]

Options:
    --model          Choose model: dinov3_small, dinov3_base, dinov2_base (default: dinov3_small)
    --rich-fps       FPS for feature-rich areas (default: 2.0)
    --poor-fps       FPS for feature-poor areas (default: 0.25)
    --threshold      Feature-rich threshold 0.0-1.0 (default: 0.7)
    --max-frames     Maximum frames to extract (default: 1000)
    --output-dir     Output directory name (default: auto-generated)
    --video          Video path (default: IMG_5200.MOV)
    --resolution     Target resolution WxH (default: 1920x1080)
    
Examples:
    # Basic usage with DINOv3 Small
    python extract_dinov3_custom.py
    
    # Use DINOv3 Base with custom rates
    python extract_dinov3_custom.py --model dinov3_base --rich-fps 3.0 --poor-fps 0.5
    
    # Higher quality settings
    python extract_dinov3_custom.py --threshold 0.8 --max-frames 1500 --resolution 2560x1440
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.core.config import ProcessingConfig, DINOModelType
from src.pipeline.adaptive_extraction_pipeline import AdaptiveExtractionPipeline


def setup_logging(log_file: str):
    """Setup detailed logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    # Set specific loggers to INFO for detailed output
    logging.getLogger('src.core.dynamic_sampler').setLevel(logging.INFO)
    logging.getLogger('src.core.saliency_analyzer').setLevel(logging.INFO)
    logging.getLogger('src.pipeline.adaptive_extraction_pipeline').setLevel(logging.INFO)


def parse_resolution(resolution_str: str) -> tuple:
    """Parse resolution string like '1920x1080' to (1920, 1080)"""
    try:
        width, height = resolution_str.split('x')
        return (int(width), int(height))
    except:
        raise ValueError(f"Invalid resolution format: {resolution_str}. Use format: WIDTHxHEIGHT")


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


def main():
    """Main extraction function"""
    parser = argparse.ArgumentParser(
        description="Extract frames dynamically using DINOv3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model selection
    parser.add_argument('--model', 
                       choices=['dinov2_base', 'dinov3_small', 'dinov3_base'],
                       default='dinov3_small',
                       help='Model to use for extraction (default: dinov3_small)')
    
    # Extraction parameters
    parser.add_argument('--rich-fps', type=float, default=2.0,
                       help='FPS for feature-rich areas (default: 2.0)')
    parser.add_argument('--poor-fps', type=float, default=0.25,
                       help='FPS for feature-poor areas (default: 0.25)')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Feature-rich threshold 0.0-1.0 (default: 0.7)')
    
    # Quality settings
    parser.add_argument('--max-frames', type=int, default=1000,
                       help='Maximum frames to extract (default: 1000)')
    parser.add_argument('--resolution', default='1920x1080',
                       help='Target resolution WxH (default: 1920x1080)')
    
    # I/O settings
    parser.add_argument('--video', 
                       default='/Users/jigarzonalva/Downloads/IMG_5200.MOV',
                       help='Input video path')
    parser.add_argument('--output-dir',
                       help='Output directory (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Validate inputs
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    try:
        target_resolution = parse_resolution(args.resolution)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    if not (0.0 <= args.threshold <= 1.0):
        print(f"❌ Error: Threshold must be between 0.0 and 1.0, got {args.threshold}")
        sys.exit(1)
    
    # Generate output directory name
    if args.output_dir:
        output_dir = Path("results") / args.output_dir
    else:
        model_short = args.model.replace('_', '')
        output_dir = Path("results") / f"dinov3_extract_{model_short}_{args.rich_fps}fps_{args.poor_fps}fps"
    
    # Generate log filename
    log_file = f"dinov3_extraction_{args.model}_{int(time.time())}.log"
    
    # Setup logging
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    # Print configuration
    print("🚀 DINOv3 DYNAMIC FRAME EXTRACTION")
    print("=" * 60)
    print(f"Video: {video_path.name}")
    print(f"Model: {args.model}")
    print(f"Feature-rich FPS: {args.rich_fps}")
    print(f"Feature-poor FPS: {args.poor_fps}")
    print(f"Feature threshold: {args.threshold}")
    print(f"Max frames: {args.max_frames}")
    print(f"Resolution: {args.resolution}")
    print(f"Output: {output_dir}")
    print(f"Log: {log_file}")
    print("=" * 60)
    
    try:
        # Create configuration
        config = ProcessingConfig(
            input_video_path=video_path,
            output_frames_dir=output_dir,
            device="auto",  # Auto-detect best device (MPS > CUDA > CPU)
            
            # Dynamic extraction settings - YOUR PARAMETERS
            feature_rich_threshold=args.threshold,
            feature_rich_fps=args.rich_fps,
            feature_poor_fps=args.poor_fps,
            
            # Quality settings
            saliency_threshold=0.6,         # Lower threshold for broader selection
            max_frames=args.max_frames,
            target_resolution=target_resolution,
            batch_size=4 if 'base' in args.model else 8,  # Optimize by model size
            
            # Scoring weights for visual detail detection
            scoring_weights={
                'spatial_complexity': 0.4,     # Emphasize spatial detail
                'semantic_richness': 0.3,      # Semantic diversity  
                'geometric_information': 0.2,   # Geometric features
                'texture_complexity': 0.1       # Texture variation
            }
        )
        
        # Set the selected model
        model_type = get_model_type(args.model)
        config.set_model(model_type)
        
        # Log detailed configuration
        model_info = config.get_model_info()
        logger.info(f"Model configuration: {model_info}")
        logger.info(f"Extraction rates: {args.rich_fps} fps (rich) / {args.poor_fps} fps (poor)")
        logger.info(f"Expected rich interval: {1/args.rich_fps:.2f}s")
        logger.info(f"Expected poor interval: {1/args.poor_fps:.2f}s")
        
        # Initialize and run pipeline
        start_time = time.time()
        pipeline = AdaptiveExtractionPipeline(config)
        results = pipeline.run_full_pipeline()
        total_time = time.time() - start_time
        
        # Print detailed results
        print("\n" + "=" * 80)
        print("✅ EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"Video: {video_path.name}")
        print(f"Model: {args.model} ({model_info['estimated_params']})")
        print(f"Duration: {results['video_info']['duration']:.1f} seconds")
        print(f"Total frames in video: {results['video_info']['total_frames']}")
        print(f"Video FPS: {results['video_info']['fps']:.2f}")
        print(f"Selected frames: {len(results['selected_frames'])}")
        print(f"Selection rate: {len(results['selected_frames'])/results['video_info']['total_frames']*100:.1f}%")
        print(f"Processing time: {total_time:.1f} seconds")
        
        # Analyze extraction patterns
        if results['selected_frames']:
            rich_frames = [f for f in results['selected_frames'] 
                          if f.get('saliency_scores', {}).get('region_type') == 'rich']
            poor_frames = [f for f in results['selected_frames'] 
                          if f.get('saliency_scores', {}).get('region_type') == 'poor']
            
            print(f"\n📊 Extraction Pattern Analysis:")
            print(f"Feature-rich frames: {len(rich_frames)}")
            print(f"Feature-poor frames: {len(poor_frames)}")
            print(f"Rich/Poor ratio: {len(rich_frames)/(len(poor_frames)+1e-6):.2f}")
            
            # Calculate actual extraction rates
            if len(results['selected_frames']) > 1:
                frame_times = [(f['original_frame_idx'] / results['video_info']['fps']) 
                              for f in results['selected_frames']]
                time_intervals = [frame_times[i+1] - frame_times[i] 
                                for i in range(len(frame_times)-1)]
                
                if time_intervals:
                    avg_interval = sum(time_intervals) / len(time_intervals)
                    actual_avg_fps = 1.0 / avg_interval if avg_interval > 0 else 0
                    print(f"Average time interval: {avg_interval:.2f} seconds")
                    print(f"Actual average FPS: {actual_avg_fps:.2f}")
                    
                    # Check rate accuracy
                    rich_intervals = []
                    poor_intervals = []
                    
                    for i, frame_meta in enumerate(results['selected_frames'][:-1]):
                        region_type = frame_meta.get('saliency_scores', {}).get('region_type', 'unknown')
                        if region_type == 'rich' and i < len(time_intervals):
                            rich_intervals.append(time_intervals[i])
                        elif region_type == 'poor' and i < len(time_intervals):
                            poor_intervals.append(time_intervals[i])
                    
                    if rich_intervals:
                        rich_avg_fps = 1.0 / (sum(rich_intervals) / len(rich_intervals))
                        accuracy = (rich_avg_fps / args.rich_fps) * 100
                        print(f"Rich regions - Actual: {rich_avg_fps:.2f} fps (target: {args.rich_fps}) - {accuracy:.1f}% accuracy")
                    
                    if poor_intervals:
                        poor_avg_fps = 1.0 / (sum(poor_intervals) / len(poor_intervals))
                        accuracy = (poor_avg_fps / args.poor_fps) * 100
                        print(f"Poor regions - Actual: {poor_avg_fps:.2f} fps (target: {args.poor_fps}) - {accuracy:.1f}% accuracy")
        
        # Quality summary
        if 'quality_summary' in results:
            quality = results['quality_summary']
            print(f"\n🎯 Quality Assessment: {quality.get('quality_assessment', 'N/A')}")
            print(f"Average composite score: {quality.get('average_composite_score', 0):.3f}")
            print(f"Average SfM contribution: {quality.get('average_reconstruction_contribution', 0):.3f}")
        
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
        print(f"• Log: {log_file}")
        
        logger.info(f"✅ Extraction completed successfully using {args.model}")
        print(f"\n🎉 SUCCESS! {len(results['selected_frames'])} frames extracted using {args.model.upper()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        print(f"Check the log file '{log_file}' for detailed error information.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)