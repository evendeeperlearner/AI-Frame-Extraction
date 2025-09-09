#!/usr/bin/env python3
"""
Test script for dynamic frame extraction with variable rates.
Extracts 1 frame per 2 seconds in feature-poor areas and 4 fps in feature-rich areas.
"""

import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.core.config import ProcessingConfig
from src.pipeline.adaptive_extraction_pipeline import AdaptiveExtractionPipeline


def setup_logging():
    """Setup detailed logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dynamic_extraction.log')
        ]
    )
    
    # Set specific loggers to INFO for detailed output
    logging.getLogger('src.core.dynamic_sampler').setLevel(logging.INFO)
    logging.getLogger('src.core.saliency_analyzer').setLevel(logging.INFO)
    logging.getLogger('src.pipeline.adaptive_extraction_pipeline').setLevel(logging.INFO)


def main():
    """Test dynamic frame extraction"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Use the provided video path
    video_path = Path("/Users/jigarzonalva/Proyecto_GaussianSplating/IMG_0029.mp4")
    
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    
    try:
        # Create configuration for dynamic extraction
        config = ProcessingConfig(
            input_video_path=video_path,
            output_frames_dir=Path("results") / "dynamic_extraction",
            device="auto",  # Auto-detect best device (MPS > CUDA > CPU)
            
            # Dynamic extraction settings
            feature_rich_threshold=0.7,     # Threshold for feature-rich regions
            feature_rich_fps=4.0,           # 4 fps in feature-rich areas
            feature_poor_fps=0.5,           # 1 frame every 2 seconds (0.5 fps)
            
            # General settings
            saliency_threshold=0.6,         # Lower threshold for broader selection
            max_frames=1000,                # Reasonable limit
            target_resolution=(1920, 1080), # Keep original resolution
            batch_size=4,                   # Conservative for stability
            
            # Scoring weights optimized for detail detection
            scoring_weights={
                'spatial_complexity': 0.4,     # Emphasize spatial detail
                'semantic_richness': 0.3,      # Semantic diversity
                'geometric_information': 0.2,   # Geometric features
                'texture_complexity': 0.1       # Texture variation
            }
        )
        
        logger.info("="*80)
        logger.info("DYNAMIC FRAME EXTRACTION TEST")
        logger.info("="*80)
        logger.info(f"Video: {video_path.name}")
        logger.info(f"Output directory: {config.output_frames_dir}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Feature-rich threshold: {config.feature_rich_threshold}")
        logger.info(f"Feature-rich FPS: {config.feature_rich_fps}")
        logger.info(f"Feature-poor FPS: {config.feature_poor_fps}")
        logger.info("="*80)
        
        # Initialize and run pipeline
        start_time = time.time()
        pipeline = AdaptiveExtractionPipeline(config)
        results = pipeline.run_full_pipeline()
        total_time = time.time() - start_time
        
        # Print detailed results
        print("\n" + "="*80)
        print("DYNAMIC EXTRACTION COMPLETE")
        print("="*80)
        print(f"Video: {video_path.name}")
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
            
            print(f"\nExtraction Pattern Analysis:")
            print(f"Feature-rich frames selected: {len(rich_frames)}")
            print(f"Feature-poor frames selected: {len(poor_frames)}")
            print(f"Rich/Poor ratio: {len(rich_frames)/(len(poor_frames)+1e-6):.2f}")
            
            # Calculate actual extraction rates
            if results['selected_frames']:
                frame_times = [(f['original_frame_idx'] / results['video_info']['fps']) 
                              for f in results['selected_frames']]
                time_intervals = [frame_times[i+1] - frame_times[i] 
                                for i in range(len(frame_times)-1)]
                
                if time_intervals:
                    avg_interval = sum(time_intervals) / len(time_intervals)
                    actual_avg_fps = 1.0 / avg_interval if avg_interval > 0 else 0
                    print(f"Average time interval: {avg_interval:.2f} seconds")
                    print(f"Actual average FPS: {actual_avg_fps:.2f}")
        
        # Quality summary
        if 'quality_summary' in results:
            quality = results['quality_summary']
            print(f"\nQuality Assessment: {quality.get('quality_assessment', 'N/A')}")
            print(f"Average composite score: {quality.get('average_composite_score', 0):.3f}")
            print(f"Average SfM contribution: {quality.get('average_reconstruction_contribution', 0):.3f}")
        
        # Get dynamic sampler statistics
        if hasattr(pipeline.frame_sampler, 'get_selection_stats'):
            sampler_stats = pipeline.frame_sampler.get_selection_stats()
            print(f"\nDynamic Sampler Statistics:")
            print(f"Feature-rich regions: {sampler_stats.get('feature_rich_regions', 0)}")
            print(f"Feature-poor regions: {sampler_stats.get('feature_poor_regions', 0)}")
            print(f"Rich frames selected: {sampler_stats.get('feature_rich_frames_selected', 0)}")
            print(f"Poor frames selected: {sampler_stats.get('feature_poor_frames_selected', 0)}")
        
        print(f"\nOutput Files:")
        print(f"- Frames: {config.output_frames_dir}/")
        print(f"- Results: extraction_results.json")
        print(f"- Metadata: sfm_metadata.json")
        print(f"- Log: dynamic_extraction.log")
        
        logger.info("Dynamic extraction test completed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        print("Check the log file 'dynamic_extraction.log' for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()