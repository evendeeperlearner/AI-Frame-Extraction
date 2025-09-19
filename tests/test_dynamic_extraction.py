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
from src.utils.logging_setup import setup_extraction_logging
from src.utils.test_helpers import validate_video_path, analyze_extraction_results, print_extraction_summary


def main():
    """Test dynamic frame extraction"""
    logger = setup_extraction_logging('dynamic_extraction')
    
    # Use the provided video path
    video_path = Path("/Users/jigarzonalva/Proyecto_GaussianSplating/IMG_0029.mp4")
    
    if not validate_video_path(video_path):
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
        
        # Analyze and print results using centralized utilities
        analysis = analyze_extraction_results(results)
        model_info = {'model_version': 'DINOv2', 'estimated_params': '~86M parameters'}
        print_extraction_summary(results, analysis, video_path, total_time, model_info)
        
        # Get dynamic sampler statistics
        if hasattr(pipeline.frame_sampler, 'get_selection_stats'):
            sampler_stats = pipeline.frame_sampler.get_selection_stats()
            print(f"\nDynamic Sampler Statistics:")
            print(f"Feature-rich regions: {sampler_stats.get('feature_rich_regions', 0)}")
            print(f"Feature-poor regions: {sampler_stats.get('feature_poor_regions', 0)}")
            print(f"Rich frames selected: {sampler_stats.get('feature_rich_frames_selected', 0)}")
            print(f"Poor frames selected: {sampler_stats.get('feature_poor_frames_selected', 0)}")
        
        print(f"\n📁 Output Files:")
        print(f"• Frames: {config.output_frames_dir}/")
        print(f"• Results: extraction_results.json")
        print(f"• Metadata: sfm_metadata.json")
        print(f"• Log: dynamic_extraction_extraction.log")
        
        logger.info("Dynamic extraction test completed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        print("Check the log file 'dynamic_extraction_extraction.log' for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()