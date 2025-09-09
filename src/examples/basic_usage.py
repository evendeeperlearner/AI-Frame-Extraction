#!/usr/bin/env python3
"""
Basic usage example for adaptive frame extraction.
Demonstrates simple pipeline execution with default settings.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.config import ProcessingConfig
from src.pipeline.adaptive_extraction_pipeline import AdaptiveExtractionPipeline


def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('adaptive_extraction.log')
        ]
    )


def main():
    """Basic usage demonstration"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check if video path is provided
    if len(sys.argv) < 2:
        logger.error("Usage: python basic_usage.py <video_path>")
        logger.info("Example: python basic_usage.py /path/to/video.mp4")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    
    try:
        # Create configuration with default settings
        config = ProcessingConfig(
            input_video_path=video_path,
            output_frames_dir=Path("results") / "extracted_frames",
            device="auto",  # Auto-detect best device (MPS > CUDA > CPU)
            saliency_threshold=0.75,  # Standard threshold for high-quality frames
            max_frames=500,  # Reasonable limit for testing
            min_temporal_spacing=30,  # 1 second at 30fps
            target_resolution=(1920, 1080)  # Full HD
        )
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Output directory: {config.output_frames_dir}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Saliency threshold: {config.saliency_threshold}")
        
        # Initialize and run pipeline
        pipeline = AdaptiveExtractionPipeline(config)
        results = pipeline.run_full_pipeline()
        
        # Print results summary
        print("\n" + "="*60)
        print("EXTRACTION COMPLETE")
        print("="*60)
        print(f"Video: {video_path.name}")
        print(f"Duration: {results['video_info']['duration']:.1f} seconds")
        print(f"Total frames: {results['video_info']['total_frames']}")
        print(f"Selected frames: {len(results['selected_frames'])}")
        print(f"Selection rate: {len(results['selected_frames'])/results['video_info']['total_frames']*100:.1f}%")
        print(f"Processing time: {results['performance_metrics']['total_processing_time']:.1f} seconds")
        print(f"Output directory: {config.output_frames_dir}")
        
        # Quality summary
        if 'quality_summary' in results:
            quality = results['quality_summary']
            print(f"\nQuality Assessment: {quality.get('quality_assessment', 'N/A')}")
            print(f"Average composite score: {quality.get('average_composite_score', 0):.3f}")
        
        print("\nFiles created:")
        print(f"- extraction_results.json: Complete results and metadata")
        print(f"- frame_list.txt: List of extracted frame paths")
        print(f"- sfm_metadata.json: SfM-optimized metadata")
        print(f"- {len(results['selected_frames'])} frame images")
        
        logger.info("Basic usage example completed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        print(f"\nERROR: {e}")
        print("Check the log file 'adaptive_extraction.log' for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()