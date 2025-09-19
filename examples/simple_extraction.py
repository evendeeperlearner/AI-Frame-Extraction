#!/usr/bin/env python3
"""
Simple extraction example - minimal setup for basic frame extraction.
Perfect for getting started with the panoramic video preparation toolkit.

Usage:
    python simple_extraction.py video.mp4
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import ProcessingConfig, DINOModelType
from src.pipeline.adaptive_extraction_pipeline import AdaptiveExtractionPipeline
from src.utils.logging_setup import setup_logging


def simple_extract(video_path: Path, output_dir: Path = None):
    """
    Simple frame extraction with sensible defaults.
    
    Args:
        video_path: Path to input video
        output_dir: Optional output directory (auto-generated if None)
    """
    # Setup minimal logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Validate video
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path("extracted_frames") / video_path.stem
    
    # Create simple configuration
    config = ProcessingConfig(
        input_video_path=video_path,
        output_frames_dir=output_dir,
        device="auto",  # Auto-detect best device
        
        # Simple dynamic extraction - balanced settings
        feature_rich_threshold=0.7,
        feature_rich_fps=3.0,      # 3 fps in detailed areas
        feature_poor_fps=0.5,      # 1 frame every 2 seconds in simple areas
        
        # Conservative settings for reliability
        saliency_threshold=0.6,
        max_frames=500,            # Reasonable limit
        target_resolution=(1920, 1080),  # Full HD
        batch_size=4
    )
    
    # Use DINOv2 by default (most compatible)
    config.set_model(DINOModelType.DINOV2_BASE)
    
    logger.info(f"Extracting frames from: {video_path.name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Run extraction
    pipeline = AdaptiveExtractionPipeline(config)
    results = pipeline.run_full_pipeline()
    
    # Simple results summary
    selected_frames = len(results['selected_frames'])
    total_frames = results['video_info']['total_frames']
    duration = results['video_info']['duration']
    
    print(f"\\n✅ Extraction completed!")
    print(f"   Video: {video_path.name} ({duration:.1f}s)")
    print(f"   Frames extracted: {selected_frames}/{total_frames}")
    print(f"   Selection rate: {(selected_frames/total_frames)*100:.1f}%")
    print(f"   Output: {output_dir}/")
    
    return results


def main():
    """Command line interface"""
    if len(sys.argv) != 2:
        print("Usage: python simple_extraction.py <video_path>")
        print("Example: python simple_extraction.py video.mp4")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    
    try:
        simple_extract(video_path)
        print("\\n🎉 Success! Check the 'extracted_frames' directory for your frames.")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()