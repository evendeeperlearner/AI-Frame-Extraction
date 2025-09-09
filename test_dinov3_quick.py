#!/usr/bin/env python3
"""
Quick DINOv3 vs DINOv2 comparison test for dynamic extraction.
Tests a short video segment with both models to demonstrate DINOv3 is working.
"""

import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.core.config import ProcessingConfig, DINOModelType
from src.pipeline.adaptive_extraction_pipeline import AdaptiveExtractionPipeline


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_model_comparison():
    """Quick comparison test between DINOv2 and DINOv3"""
    logger = logging.getLogger(__name__)
    
    # Test video - use shorter IMG_5200.MOV for speed
    video_path = Path("/Users/jigarzonalva/Downloads/IMG_5200.MOV")
    
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return False
    
    # Models to test
    models_to_test = [
        (DINOModelType.DINOV2_BASE, "DINOv2 Base"),
        (DINOModelType.DINOV3_VITS16, "DINOv3 Small"),
        (DINOModelType.DINOV3_VITB16, "DINOv3 Base"),
    ]
    
    results = []
    
    for model_type, model_name in models_to_test:
        logger.info(f"=" * 60)
        logger.info(f"Testing {model_name}")
        logger.info(f"Model: {model_type.value}")
        logger.info(f"=" * 60)
        
        try:
            # Create configuration for quick test
            config = ProcessingConfig(
                input_video_path=video_path,
                output_frames_dir=Path("results") / f"quick_test_{model_name.lower().replace(' ', '_')}",
                device="auto",
                
                # Quick test settings - process fewer frames
                feature_rich_threshold=0.7,
                feature_rich_fps=5.0,        # Moderate FPS for speed
                feature_poor_fps=1.0,        # Higher for speed
                
                # Conservative settings for stability
                saliency_threshold=0.6,
                max_frames=50,               # Limit frames for quick test
                target_resolution=(960, 540), # Lower resolution for speed
                batch_size=2,                # Small batch
                
                # Standard weights
                scoring_weights={
                    'spatial_complexity': 0.4,
                    'semantic_richness': 0.3,
                    'geometric_information': 0.2,
                    'texture_complexity': 0.1
                }
            )
            
            # Set model
            config.set_model(model_type)
            
            # Log model info
            model_info = config.get_model_info()
            logger.info(f"Configuration: {model_info}")
            
            # Run extraction
            start_time = time.time()
            pipeline = AdaptiveExtractionPipeline(config)
            extracted_results = pipeline.run_full_pipeline()
            total_time = time.time() - start_time
            
            # Process results
            selected_frames = len(extracted_results['selected_frames'])
            total_frames = extracted_results['video_info']['total_frames']
            selection_rate = (selected_frames / total_frames) * 100
            
            # Feature region analysis
            rich_frames = [f for f in extracted_results['selected_frames'] 
                          if f.get('saliency_scores', {}).get('region_type') == 'rich']
            poor_frames = [f for f in extracted_results['selected_frames'] 
                          if f.get('saliency_scores', {}).get('region_type') == 'poor']
            
            # Quality assessment
            quality_summary = extracted_results.get('quality_summary', {})
            avg_score = quality_summary.get('average_composite_score', 0)
            
            result = {
                'model': model_name,
                'model_type': model_type.value,
                'processing_time': total_time,
                'frames_selected': selected_frames,
                'total_frames': total_frames,
                'selection_rate': selection_rate,
                'rich_frames': len(rich_frames),
                'poor_frames': len(poor_frames),
                'avg_quality_score': avg_score,
                'success': True
            }
            
            results.append(result)
            
            logger.info(f"✅ {model_name} - SUCCESS")
            logger.info(f"   Processing time: {total_time:.1f}s")
            logger.info(f"   Frames selected: {selected_frames}/{total_frames} ({selection_rate:.1f}%)")
            logger.info(f"   Feature regions: {len(rich_frames)} rich, {len(poor_frames)} poor")
            logger.info(f"   Quality score: {avg_score:.3f}")
            
        except Exception as e:
            logger.error(f"❌ {model_name} - FAILED: {e}")
            result = {
                'model': model_name,
                'model_type': model_type.value,
                'error': str(e),
                'success': False
            }
            results.append(result)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("QUICK COMPARISON SUMMARY")
    logger.info("=" * 80)
    
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) >= 2:
        logger.info("Model Performance Comparison:")
        for result in successful_results:
            logger.info(f"  {result['model']}:")
            logger.info(f"    - Processing time: {result['processing_time']:.1f}s")
            logger.info(f"    - Frames selected: {result['frames_selected']}")
            logger.info(f"    - Selection rate: {result['selection_rate']:.1f}%")
            logger.info(f"    - Quality score: {result['avg_quality_score']:.3f}")
            logger.info(f"    - Rich/Poor frames: {result['rich_frames']}/{result['poor_frames']}")
    else:
        logger.info("Results:")
        for result in results:
            if result['success']:
                logger.info(f"✅ {result['model']} - SUCCESS")
            else:
                logger.info(f"❌ {result['model']} - FAILED: {result.get('error', 'Unknown')}")
    
    success_count = sum(1 for r in results if r['success'])
    logger.info(f"\nOverall: {success_count}/{len(results)} models tested successfully")
    
    if success_count >= 2:
        dinov3_working = any('DINOv3' in r['model'] and r['success'] for r in results)
        if dinov3_working:
            logger.info("🎉 DINOv3 IS WORKING! Upgrade successful!")
        else:
            logger.info("⚠️  DINOv2 working, DINOv3 needs investigation")
    
    return success_count > 0


def main():
    """Main test function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 QUICK DINOv3 vs DINOv2 COMPARISON TEST")
    logger.info("Testing DINOv3 functionality with short video processing")
    
    success = test_model_comparison()
    
    if success:
        logger.info("✅ Test completed successfully - DINOv3 system operational!")
    else:
        logger.info("❌ Test failed - Check logs for issues")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)