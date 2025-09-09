#!/usr/bin/env python3
"""
Test DINOv3 dynamic extraction and compare with DINOv2 results.
Run this once you have DINOv3 model access.
"""

import logging
import sys
import time
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.core.config import ProcessingConfig, DINOModelType
from src.pipeline.adaptive_extraction_pipeline import AdaptiveExtractionPipeline


def setup_logging():
    """Setup detailed logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dinov3_comparison.log')
        ]
    )
    
    # Set specific loggers to INFO for detailed output
    logging.getLogger('src.core.dynamic_sampler').setLevel(logging.INFO)
    logging.getLogger('src.core.saliency_analyzer').setLevel(logging.INFO)
    logging.getLogger('src.pipeline.adaptive_extraction_pipeline').setLevel(logging.INFO)


def test_dynamic_extraction(model_type: DINOModelType, output_suffix: str):
    """Test dynamic extraction with specified model"""
    logger = logging.getLogger(__name__)
    
    model_info = {
        DINOModelType.DINOV2_BASE: "DINOv2 Base",
        DINOModelType.DINOV3_VITS16: "DINOv3 Small", 
        DINOModelType.DINOV3_VITB16: "DINOv3 Base"
    }
    
    test_name = model_info.get(model_type, str(model_type))
    
    logger.info("=" * 80)
    logger.info(f"TESTING DYNAMIC EXTRACTION: {test_name}")
    logger.info(f"Model: {model_type.value}")
    logger.info("=" * 80)
    
    try:
        # Create configuration
        config = ProcessingConfig(
            input_video_path=Path("/Users/jigarzonalva/Proyecto_GaussianSplating/IMG_0029.mp4"),
            output_frames_dir=Path("results") / f"dynamic_extraction_{output_suffix}",
            device="auto",
            
            # Dynamic extraction settings
            feature_rich_threshold=0.7,
            feature_rich_fps=4.0,
            feature_poor_fps=0.5,
            
            # Quality settings
            saliency_threshold=0.6,
            max_frames=1000,
            target_resolution=(1920, 1080),
            batch_size=4,  # Conservative for compatibility
            
            # Scoring weights
            scoring_weights={
                'spatial_complexity': 0.4,
                'semantic_richness': 0.3,
                'geometric_information': 0.2,
                'texture_complexity': 0.1
            }
        )
        
        # Set the model
        config.set_model(model_type)
        
        logger.info(f"Configuration: {config.get_model_info()}")
        
        # Run extraction
        start_time = time.time()
        pipeline = AdaptiveExtractionPipeline(config)
        results = pipeline.run_full_pipeline()
        total_time = time.time() - start_time
        
        # Analyze results
        logger.info("=" * 80)
        logger.info(f"{test_name} RESULTS")
        logger.info("=" * 80)
        logger.info(f"Processing time: {total_time:.1f} seconds")
        logger.info(f"Selected frames: {len(results['selected_frames'])}")
        logger.info(f"Selection rate: {len(results['selected_frames'])/results['video_info']['total_frames']*100:.1f}%")
        
        # Feature region analysis
        if results['selected_frames']:
            rich_frames = [f for f in results['selected_frames'] 
                          if f.get('saliency_scores', {}).get('region_type') == 'rich']
            poor_frames = [f for f in results['selected_frames'] 
                          if f.get('saliency_scores', {}).get('region_type') == 'poor']
            
            logger.info(f"Feature-rich frames: {len(rich_frames)}")
            logger.info(f"Feature-poor frames: {len(poor_frames)}")
            logger.info(f"Rich/Poor ratio: {len(rich_frames)/(len(poor_frames)+1e-6):.2f}")
        
        # Quality metrics
        if 'quality_summary' in results:
            quality = results['quality_summary']
            logger.info(f"Quality assessment: {quality.get('quality_assessment', 'N/A')}")
            logger.info(f"Average composite score: {quality.get('average_composite_score', 0):.3f}")
            logger.info(f"Average SfM contribution: {quality.get('average_reconstruction_contribution', 0):.3f}")
        
        return {
            'model': test_name,
            'processing_time': total_time,
            'frames_selected': len(results['selected_frames']),
            'selection_rate': len(results['selected_frames'])/results['video_info']['total_frames'],
            'rich_frames': len(rich_frames) if 'rich_frames' in locals() else 0,
            'poor_frames': len(poor_frames) if 'poor_frames' in locals() else 0,
            'quality': results.get('quality_summary', {}),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ {test_name} FAILED: {e}")
        
        if "gated repo" in str(e) or "403" in str(e):
            logger.error(f"   → Access denied. Request access at:")
            logger.error(f"     https://huggingface.co/{model_type.value}")
        
        return {
            'model': test_name,
            'error': str(e),
            'success': False
        }


def main():
    """Compare DINOv2 vs DINOv3 dynamic extraction"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check authentication
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        logger.info(f"✅ Authenticated as: {user_info['name']}")
    except:
        logger.error("❌ Not authenticated. Set HUGGINGFACE_HUB_TOKEN or run auth_hf.py")
        return False
    
    logger.info("🔬 DINOV2 vs DINOV3 DYNAMIC EXTRACTION COMPARISON")
    logger.info("Testing dynamic frame extraction with different DINO models")
    
    # Test models
    models_to_test = [
        (DINOModelType.DINOV2_BASE, "dinov2_base"),
        (DINOModelType.DINOV3_VITS16, "dinov3_small"),
        (DINOModelType.DINOV3_VITB16, "dinov3_base"),
    ]
    
    results = []
    
    for model_type, output_suffix in models_to_test:
        result = test_dynamic_extraction(model_type, output_suffix)
        results.append(result)
        
        # Brief pause between tests
        time.sleep(2)
    
    # Comparison Summary
    logger.info("\\n" + "=" * 80)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 80)
    
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) >= 2:
        logger.info("Model Performance Comparison:")
        for result in successful_results:
            logger.info(f"  {result['model']}:")
            logger.info(f"    - Processing time: {result['processing_time']:.1f}s")
            logger.info(f"    - Frames selected: {result['frames_selected']}")
            logger.info(f"    - Selection rate: {result['selection_rate']*100:.1f}%")
            logger.info(f"    - Rich/Poor ratio: {result['rich_frames']/(result['poor_frames']+1e-6):.2f}")
    else:
        logger.info("Not enough successful tests for comparison")
        for result in results:
            if result['success']:
                logger.info(f"✅ {result['model']} - SUCCESS")
            else:
                logger.info(f"❌ {result['model']} - FAILED: {result.get('error', 'Unknown')}")
    
    success_count = sum(1 for r in results if r['success'])
    logger.info(f"\\nOverall: {success_count}/{len(results)} models tested successfully")
    
    if success_count > 0:
        logger.info("🎉 At least one model working - system operational!")
    
    return success_count > 0


if __name__ == "__main__":
    # Set token from environment if available
    token = os.environ.get('HUGGINGFACE_HUB_TOKEN')
    if not token:
        print("Set HUGGINGFACE_HUB_TOKEN environment variable or run with:")
        print("export HUGGINGFACE_HUB_TOKEN='your_token_here' && python test_dinov3_dynamic_extraction.py")
    
    success = main()
    sys.exit(0 if success else 1)