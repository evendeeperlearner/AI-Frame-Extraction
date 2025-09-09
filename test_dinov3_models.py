#!/usr/bin/env python3
"""
Test script to verify DINOv2 and DINOv3 model compatibility.
Tests model loading and basic feature extraction for both versions.
"""

import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.core.config import ProcessingConfig, DINOModelType
from src.core.feature_extractor import DINOFeatureExtractor


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_model_loading(model_type: DINOModelType, test_name: str):
    """Test loading and basic functionality of a DINO model"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"=" * 60)
    logger.info(f"TESTING: {test_name}")
    logger.info(f"Model: {model_type.value}")
    logger.info(f"=" * 60)
    
    try:
        # Create configuration
        config = ProcessingConfig(
            input_video_path=Path("/Users/jigarzonalva/Proyecto_GaussianSplating/IMG_0029.mp4"),
            output_frames_dir=Path("test_output"),
            device="auto",
            batch_size=2  # Small batch for testing
        )
        
        # Set the model
        config.set_model(model_type)
        
        # Log model info
        model_info = config.get_model_info()
        logger.info(f"Model Info: {model_info}")
        
        # Test feature extractor initialization
        start_time = time.time()
        feature_extractor = DINOFeatureExtractor(config)
        load_time = time.time() - start_time
        
        logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
        
        # Test basic functionality (without actual frame processing)
        stats = feature_extractor.get_processing_stats()
        logger.info(f"✅ Basic functionality verified")
        logger.info(f"Processing stats: {stats}")
        
        # Cleanup
        feature_extractor.cleanup()
        logger.info(f"✅ {test_name} - ALL TESTS PASSED")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ {test_name} FAILED: {e}")
        return False


def main():
    """Test both DINOv2 and DINOv3 models"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 DINO MODEL COMPATIBILITY TEST")
    logger.info("Testing both DINOv2 and DINOv3 model loading and basic functionality")
    
    test_results = {}
    
    # Test models in order of increasing size/complexity
    models_to_test = [
        (DINOModelType.DINOV2_BASE, "DINOv2 Base (Current Working Model)"),
        (DINOModelType.DINOV3_VITS16, "DINOv3 Small (21M params)"),  # Start with smallest DINOv3
        (DINOModelType.DINOV3_VITB16, "DINOv3 Base (86M params)"),   # Medium size
        # Skip the 7B model for now as it's very large
    ]
    
    for model_type, test_name in models_to_test:
        try:
            success = test_model_loading(model_type, test_name)
            test_results[test_name] = success
        except Exception as e:
            logger.error(f"Test setup failed for {test_name}: {e}")
            test_results[test_name] = False
    
    # Summary
    logger.info("\\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    
    for test_name, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    logger.info(f"\\nResults: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL MODELS ARE COMPATIBLE!")
    elif passed_tests > 0:
        logger.info("⚠️  Some models working, others may need debugging")
    else:
        logger.error("💥 No models working - check dependencies")
    
    return passed_tests > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)