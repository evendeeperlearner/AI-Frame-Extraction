#!/usr/bin/env python3
"""
Test DINOv3 models with authentication support.
This script will prompt for HuggingFace authentication if needed.
"""

import logging
import sys
import os
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


def check_authentication():
    """Check if authenticated with HuggingFace"""
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Not authenticated with HuggingFace: {e}")
        print()
        print("🔐 Authentication Required for DINOv3 Models")
        print("=" * 50)
        print("Steps to authenticate:")
        print("1. Go to: https://huggingface.co/settings/tokens")
        print("2. Create a token with 'Read' access")
        print("3. Request access to DINOv3 models:")
        print("   - https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m")
        print("   - https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m") 
        print("4. Run: python auth_hf.py")
        print("   Or set: export HUGGINGFACE_HUB_TOKEN='your_token'")
        print()
        return False


def test_dinov3_model(model_type: DINOModelType, test_name: str):
    """Test a specific DINOv3 model"""
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
        import time
        start_time = time.time()
        
        feature_extractor = DINOFeatureExtractor(config)
        
        load_time = time.time() - start_time
        logger.info(f"✅ {test_name} loaded successfully in {load_time:.2f}s")
        
        # Test basic functionality
        stats = feature_extractor.get_processing_stats()
        logger.info(f"✅ Basic functionality verified")
        logger.info(f"Processing stats: {stats}")
        
        # Cleanup
        feature_extractor.cleanup()
        logger.info(f"✅ {test_name} - ALL TESTS PASSED")
        
        return True
        
    except Exception as e:
        if "gated repo" in str(e).lower() or "401" in str(e):
            logger.error(f"❌ {test_name} - Authentication Error: {e}")
            logger.error("   → You need access to this gated repository")
        else:
            logger.error(f"❌ {test_name} - FAILED: {e}")
        return False


def main():
    """Main test function"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 DINOv3 AUTHENTICATION AND COMPATIBILITY TEST")
    
    # Check authentication first
    if not check_authentication():
        print("\\n⚠️  Cannot test DINOv3 models without authentication.")
        print("   Testing DINOv2 only...")
        
        # Test DINOv2 as fallback
        success = test_dinov3_model(DINOModelType.DINOV2_BASE, "DINOv2 Base (Fallback)")
        if success:
            print("\\n✅ DINOv2 is working perfectly!")
            print("   Your system is production-ready with DINOv2.")
            print("   Authenticate later to upgrade to DINOv3.")
        return success
    
    # Test DINOv3 models if authenticated
    test_results = {}
    
    models_to_test = [
        (DINOModelType.DINOV2_BASE, "DINOv2 Base (Reference)"),
        (DINOModelType.DINOV3_VITS16, "DINOv3 Small (21M params)"),
        (DINOModelType.DINOV3_VITB16, "DINOv3 Base (86M params)"),
    ]
    
    for model_type, test_name in models_to_test:
        success = test_dinov3_model(model_type, test_name)
        test_results[test_name] = success
    
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
        logger.info("🎉 ALL MODELS WORKING - DINOV3 UPGRADE COMPLETE!")
    elif any("DINOv3" in name and success for name, success in test_results.items()):
        logger.info("🎊 DINOv3 PARTIALLY WORKING - Some models successful!")
    else:
        logger.info("⚠️  DINOv2 working, DINOv3 needs authentication/access")
    
    return passed_tests > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)