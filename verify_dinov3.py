#!/usr/bin/env python3
"""
Simple DINOv3 verification script - just load models to confirm they work.
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.core.config import ProcessingConfig, DINOModelType
from src.core.feature_extractor import DINOFeatureExtractor


def test_model_loading():
    """Test loading both DINOv2 and DINOv3 models"""
    
    models_to_test = [
        (DINOModelType.DINOV2_BASE, "DINOv2 Base"),
        (DINOModelType.DINOV3_VITS16, "DINOv3 Small"), 
        (DINOModelType.DINOV3_VITB16, "DINOv3 Base"),
    ]
    
    print("🧪 DINOv3 MODEL LOADING VERIFICATION")
    print("=" * 60)
    
    results = []
    
    for model_type, model_name in models_to_test:
        print(f"\n🔹 Testing {model_name}")
        print(f"   Model ID: {model_type.value}")
        
        try:
            # Create minimal config
            config = ProcessingConfig(
                input_video_path=Path("/tmp/dummy.mp4"),  # Won't be used
                output_frames_dir=Path("/tmp/output"),     # Won't be used
                device="auto",
                batch_size=1
            )
            config.set_model(model_type)
            
            # Test model loading
            start_time = time.time()
            extractor = DINOFeatureExtractor(config)
            load_time = time.time() - start_time
            
            # Get model info
            model_info = config.get_model_info()
            
            print(f"   ✅ SUCCESS - Loaded in {load_time:.2f}s")
            print(f"      {model_info['estimated_params']}")
            print(f"      Batch size: {model_info['recommended_batch_size']}")
            
            # Cleanup
            extractor.cleanup()
            
            results.append({
                'model': model_name,
                'success': True,
                'load_time': load_time,
                'params': model_info['estimated_params']
            })
            
        except Exception as e:
            print(f"   ❌ FAILED - {e}")
            results.append({
                'model': model_name,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    
    successful_models = [r for r in results if r['success']]
    failed_models = [r for r in results if not r['success']]
    
    print(f"\n✅ Successful Models ({len(successful_models)}/{len(results)}):")
    for result in successful_models:
        print(f"   • {result['model']} - {result['load_time']:.1f}s - {result['params']}")
    
    if failed_models:
        print(f"\n❌ Failed Models ({len(failed_models)}):")
        for result in failed_models:
            print(f"   • {result['model']} - {result['error']}")
    
    # Final assessment
    dinov3_working = any('DINOv3' in r['model'] and r['success'] for r in results)
    dinov2_working = any('DINOv2' in r['model'] and r['success'] for r in results)
    
    print(f"\n📊 FINAL ASSESSMENT:")
    if dinov3_working and dinov2_working:
        print("🎉 EXCELLENT - Both DINOv2 and DINOv3 working!")
        print("   You have full model flexibility for frame extraction.")
    elif dinov2_working:
        print("✅ GOOD - DINOv2 working (production ready)")
        print("   DINOv3 may need authentication or troubleshooting.")
    else:
        print("❌ ISSUES - Check model access and dependencies.")
    
    return len(successful_models) > 0


if __name__ == "__main__":
    success = test_model_loading()
    print(f"\nVerification {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)