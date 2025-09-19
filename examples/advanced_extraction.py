#!/usr/bin/env python3
"""
Advanced extraction example - demonstrates full configuration options.
Shows how to use DINOv3, custom parameters, and detailed analysis.

Usage:
    python advanced_extraction.py --video video.mp4 --model dinov3_small --rich-fps 5.0
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.examples.production_extract import main as production_main


def main():
    """
    Advanced extraction using the production script.
    This is a wrapper that demonstrates the full parameter set.
    """
    print("🎯 Advanced Frame Extraction Example")
    print("=" * 50)
    print("This example uses the full-featured production extraction script.")
    print("You can customize all parameters for optimal results.")
    print()
    
    # Show example usage
    print("Example commands:")
    print("  # High-quality extraction with DINOv3")
    print("  python advanced_extraction.py --video video.mp4 --model dinov3_small --rich-fps 8.0 --poor-fps 0.25")
    print()
    print("  # Maximum quality with large model")
    print("  python advanced_extraction.py --video video.mp4 --model dinov3_base --threshold 0.8 --max-frames 2000")
    print()
    print("  # Speed-optimized extraction")
    print("  python advanced_extraction.py --video video.mp4 --rich-fps 2.0 --poor-fps 1.0 --resolution 1280x720")
    print()
    
    # If arguments provided, run the production script
    if len(sys.argv) > 1:
        print("Running with provided arguments...")
        print("=" * 50)
        # Pass through to the production script
        success = production_main()
        
        if success:
            print()
            print("🎉 Advanced extraction completed successfully!")
            print("💡 Tip: Check the log file for detailed processing information.")
        
        sys.exit(0 if success else 1)
    else:
        print("💡 Add arguments to run extraction, or use --help for all options.")


if __name__ == "__main__":
    main()