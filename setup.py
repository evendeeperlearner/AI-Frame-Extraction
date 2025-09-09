#!/usr/bin/env python3
"""
Setup script for FFmpeg + DINOv3 Adaptive Frame Extraction.
Production-ready installation with dependency management.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Core requirements (without optional dependencies)
core_requirements = [
    "torch>=2.0.0",
    "torchvision>=0.15.0", 
    "transformers>=4.30.0",
    "opencv-python>=4.8.0",
    "pillow>=9.5.0",
    "scikit-image>=0.21.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.10.0",
    "psutil>=5.9.0",
    "tqdm>=4.65.0",
    "pyyaml>=6.0"
]

# Optional dependencies
optional_requirements = {
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0"
    ],
    "viz": [
        "matplotlib>=3.7.0",
        "tensorboard>=2.13.0"
    ],
    "fast": [
        "numba>=0.57.0",
        "joblib>=1.3.0"
    ],
    "ml": [
        "wandb>=0.15.0",
        "jupyter>=1.0.0"
    ]
}

# All optional dependencies
optional_requirements["all"] = [
    req for reqs in optional_requirements.values() for req in reqs
]

setup(
    name="adaptive-frame-extraction",
    version="1.0.0",
    author="PanoramicVideoPrep",
    author_email="dev@panoramicvideoprep.com",
    description="Adaptive video frame extraction using DINOv3 dense features for improved SfM reconstruction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/panoramicvideoprep/adaptive-frame-extraction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Video",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=core_requirements,
    extras_require=optional_requirements,
    entry_points={
        "console_scripts": [
            "adaptive-extract=src.examples.basic_usage:main",
        ],
    },
    package_data={
        "src": ["**/*.yaml", "**/*.json"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "computer vision",
        "video processing", 
        "frame extraction",
        "structure from motion",
        "sfm",
        "dinov3",
        "deep learning",
        "adaptive sampling",
        "visual attention"
    ],
    project_urls={
        "Bug Reports": "https://github.com/panoramicvideoprep/adaptive-frame-extraction/issues",
        "Source": "https://github.com/panoramicvideoprep/adaptive-frame-extraction",
        "Documentation": "https://adaptive-frame-extraction.readthedocs.io/",
    }
)