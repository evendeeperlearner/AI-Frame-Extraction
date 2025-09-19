# Frame Extraction Examples

This directory contains example scripts demonstrating how to use the panoramic video preparation toolkit.

## Quick Start Examples

### 1. Simple Extraction (`simple_extraction.py`)
Perfect for beginners - minimal setup with sensible defaults.

```bash
python simple_extraction.py your_video.mp4
```

**Features:**
- Automatic device detection (MPS/CUDA/CPU)
- Balanced 3fps/0.5fps extraction rates  
- DINOv2 model for maximum compatibility
- Simple output and error handling

### 2. Advanced Extraction (`advanced_extraction.py`)
Full-featured extraction with all customization options.

```bash
python advanced_extraction.py --video video.mp4 --model dinov3_small --rich-fps 5.0 --poor-fps 0.25
```

**Features:**
- DINOv3 model support
- Custom extraction rates and thresholds
- Resolution and quality control
- Detailed logging and analysis

## Production Scripts

### Production Extract (`../src/examples/production_extract.py`)
The main production script with complete parameter control:

```bash
python ../src/examples/production_extract.py --help
```

**Key Parameters:**
- `--model`: Choose between `dinov2_base`, `dinov3_small`, `dinov3_base`
- `--rich-fps`: Frame rate for detail-rich regions
- `--poor-fps`: Frame rate for simple regions  
- `--threshold`: Feature detection sensitivity (0.0-1.0)
- `--resolution`: Output resolution (e.g., 1920x1080)

## Model Recommendations

| Use Case | Model | Rich FPS | Poor FPS | Notes |
|----------|-------|----------|----------|-------|
| **Quick Test** | dinov2_base | 2.0 | 0.5 | Most compatible |
| **Balanced Quality** | dinov3_small | 4.0 | 0.25 | Good speed/quality |
| **Maximum Quality** | dinov3_base | 8.0 | 0.1 | Best results, slower |
| **Speed Optimized** | dinov3_small | 2.0 | 1.0 | Faster processing |

## Common Workflows

### For Gaussian Splatting
```bash
python advanced_extraction.py --video scene.mp4 --model dinov3_base --rich-fps 6.0 --poor-fps 0.2 --threshold 0.8
```

### For General 3D Reconstruction  
```bash
python simple_extraction.py scene_video.mp4
```

### For High Frame Rate Videos
```bash
python advanced_extraction.py --video high_fps.mp4 --rich-fps 10.0 --poor-fps 0.25 --max-frames 1500
```

## Output Structure

All scripts create organized output:
```
extracted_frames/
├── video_name/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
├── extraction_results.json
├── sfm_metadata.json  
└── extraction.log
```

## Troubleshooting

**Video not found?**
- Use absolute paths: `/full/path/to/video.mp4`
- Check file permissions

**Out of memory?**  
- Reduce `--batch-size` (try 2 or 1)
- Lower `--resolution` (try 1280x720)
- Reduce `--max-frames`

**Poor frame selection?**
- Lower `--threshold` (try 0.6 or 0.5) 
- Adjust `--rich-fps` and `--poor-fps` rates
- Check video quality and content type

For more help, check the main documentation in `../docs/`.