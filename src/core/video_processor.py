"""
FFmpeg-based video processing with streaming support and quality assessment.
Optimized for adaptive frame extraction with visual detail analysis.
"""

import subprocess
import logging
import json
from pathlib import Path
from typing import List, Optional, Tuple, Generator, Dict, Any
import cv2
import numpy as np

from .config import ProcessingConfig, QualityMetrics
from .exceptions import FFmpegError, VideoProcessingError


class VideoProcessor:
    """FFmpeg-based video processing with error handling and streaming support"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._validate_ffmpeg()
    
    def _validate_ffmpeg(self):
        """Validate FFmpeg installation and version"""
        try:
            result = subprocess.run(
                [self.config.ffmpeg_path, "-version"],
                capture_output=True, text=True, check=True, timeout=10
            )
            version_line = result.stdout.split('\n')[0]
            self.logger.info(f"FFmpeg validated: {version_line}")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise FFmpegError(f"FFmpeg validation failed: {e}")
    
    def get_video_info(self) -> Dict[str, Any]:
        """Extract comprehensive video metadata using ffprobe"""
        ffprobe_cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(self.config.input_video_path)
        ]
        
        try:
            result = subprocess.run(
                ffprobe_cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            
            video_info = json.loads(result.stdout)
            
            # Extract key information
            video_stream = next(
                (stream for stream in video_info['streams'] 
                 if stream['codec_type'] == 'video'), None
            )
            
            if not video_stream:
                raise VideoProcessingError("No video stream found in input file")
            
            processed_info = {
                'duration': float(video_info['format'].get('duration', 0)),
                'size_bytes': int(video_info['format'].get('size', 0)),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': self._parse_framerate(video_stream.get('r_frame_rate', '0/1')),
                'total_frames': int(video_stream.get('nb_frames', 0)),
                'codec': video_stream.get('codec_name', 'unknown'),
                'pixel_format': video_stream.get('pix_fmt', 'unknown'),
                'bit_rate': int(video_stream.get('bit_rate', 0)),
                'format_name': video_info['format'].get('format_name', 'unknown')
            }
            
            self.logger.info(f"Video info: {processed_info['width']}x{processed_info['height']} "
                           f"@ {processed_info['fps']}fps, {processed_info['duration']:.1f}s")
            
            return processed_info
            
        except subprocess.TimeoutExpired:
            raise VideoProcessingError("FFprobe timeout - video file may be corrupted or too large")
        except json.JSONDecodeError as e:
            raise VideoProcessingError(f"Failed to parse video metadata: {e}")
        except Exception as e:
            raise VideoProcessingError(f"Failed to get video info: {e}")
    
    def _parse_framerate(self, framerate_str: str) -> float:
        """Parse framerate from fraction string"""
        try:
            if '/' in framerate_str:
                num, den = framerate_str.split('/')
                return float(num) / float(den) if float(den) > 0 else 0
            return float(framerate_str)
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def extract_frames_streaming(self) -> Generator[Tuple[int, np.ndarray, QualityMetrics], None, None]:
        """
        Stream frames from video with quality assessment.
        Yields: (frame_index, frame_array, quality_metrics)
        """
        w, h = self.config.target_resolution
        
        ffmpeg_cmd = [
            self.config.ffmpeg_path,
            "-i", str(self.config.input_video_path),
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}"
        ]
        
        if self.config.fps_extract:
            ffmpeg_cmd.extend(["-r", str(self.config.fps_extract)])
        
        ffmpeg_cmd.append("-")
        
        try:
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            frame_count = 0
            frame_size = w * h * 3  # RGB
            
            while True:
                raw_frame = process.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    break
                
                # Convert raw bytes to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((h, w, 3))
                
                # Assess frame quality
                quality_metrics = self._assess_frame_quality(frame)
                
                yield frame_count, frame, quality_metrics
                frame_count += 1
            
            process.stdout.close()
            return_code = process.wait()
            
            if return_code != 0:
                stderr_output = process.stderr.read().decode()
                raise FFmpegError(f"FFmpeg failed with return code {return_code}: {stderr_output}")
        
        except Exception as e:
            if 'process' in locals():
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    process.kill()
            raise VideoProcessingError(f"Frame extraction failed: {e}")
    
    def _assess_frame_quality(self, frame: np.ndarray) -> QualityMetrics:
        """Assess frame quality using multiple metrics"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # 1. Sharpness assessment (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(sharpness / 1000.0, 1.0)  # Normalize
            
            # 2. Exposure quality (histogram analysis)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist / hist.sum()
            
            # Check for over/under exposure
            overexposed = hist_norm[240:].sum()
            underexposed = hist_norm[:15].sum() 
            exposure_score = 1.0 - max(overexposed, underexposed)
            
            # 3. Motion blur detection (edge energy)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edge_energy = np.sqrt(sobel_x**2 + sobel_y**2).mean()
            motion_blur_score = min(edge_energy / 50.0, 1.0)
            
            # 4. Noise level estimation
            noise_level = self._estimate_noise_level(gray)
            noise_score = max(0, 1.0 - noise_level / 20.0)
            
            # 5. Feature density (corner detection)
            corners = cv2.cornerHarris(gray, 2, 3, 0.04)
            feature_density = (corners > 0.01 * corners.max()).sum() / gray.size
            
            # Composite quality score
            composite_quality = (
                0.3 * sharpness_score +
                0.2 * exposure_score +
                0.2 * motion_blur_score +
                0.15 * noise_score +
                0.15 * feature_density
            )
            
            return QualityMetrics(
                sharpness_score=sharpness_score,
                exposure_score=exposure_score,
                motion_blur_score=motion_blur_score,
                noise_level=1.0 - noise_score,
                feature_density=feature_density,
                composite_quality=composite_quality
            )
        
        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {e}")
            return QualityMetrics()  # Return default metrics
    
    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """Estimate noise level using Laplacian method"""
        try:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_level = laplacian.var() ** 0.5
            return min(noise_level, 100.0)  # Cap at reasonable level
        except:
            return 0.0
    
    def extract_frame_at_timestamp(self, timestamp: float) -> Tuple[np.ndarray, QualityMetrics]:
        """Extract a single frame at specific timestamp"""
        w, h = self.config.target_resolution
        
        ffmpeg_cmd = [
            self.config.ffmpeg_path,
            "-ss", str(timestamp),
            "-i", str(self.config.input_video_path),
            "-vframes", "1",
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-"
        ]
        
        try:
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode != 0:
                raise FFmpegError(f"Frame extraction failed: {result.stderr.decode()}")
            
            frame_size = w * h * 3
            if len(result.stdout) != frame_size:
                raise VideoProcessingError(f"Unexpected frame size: {len(result.stdout)} vs {frame_size}")
            
            frame = np.frombuffer(result.stdout, dtype=np.uint8)
            frame = frame.reshape((h, w, 3))
            
            quality_metrics = self._assess_frame_quality(frame)
            
            return frame, quality_metrics
        
        except subprocess.TimeoutExpired:
            raise VideoProcessingError(f"Timeout extracting frame at timestamp {timestamp}")
        except Exception as e:
            raise VideoProcessingError(f"Failed to extract frame at {timestamp}s: {e}")
    
    def validate_video_file(self) -> bool:
        """Validate that the input video file is readable and processable"""
        try:
            if not self.config.input_video_path.exists():
                raise VideoProcessingError(f"Video file does not exist: {self.config.input_video_path}")
            
            if self.config.input_video_path.stat().st_size == 0:
                raise VideoProcessingError("Video file is empty")
            
            # Try to get video info as validation
            video_info = self.get_video_info()
            
            if video_info['duration'] <= 0:
                raise VideoProcessingError("Video has zero duration")
            
            if video_info['width'] <= 0 or video_info['height'] <= 0:
                raise VideoProcessingError("Video has invalid dimensions")
            
            self.logger.info(f"Video validation successful: {video_info}")
            return True
        
        except Exception as e:
            self.logger.error(f"Video validation failed: {e}")
            raise