"""
Custom exceptions for the adaptive frame extraction pipeline.
"""


class AdaptiveExtractionError(Exception):
    """Base exception for adaptive extraction pipeline"""
    pass


class FFmpegError(AdaptiveExtractionError):
    """FFmpeg-related errors"""
    pass


class VideoProcessingError(AdaptiveExtractionError):
    """Video processing errors"""
    pass


class FeatureExtractionError(AdaptiveExtractionError):
    """Feature extraction errors"""
    pass


class SaliencyAnalysisError(AdaptiveExtractionError):
    """Saliency analysis errors"""
    pass


class ConfigurationError(AdaptiveExtractionError):
    """Configuration-related errors"""
    pass


class DeviceError(AdaptiveExtractionError):
    """Device-specific errors (MPS, CUDA, CPU)"""
    pass


class MemoryError(AdaptiveExtractionError):
    """Memory management errors"""
    pass


class QualityAssessmentError(AdaptiveExtractionError):
    """Frame quality assessment errors"""
    pass


class SfMOptimizationError(AdaptiveExtractionError):
    """SfM optimization errors"""
    pass