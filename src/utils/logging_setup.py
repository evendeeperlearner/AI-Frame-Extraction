#!/usr/bin/env python3
"""
Centralized logging setup for the panoramic video preparation toolkit.
Consolidates duplicate logging configuration across multiple scripts.
"""

import logging
from pathlib import Path
from typing import Optional, List


def setup_logging(
    log_file: Optional[str] = None, 
    level: int = logging.INFO,
    enable_detailed_loggers: bool = True,
    additional_loggers: Optional[List[str]] = None
) -> None:
    """
    Setup standardized logging configuration for the application.
    
    Args:
        log_file: Optional log file path. If None, only console logging is used.
        level: Logging level (default: INFO)
        enable_detailed_loggers: Enable detailed logging for core modules
        additional_loggers: Additional logger names to set to INFO level
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        # Ensure log directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Set specific loggers to INFO for detailed output
    if enable_detailed_loggers:
        detailed_loggers = [
            'src.core.dynamic_sampler',
            'src.core.saliency_analyzer', 
            'src.pipeline.adaptive_extraction_pipeline',
            'src.core.feature_extractor',
            'src.core.video_processor'
        ]
        
        for logger_name in detailed_loggers:
            logging.getLogger(logger_name).setLevel(logging.INFO)
    
    # Set additional loggers if provided
    if additional_loggers:
        for logger_name in additional_loggers:
            logging.getLogger(logger_name).setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def setup_extraction_logging(script_name: str) -> logging.Logger:
    """
    Setup logging specifically for extraction scripts.
    
    Args:
        script_name: Name of the script (used for log filename)
        
    Returns:
        Configured logger instance
    """
    log_file = f"{script_name}_extraction.log"
    setup_logging(
        log_file=log_file,
        enable_detailed_loggers=True
    )
    return get_logger(script_name)