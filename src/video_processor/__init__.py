"""
HyperbolicLearner Video Processing Components

This package contains modules for downloading, processing, and analyzing video content,
with a focus on extracting knowledge and detecting interactions at accelerated speeds.
"""

from .youtube_learner import YouTubeLearner, ContentQualityAnalyzer
from .downloader import VideoDownloader, DownloadManager, VideoMetadata
from .accelerator import VideoAccelerator, ContentAwareAccelerator, SpeedProfile

__all__ = [
    'YouTubeLearner',
    'ContentQualityAnalyzer',
    'VideoDownloader',
    'DownloadManager',
    'VideoMetadata',
    'VideoAccelerator',
    'ContentAwareAccelerator',
    'SpeedProfile',
]

