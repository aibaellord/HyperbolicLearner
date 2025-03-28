#!/usr/bin/env python3
"""
YouTube Video Downloader Module

This module provides a high-performance, feature-rich YouTube video downloader
with advanced capabilities including:
- Efficient multi-threaded downloading
- Smart caching to prevent re-downloading
- Quality selection
- Comprehensive metadata extraction
- Robust error handling and retry mechanisms
- Rate limit handling

Usage:
    from video_processor.downloader import YouTubeDownloader
    
    downloader = YouTubeDownloader(cache_dir="./cache")
    video_path = downloader.download("https://www.youtube.com/watch?v=VIDEO_ID", quality="1080p")
    metadata = downloader.get_metadata("https://www.youtube.com/watch?v=VIDEO_ID")
"""

import os
import re
import json
import time
import logging
import hashlib
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from urllib.parse import parse_qs, urlparse
from dataclasses import dataclass, asdict
from datetime import datetime

# Use yt-dlp which is an actively maintained fork of youtube-dl with better performance
try:
    import yt_dlp as youtube_dl
except ImportError:
    try:
        import youtube_dl
        logging.warning("Using youtube_dl instead of yt_dlp. Consider installing yt_dlp for better performance.")
    except ImportError:
        raise ImportError("Either yt_dlp or youtube_dl is required. Install with 'pip install yt-dlp'")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("youtube_downloader")


@dataclass
class VideoMetadata:
    """Dataclass to store video metadata"""
    video_id: str
    title: str
    description: str
    upload_date: str
    uploader: str
    uploader_id: str
    uploader_url: str
    duration: int  # in seconds
    view_count: int
    like_count: Optional[int] = None
    dislike_count: Optional[int] = None
    average_rating: Optional[float] = None
    categories: List[str] = None
    tags: List[str] = None
    thumbnail_url: str = None
    formats: List[Dict[str, Any]] = None
    subtitles: Dict[str, List[Dict[str, str]]] = None
    automatic_captions: Dict[str, List[Dict[str, str]]] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []
        if self.tags is None:
            self.tags = []
        if self.formats is None:
            self.formats = []


class DownloadCache:
    """
    Handles caching of downloaded videos and metadata to prevent redundant downloads
    and enable fast access to previously downloaded content.
    """
    
    def __init__(self, cache_dir: Union[str, Path]):
        self.cache_dir = Path(cache_dir)
        self.metadata_dir = self.cache_dir / "metadata"
        self.video_dir = self.cache_dir / "videos"
        self.lock = threading.Lock()
        
        # Create cache directories if they don't exist
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Cache initialized at {self.cache_dir}")
    
    def get_video_id(self, url: str) -> str:
        """Extract video ID from a YouTube URL"""
        parsed_url = urlparse(url)
        if parsed_url.netloc in ('youtu.be', 'www.youtu.be'):
            return parsed_url.path.lstrip('/')
        elif parsed_url.netloc in ('youtube.com', 'www.youtube.com'):
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query).get('v', [None])[0]
            elif '/v/' in parsed_url.path:
                return parsed_url.path.split('/v/')[1]
            elif '/embed/' in parsed_url.path:
                return parsed_url.path.split('/embed/')[1]
        return None
    
    def get_cache_path(self, video_id: str, quality: str = None) -> Path:
        """Generate cache path for a video"""
        if quality:
            return self.video_dir / f"{video_id}_{quality}.mp4"
        return self.video_dir / f"{video_id}.mp4"
    
    def get_metadata_path(self, video_id: str) -> Path:
        """Generate metadata cache path for a video"""
        return self.metadata_dir / f"{video_id}.json"
    
    def video_exists(self, video_id: str, quality: str = None) -> bool:
        """Check if video exists in cache"""
        cache_path = self.get_cache_path(video_id, quality)
        return cache_path.exists() and cache_path.stat().st_size > 0
    
    def metadata_exists(self, video_id: str) -> bool:
        """Check if metadata exists in cache"""
        metadata_path = self.get_metadata_path(video_id)
        return metadata_path.exists()
    
    def save_metadata(self, video_id: str, metadata: VideoMetadata) -> Path:
        """Save metadata to cache"""
        with self.lock:
            metadata_path = self.get_metadata_path(video_id)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, ensure_ascii=False, indent=2)
            return metadata_path
    
    def get_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """Retrieve metadata from cache if it exists"""
        metadata_path = self.get_metadata_path(video_id)
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return VideoMetadata(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to load metadata from cache: {e}")
            return None
    
    def invalidate(self, video_id: str):
        """Remove cached video and metadata"""
        with self.lock:
            # Remove metadata
            metadata_path = self.get_metadata_path(video_id)
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Remove all quality variants of the video
            for file in self.video_dir.glob(f"{video_id}_*.mp4"):
                file.unlink()
            
            # Remove default quality video
            default_path = self.get_cache_path(video_id)
            if default_path.exists():
                default_path.unlink()


class RateLimiter:
    """
    Implements rate limiting to prevent hitting YouTube's rate limits.
    Uses a token bucket algorithm for throttling requests.
    """
    
    def __init__(self, rate: float = 1.0, per: float = 5.0, burst: int = 2):
        """
        Initialize rate limiter
        
        Args:
            rate: Number of requests allowed per time period
            per: Time period in seconds
            burst: Maximum number of requests allowed in a burst
        """
        self.rate = rate
        self.per = per
        self.tokens = burst
        self.max_tokens = burst
        self.updated_at = time.monotonic()
        self.lock = threading.Lock()
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def acquire(self, block: bool = True) -> bool:
        """Acquire a token, blocking if necessary"""
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.updated_at
            new_tokens = elapsed * (self.rate / self.per)
            self.tokens = min(self.max_tokens, self.tokens + new_tokens)
            self.updated_at = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            if not block:
                return False
            
            # Calculate wait time
            wait_time = (1.0 - self.tokens) * (self.per / self.rate)
            
        # Sleep outside the lock
        time.sleep(wait_time)
        return self.acquire(block=False)


class YouTubeDownloader:
    """
    High-performance YouTube video downloader with advanced features.
    
    Features:
    - Efficient multi-threaded downloading
    - Quality selection
    - Smart caching to prevent re-downloading
    - Comprehensive metadata extraction
    - Robust error handling and retry mechanisms
    - Rate limit handling
    """
    
    def __init__(
        self, 
        cache_dir: Union[str, Path] = None,
        max_retries: int = 3, 
        timeout: int = 60,
        rate_limit: float = 1.0,
        rate_period: float = 5.0,
        verbose: bool = False
    ):
        """
        Initialize the YouTube downloader
        
        Args:
            cache_dir: Directory to store downloaded videos and metadata
            max_retries: Maximum number of retry attempts for failed downloads
            timeout: Timeout in seconds for download operations
            rate_limit: Maximum number of requests per rate_period
            rate_period: Time period in seconds for rate limiting
            verbose: Enable verbose logging
        """
        # Set up caching
        if cache_dir is None:
            cache_dir = Path.home() / ".hyperbolic_learner" / "cache"
        self.cache = DownloadCache(cache_dir)
        
        # Config
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Rate limiting
        self.rate_limiter = RateLimiter(rate=rate_limit, per=rate_period)
        
        # Set up logging
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        logger.info(f"YouTube downloader initialized with cache at {cache_dir}")
    
    def _get_ydl_opts(self, output_path: Union[str, Path], quality: str = None) -> Dict:
        """
        Configure youtube-dl options based on quality and output path
        
        Args:
            output_path: Path where the video will be saved
            quality: Desired video quality (e.g., '1080p', '720p', 'best')
            
        Returns:
            Dictionary of youtube-dl options
        """
        format_selector = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
        
        if quality:
            if quality.lower() == 'audio':
                format_selector = 'bestaudio[ext=m4a]/bestaudio'
            else:
                # Strip non-numeric characters and convert to int
                height = int(re.sub(r'[^\d]', '', quality))
                format_selector = f'bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={height}][ext=mp4]/best'
        
        return {
            'format': format_selector,
            'outtmpl': str(output_path),
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': False,
            'nooverwrites': True,
            'socket_timeout': self.timeout,
            'retries': self.max_retries,
            'noprogress': True,
            'noplaylist': True,  # Only download the video, not the whole playlist
            'merge_output_format': 'mp4',
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
        }
    
    def extract_metadata(self, url: str, force_refresh: bool = False) -> VideoMetadata:
        """
        Extract metadata from a YouTube video
        
        Args:
            url: YouTube video URL
            force_refresh: Force refresh of cached metadata
            
        Returns:
            VideoMetadata object containing video metadata
        """
        video_id = self.cache.get_video_id(url)
        if not video_id:
            raise ValueError(f"Could not extract video ID from URL: {url}")
        
        # Check cache first unless forced refresh
        if not force_refresh and self.cache.metadata_exists(video_id):
            cached_metadata = self.cache.get_metadata(video_id)
            if cached_metadata:
                logger.debug(f"Using cached metadata for video {video_id}")
                return cached_metadata
        
        # Use rate limiter to avoid hitting YouTube's rate limits
        with self.rate_limiter:
            logger.info(f"Extracting metadata for video {video_id}")
            
            # Configure youtube-dl options for metadata extraction only
            ydl_opts = {
                'format': 'best',
                'skip_download': True,
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': False,
                'noplaylist': True,
            }
            
            try:
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    
                    # Create metadata object
                    metadata = VideoMetadata(
                        video_id=video_id,
                        title=info.get('title', ''),
                        description=info.get('description', ''),
                        upload_date=info.get('upload_date', ''),
                        uploader=info.get('uploader', ''),
                        uploader_id=info.get('uploader_id', ''),
                        uploader_url=info.get('uploader_url', ''),
                        duration=info.get('duration', 0),
                        view_count=info.get('view_count', 0),
                        like_count=info.get('like_count'),
                        dislike_count=info.get('dislike_count'),
                        average_rating=info.get('average_rating'),
                        categories=info.get('categories', []),
                        tags=info.get('tags', []),
                        thumbnail_url=info.get('thumbnail'),
                        formats=info.get('formats', []),
                        subtitles=info.get('subtitles', {}),
                        automatic_captions=info.get('automatic_captions', {})
                    )
                    
                    # Cache the metadata
                    self.cache.save_metadata(video_id, metadata)
                    return metadata
                    
            except Exception as e:
                logger.error(f"Error extracting metadata: {e}")
                raise
    
    def get_metadata(self, url: str, force_refresh: bool = False) -> VideoMetadata:
        """
        Get metadata for a YouTube video, using cache when available
        
        Args:
            url: YouTube video URL
            force_refresh: Force

