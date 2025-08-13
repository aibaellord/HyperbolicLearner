#!/usr/bin/env python3
"""
Intelligent Caching System with Predictive Pre-loading
=====================================================

This module implements an advanced caching system that predicts what video content
will be processed next and pre-loads features, model outputs, and intermediate
results for ultra-fast processing.

Key Features:
- Predictive pre-loading based on usage patterns
- Multi-level caching (memory, SSD, network)
- Content similarity-based cache sharing
- Adaptive cache size management
- Compression-aware caching for video features
- User behavior pattern learning for cache optimization
"""

import os
import sys
import time
import logging
import hashlib
import pickle
import gzip
import threading
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict, OrderedDict, deque
import heapq
import sqlite3
import json
import psutil
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import aiofiles.os

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Represents a cached item with metadata"""
    key: str
    data: Any
    size_bytes: int
    creation_time: float
    last_access_time: float
    access_count: int
    ttl: Optional[float] = None  # Time to live
    compression_ratio: float = 1.0
    similarity_hash: Optional[str] = None
    user_id: Optional[str] = None
    priority_score: float = 1.0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.creation_time > self.ttl
    
    def update_access(self):
        """Update access statistics"""
        self.last_access_time = time.time()
        self.access_count += 1

@dataclass
class CacheConfig:
    """Configuration for intelligent caching system"""
    memory_cache_size_gb: float = 4.0      # Memory cache size
    ssd_cache_size_gb: float = 50.0        # SSD cache size
    max_network_cache_gb: float = 100.0    # Network cache size
    enable_compression: bool = True         # Enable data compression
    compression_threshold_mb: float = 10.0 # Compress items larger than this
    prefetch_threshold: float = 0.7        # Start prefetching at this probability
    similarity_threshold: float = 0.8      # Content similarity threshold
    cache_persistence: bool = True         # Persist cache across sessions
    predictive_loading: bool = True        # Enable predictive pre-loading
    max_prediction_horizon: int = 10       # Number of items to predict ahead
    learning_rate: float = 0.1            # Learning rate for pattern adaptation

class ContentSimilarityAnalyzer:
    """Analyzes content similarity for cache sharing"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.feature_cache = {}
        
    def extract_content_features(self, video_path: str) -> np.ndarray:
        """Extract features for similarity comparison"""
        cache_key = f"features_{hashlib.md5(video_path.encode()).hexdigest()}"
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # Extract basic file-based features
        try:
            file_path = Path(video_path)
            if not file_path.exists():
                return np.array([])
            
            # File size and modification time
            stat = file_path.stat()
            file_features = [
                stat.st_size,
                stat.st_mtime,
                len(file_path.stem),  # filename length
                hash(file_path.suffix) % 1000,  # file extension hash
            ]
            
            # Content-based features would go here
            # For now, using file-based features as proxy
            features = np.array(file_features, dtype=np.float32)
            
            # Normalize features
            if len(features) > 0:
                features = features / (np.linalg.norm(features) + 1e-8)
            
            self.feature_cache[cache_key] = features
            return features
            
        except Exception as e:
            logger.warning(f"Failed to extract features for {video_path}: {e}")
            return np.array([])
    
    def calculate_similarity(self, video1: str, video2: str) -> float:
        """Calculate similarity between two videos"""
        features1 = self.extract_content_features(video1)
        features2 = self.extract_content_features(video2)
        
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-8
        )
        
        return float(similarity)
    
    def find_similar_content(self, video_path: str, cached_videos: List[str]) -> List[Tuple[str, float]]:
        """Find similar content in cache"""
        similarities = []
        
        for cached_video in cached_videos:
            similarity = self.calculate_similarity(video_path, cached_video)
            if similarity >= self.similarity_threshold:
                similarities.append((cached_video, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities

class PredictivePatternLearner:
    """Learns user patterns to predict next content requests"""
    
    def __init__(self, learning_rate: float = 0.1, max_history: int = 1000):
        self.learning_rate = learning_rate
        self.max_history = max_history
        self.access_history = deque(maxlen=max_history)
        self.pattern_weights = defaultdict(float)
        self.sequence_patterns = defaultdict(lambda: defaultdict(float))
        self.temporal_patterns = defaultdict(list)
        self.user_preferences = defaultdict(dict)
        
    def record_access(self, key: str, user_id: Optional[str] = None, timestamp: Optional[float] = None):
        """Record an access pattern"""
        if timestamp is None:
            timestamp = time.time()
        
        access_record = {
            'key': key,
            'user_id': user_id,
            'timestamp': timestamp,
            'hour': time.localtime(timestamp).tm_hour,
            'weekday': time.localtime(timestamp).tm_wday
        }
        
        self.access_history.append(access_record)
        self._update_patterns(access_record)
    
    def _update_patterns(self, access_record: Dict[str, Any]):
        """Update pattern weights based on new access"""
        key = access_record['key']
        user_id = access_record['user_id']
        timestamp = access_record['timestamp']
        
        # Update sequence patterns (what comes after what)
        if len(self.access_history) >= 2:
            prev_key = self.access_history[-2]['key']
            self.sequence_patterns[prev_key][key] += self.learning_rate
        
        # Update temporal patterns (when things are accessed)
        hour = access_record['hour']
        weekday = access_record['weekday']
        
        temporal_key = f"{hour}:{weekday}"
        self.temporal_patterns[temporal_key].append((key, timestamp))
        
        # Keep only recent temporal data
        cutoff_time = timestamp - 7 * 24 * 3600  # 7 days
        self.temporal_patterns[temporal_key] = [
            (k, t) for k, t in self.temporal_patterns[temporal_key] 
            if t > cutoff_time
        ]
        
        # Update user preferences
        if user_id:
            if key not in self.user_preferences[user_id]:
                self.user_preferences[user_id][key] = 0
            self.user_preferences[user_id][key] += 1
    
    def predict_next_items(self, current_key: str, num_predictions: int = 5, 
                          user_id: Optional[str] = None) -> List[Tuple[str, float]]:
        """Predict next likely items to be accessed"""
        predictions = defaultdict(float)
        
        # Sequence-based predictions
        if current_key in self.sequence_patterns:
            for next_key, weight in self.sequence_patterns[current_key].items():
                predictions[next_key] += weight * 0.4
        
        # Temporal predictions
        current_time = time.time()
        current_hour = time.localtime(current_time).tm_hour
        current_weekday = time.localtime(current_time).tm_wday
        temporal_key = f"{current_hour}:{current_weekday}"
        
        if temporal_key in self.temporal_patterns:
            recent_items = self.temporal_patterns[temporal_key]
            for item_key, item_time in recent_items:
                # More recent items get higher weight
                time_weight = 1.0 / (1.0 + (current_time - item_time) / 3600)  # Hour-based decay
                predictions[item_key] += time_weight * 0.3
        
        # User preference predictions
        if user_id and user_id in self.user_preferences:
            user_prefs = self.user_preferences[user_id]
            total_accesses = sum(user_prefs.values())
            
            for item_key, access_count in user_prefs.items():
                preference_score = access_count / total_accesses
                predictions[item_key] += preference_score * 0.3
        
        # Sort predictions by score
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_predictions[:num_predictions]
    
    def get_prefetch_probability(self, key: str, user_id: Optional[str] = None) -> float:
        """Get probability that an item should be prefetched"""
        predictions = self.predict_next_items(
            self.access_history[-1]['key'] if self.access_history else '', 
            num_predictions=10, 
            user_id=user_id
        )
        
        for predicted_key, score in predictions:
            if predicted_key == key:
                return min(1.0, score)
        
        return 0.0

class MultiLevelCache:
    """Multi-level caching system (Memory -> SSD -> Network)"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Memory cache (L1)
        self.memory_cache = OrderedDict()
        self.memory_cache_size = 0
        self.memory_cache_lock = threading.RLock()
        
        # SSD cache directory (L2)
        self.ssd_cache_dir = Path('./cache/ssd_cache')
        self.ssd_cache_dir.mkdir(parents=True, exist_ok=True)
        self.ssd_cache_index = {}
        self.ssd_cache_size = 0
        self.ssd_cache_lock = threading.RLock()
        
        # Network cache (L3) - simulated with local directory
        self.network_cache_dir = Path('./cache/network_cache')
        self.network_cache_dir.mkdir(parents=True, exist_ok=True)
        self.network_cache_index = {}
        self.network_cache_lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            'memory_hits': 0,
            'memory_misses': 0,
            'ssd_hits': 0,
            'ssd_misses': 0,
            'network_hits': 0,
            'network_misses': 0,
            'prefetch_hits': 0,
            'prefetch_misses': 0
        }
        
        # Load existing cache indices
        self._load_cache_indices()
        
        logger.info(f"Initialized multi-level cache with {config.memory_cache_size_gb}GB memory, "
                   f"{config.ssd_cache_size_gb}GB SSD, {config.max_network_cache_gb}GB network")
    
    def _generate_cache_key(self, data: Any) -> str:
        """Generate a unique cache key for data"""
        if isinstance(data, str):
            content = data
        elif hasattr(data, '__dict__'):
            content = str(data.__dict__)
        else:
            content = str(data)
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data if configured"""
        serialized = pickle.dumps(data)
        
        if self.config.enable_compression and len(serialized) > self.config.compression_threshold_mb * 1024 * 1024:
            compressed = gzip.compress(serialized)
            logger.debug(f"Compressed data from {len(serialized)} to {len(compressed)} bytes")
            return compressed
        
        return serialized
    
    def _decompress_data(self, data: bytes) -> Any:
        """Decompress data"""
        try:
            # Try decompressing first
            decompressed = gzip.decompress(data)
            return pickle.loads(decompressed)
        except:
            # If decompression fails, assume it's not compressed
            return pickle.loads(data)
    
    async def get(self, key: str, user_id: Optional[str] = None) -> Optional[CacheEntry]:
        """Get item from cache (checks all levels)"""
        # Check memory cache first (L1)
        with self.memory_cache_lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.is_expired():
                    entry.update_access()
                    # Move to end (most recently used)
                    self.memory_cache.move_to_end(key)
                    self.stats['memory_hits'] += 1
                    return entry
                else:
                    # Remove expired entry
                    self._remove_from_memory_cache(key)
            
            self.stats['memory_misses'] += 1
        
        # Check SSD cache (L2)
        ssd_entry = await self._get_from_ssd_cache(key)
        if ssd_entry:
            # Promote to memory cache
            await self._promote_to_memory_cache(key, ssd_entry)
            self.stats['ssd_hits'] += 1
            return ssd_entry
        
        self.stats['ssd_misses'] += 1
        
        # Check network cache (L3)
        network_entry = await self._get_from_network_cache(key)
        if network_entry:
            # Promote to SSD and memory cache
            await self._promote_to_ssd_cache(key, network_entry)
            await self._promote_to_memory_cache(key, network_entry)
            self.stats['network_hits'] += 1
            return network_entry
        
        self.stats['network_misses'] += 1
        return None
    
    async def put(self, key: str, data: Any, ttl: Optional[float] = None, 
                  user_id: Optional[str] = None, priority: float = 1.0):
        """Put item into cache"""
        # Serialize and compress data
        compressed_data = self._compress_data(data)
        size_bytes = len(compressed_data)
        compression_ratio = size_bytes / len(pickle.dumps(data))
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            data=data,
            size_bytes=size_bytes,
            creation_time=time.time(),
            last_access_time=time.time(),
            access_count=1,
            ttl=ttl,
            compression_ratio=compression_ratio,
            user_id=user_id,
            priority_score=priority
        )
        
        # Store in memory cache if it fits
        if size_bytes <= self.config.memory_cache_size_gb * 1024**3:
            await self._put_in_memory_cache(key, entry)
        
        # Also store in SSD cache
        await self._put_in_ssd_cache(key, entry, compressed_data)
    
    async def _promote_to_memory_cache(self, key: str, entry: CacheEntry):
        """Promote entry to memory cache"""
        with self.memory_cache_lock:
            # Check if we need to evict items
            while (self.memory_cache_size + entry.size_bytes > 
                   self.config.memory_cache_size_gb * 1024**3):
                if not self.memory_cache:
                    break
                self._evict_from_memory_cache()
            
            self.memory_cache[key] = entry
            self.memory_cache_size += entry.size_bytes
    
    async def _put_in_memory_cache(self, key: str, entry: CacheEntry):
        """Put entry in memory cache"""
        with self.memory_cache_lock:
            # Evict if necessary
            while (self.memory_cache_size + entry.size_bytes > 
                   self.config.memory_cache_size_gb * 1024**3):
                if not self.memory_cache:
                    break
                self._evict_from_memory_cache()
            
            if key in self.memory_cache:
                old_entry = self.memory_cache[key]
                self.memory_cache_size -= old_entry.size_bytes
            
            self.memory_cache[key] = entry
            self.memory_cache_size += entry.size_bytes
    
    def _evict_from_memory_cache(self):
        """Evict least recently used item from memory cache"""
        if not self.memory_cache:
            return
        
        # Remove least recently used (first item in OrderedDict)
        lru_key, lru_entry = self.memory_cache.popitem(last=False)
        self.memory_cache_size -= lru_entry.size_bytes
        
        logger.debug(f"Evicted {lru_key} from memory cache")
    
    def _remove_from_memory_cache(self, key: str):
        """Remove specific item from memory cache"""
        if key in self.memory_cache:
            entry = self.memory_cache.pop(key)
            self.memory_cache_size -= entry.size_bytes
    
    async def _get_from_ssd_cache(self, key: str) -> Optional[CacheEntry]:
        """Get item from SSD cache"""
        with self.ssd_cache_lock:
            if key not in self.ssd_cache_index:
                return None
            
            cache_file = self.ssd_cache_dir / f"{key}.cache"
            if not cache_file.exists():
                # Remove stale index entry
                del self.ssd_cache_index[key]
                return None
            
            try:
                async with aiofiles.open(cache_file, 'rb') as f:
                    compressed_data = await f.read()
                
                data = self._decompress_data(compressed_data)
                
                # Update index
                entry_info = self.ssd_cache_index[key]
                entry = CacheEntry(
                    key=key,
                    data=data,
                    size_bytes=entry_info['size_bytes'],
                    creation_time=entry_info['creation_time'],
                    last_access_time=time.time(),
                    access_count=entry_info.get('access_count', 1) + 1,
                    ttl=entry_info.get('ttl'),
                    compression_ratio=entry_info.get('compression_ratio', 1.0),
                    user_id=entry_info.get('user_id'),
                    priority_score=entry_info.get('priority_score', 1.0)
                )
                
                # Check if expired
                if entry.is_expired():
                    await self._remove_from_ssd_cache(key)
                    return None
                
                # Update index with new access info
                self.ssd_cache_index[key]['access_count'] = entry.access_count
                self.ssd_cache_index[key]['last_access_time'] = entry.last_access_time
                
                return entry
                
            except Exception as e:
                logger.warning(f"Failed to read from SSD cache {key}: {e}")
                await self._remove_from_ssd_cache(key)
                return None
    
    async def _put_in_ssd_cache(self, key: str, entry: CacheEntry, compressed_data: bytes):
        """Put entry in SSD cache"""
        with self.ssd_cache_lock:
            # Check if we need to evict
            while (self.ssd_cache_size + entry.size_bytes > 
                   self.config.ssd_cache_size_gb * 1024**3):
                if not self.ssd_cache_index:
                    break
                await self._evict_from_ssd_cache()
            
            cache_file = self.ssd_cache_dir / f"{key}.cache"
            
            try:
                async with aiofiles.open(cache_file, 'wb') as f:
                    await f.write(compressed_data)
                
                # Update index
                self.ssd_cache_index[key] = {
                    'size_bytes': entry.size_bytes,
                    'creation_time': entry.creation_time,
                    'last_access_time': entry.last_access_time,
                    'access_count': entry.access_count,
                    'ttl': entry.ttl,
                    'compression_ratio': entry.compression_ratio,
                    'user_id': entry.user_id,
                    'priority_score': entry.priority_score
                }
                
                self.ssd_cache_size += entry.size_bytes
                
            except Exception as e:
                logger.warning(f"Failed to write to SSD cache {key}: {e}")
    
    async def _evict_from_ssd_cache(self):
        """Evict least recently used item from SSD cache"""
        if not self.ssd_cache_index:
            return
        
        # Find LRU item
        lru_key = min(self.ssd_cache_index.keys(), 
                     key=lambda k: self.ssd_cache_index[k]['last_access_time'])
        
        await self._remove_from_ssd_cache(lru_key)
    
    async def _remove_from_ssd_cache(self, key: str):
        """Remove item from SSD cache"""
        if key in self.ssd_cache_index:
            entry_info = self.ssd_cache_index.pop(key)
            self.ssd_cache_size -= entry_info['size_bytes']
            
            cache_file = self.ssd_cache_dir / f"{key}.cache"
            try:
                await aiofiles.os.remove(cache_file)
            except FileNotFoundError:
                pass
    
    async def _promote_to_ssd_cache(self, key: str, entry: CacheEntry):
        """Promote entry to SSD cache"""
        compressed_data = self._compress_data(entry.data)
        await self._put_in_ssd_cache(key, entry, compressed_data)
    
    async def _get_from_network_cache(self, key: str) -> Optional[CacheEntry]:
        """Get item from network cache (simulated with local storage)"""
        # Similar implementation to SSD cache but with network simulation
        # For simplicity, using same logic as SSD cache
        return None  # Placeholder
    
    def _load_cache_indices(self):
        """Load cache indices from persistent storage"""
        ssd_index_file = self.ssd_cache_dir / 'index.json'
        if ssd_index_file.exists():
            try:
                with open(ssd_index_file, 'r') as f:
                    self.ssd_cache_index = json.load(f)
                
                # Calculate current cache size
                self.ssd_cache_size = sum(
                    info['size_bytes'] for info in self.ssd_cache_index.values()
                )
                
                logger.info(f"Loaded SSD cache index with {len(self.ssd_cache_index)} entries")
                
            except Exception as e:
                logger.warning(f"Failed to load SSD cache index: {e}")
                self.ssd_cache_index = {}
    
    async def save_cache_indices(self):
        """Save cache indices to persistent storage"""
        ssd_index_file = self.ssd_cache_dir / 'index.json'
        try:
            with open(ssd_index_file, 'w') as f:
                json.dump(self.ssd_cache_index, f)
        except Exception as e:
            logger.warning(f"Failed to save SSD cache index: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = sum(self.stats.values())
        
        stats = self.stats.copy()
        stats.update({
            'memory_cache_size_mb': self.memory_cache_size / 1024**2,
            'memory_cache_entries': len(self.memory_cache),
            'ssd_cache_size_mb': self.ssd_cache_size / 1024**2,
            'ssd_cache_entries': len(self.ssd_cache_index),
            'total_requests': total_requests,
            'hit_rate': (stats['memory_hits'] + stats['ssd_hits'] + stats['network_hits']) / max(total_requests, 1)
        })
        
        return stats

class IntelligentCache:
    """Main intelligent caching system coordinator"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.cache = MultiLevelCache(self.config)
        self.similarity_analyzer = ContentSimilarityAnalyzer(self.config.similarity_threshold)
        self.pattern_learner = PredictivePatternLearner(self.config.learning_rate)
        
        # Prefetching
        self.prefetch_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="prefetch")
        self.prefetch_queue = asyncio.Queue()
        self.prefetching_active = set()
        
        # Background tasks
        self.background_tasks = []
        
        # Start background processes
        self._start_background_tasks()
        
        logger.info("Initialized intelligent cache system")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Background prefetching
        if self.config.predictive_loading:
            task = asyncio.create_task(self._prefetch_worker())
            self.background_tasks.append(task)
        
        # Cache maintenance
        task = asyncio.create_task(self._cache_maintenance_worker())
        self.background_tasks.append(task)
    
    async def get_or_compute(self, key: str, compute_func: Callable, 
                           user_id: Optional[str] = None, **kwargs) -> Any:
        """Get from cache or compute if not cached"""
        # Record access pattern
        self.pattern_learner.record_access(key, user_id)
        
        # Check cache first
        cached_entry = await self.cache.get(key, user_id)
        if cached_entry:
            logger.debug(f"Cache hit for {key}")
            
            # Trigger prefetching of predicted next items
            if self.config.predictive_loading:
                await self._trigger_predictive_prefetch(key, user_id)
            
            return cached_entry.data
        
        # Not in cache, compute
        logger.debug(f"Cache miss for {key}, computing...")
        start_time = time.time()
        
        result = await self._compute_with_similarity_check(key, compute_func, **kwargs)
        
        compute_time = time.time() - start_time
        
        # Store in cache
        priority = 1.0 / max(compute_time, 0.1)  # Higher priority for expensive computations
        await self.cache.put(key, result, user_id=user_id, priority=priority)
        
        # Trigger prefetching
        if self.config.predictive_loading:
            await self._trigger_predictive_prefetch(key, user_id)
        
        return result
    
    async def _compute_with_similarity_check(self, key: str, compute_func: Callable, **kwargs) -> Any:
        """Compute result with similarity-based optimization"""
        # Check if we have similar cached content
        if hasattr(compute_func, '__self__') and hasattr(compute_func.__self__, 'video_path'):
            video_path = compute_func.__self__.video_path
            cached_videos = list(self.cache.memory_cache.keys())
            
            similar_content = self.similarity_analyzer.find_similar_content(video_path, cached_videos)
            
            if similar_content:
                most_similar, similarity = similar_content[0]
                logger.info(f"Found similar content {most_similar} with similarity {similarity:.2f}")
                
                # Could potentially reuse or adapt the similar result
                # For now, just log the finding
        
        # Compute normally
        if asyncio.iscoroutinefunction(compute_func):
            return await compute_func(**kwargs)
        else:
            # Run in executor for non-async functions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: compute_func(**kwargs))
    
    async def _trigger_predictive_prefetch(self, current_key: str, user_id: Optional[str] = None):
        """Trigger predictive prefetching based on patterns"""
        predictions = self.pattern_learner.predict_next_items(
            current_key, 
            num_predictions=self.config.max_prediction_horizon,
            user_id=user_id
        )
        
        for predicted_key, probability in predictions:
            if probability >= self.config.prefetch_threshold:
                # Add to prefetch queue
                if predicted_key not in self.prefetching_active:
                    await self.prefetch_queue.put((predicted_key, probability, user_id))
                    self.prefetching_active.add(predicted_key)
                    logger.debug(f"Queued prefetch for {predicted_key} (probability: {probability:.2f})")
    
    async def _prefetch_worker(self):
        """Background worker for prefetching"""
        while True:
            try:
                predicted_key, probability, user_id = await self.prefetch_queue.get()
                
                # Check if already cached
                if await self.cache.get(predicted_key, user_id):
                    self.prefetching_active.discard(predicted_key)
                    continue
                
                # Prefetch logic would go here
                # For now, just simulate prefetching
                logger.debug(f"Prefetching {predicted_key} with probability {probability:.2f}")
                
                # Simulate prefetch delay
                await asyncio.sleep(0.1)
                
                self.prefetching_active.discard(predicted_key)
                
            except Exception as e:
                logger.warning(f"Prefetch worker error: {e}")
            
            finally:
                self.prefetch_queue.task_done()
    
    async def _cache_maintenance_worker(self):
        """Background worker for cache maintenance"""
        while True:
            try:
                # Save cache indices periodically
                await self.cache.save_cache_indices()
                
                # Clean up expired entries
                await self._cleanup_expired_entries()
                
                # Sleep for maintenance interval
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.warning(f"Cache maintenance error: {e}")
    
    async def _cleanup_expired_entries(self):
        """Clean up expired cache entries"""
        # Clean memory cache
        with self.cache.memory_cache_lock:
            expired_keys = [
                key for key, entry in self.cache.memory_cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self.cache._remove_from_memory_cache(key)
        
        # Clean SSD cache
        with self.cache.ssd_cache_lock:
            current_time = time.time()
            expired_keys = []
            
            for key, entry_info in self.cache.ssd_cache_index.items():
                ttl = entry_info.get('ttl')
                if ttl and (current_time - entry_info['creation_time']) > ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                await self.cache._remove_from_ssd_cache(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        cache_stats = self.cache.get_cache_stats()
        
        # Pattern learning stats
        pattern_stats = {
            'access_history_length': len(self.pattern_learner.access_history),
            'sequence_patterns_count': len(self.pattern_learner.sequence_patterns),
            'temporal_patterns_count': len(self.pattern_learner.temporal_patterns),
            'users_tracked': len(self.pattern_learner.user_preferences)
        }
        
        # Similarity analyzer stats
        similarity_stats = {
            'feature_cache_size': len(self.similarity_analyzer.feature_cache),
        }
        
        report = {
            'cache_performance': cache_stats,
            'pattern_learning': pattern_stats,
            'similarity_analysis': similarity_stats,
            'prefetch_queue_size': self.prefetch_queue.qsize(),
            'active_prefetches': len(self.prefetching_active)
        }
        
        return report
    
    async def shutdown(self):
        """Shutdown the cache system gracefully"""
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for background tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Save cache state
        await self.cache.save_cache_indices()
        
        # Shutdown prefetch executor
        self.prefetch_executor.shutdown(wait=True)
        
        logger.info("Intelligent cache system shutdown complete")

# Example usage and testing
async def test_intelligent_cache():
    """Test the intelligent caching system"""
    config = CacheConfig(
        memory_cache_size_gb=1.0,
        ssd_cache_size_gb=5.0,
        predictive_loading=True
    )
    
    cache = IntelligentCache(config)
    
    # Test compute function
    def expensive_computation(value: int) -> int:
        time.sleep(0.1)  # Simulate work
        return value * value
    
    # Test cache hits and misses
    start_time = time.time()
    
    # First call - cache miss
    result1 = await cache.get_or_compute('test_1', expensive_computation, value=10)
    print(f"First call result: {result1}")
    
    # Second call - cache hit
    result2 = await cache.get_or_compute('test_1', expensive_computation, value=10)
    print(f"Second call result: {result2}")
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.3f} seconds")
    
    # Get performance report
    report = cache.get_performance_report()
    print("Performance Report:", json.dumps(report, indent=2))
    
    # Shutdown
    await cache.shutdown()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_intelligent_cache())
