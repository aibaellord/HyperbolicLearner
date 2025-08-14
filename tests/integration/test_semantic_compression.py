import unittest
import os
import time
import numpy as np
import tempfile
from pathlib import Path
import pytest
import torch

from video_processor.semantic_compression.importance_models import (
    VisualImportanceModel, 
    AudioImportanceModel, 
    TranscriptImportanceModel
)
from video_processor.semantic_compression.multimodal_fusion import MultimodalFusion
from src.video_processor.semantic_compression.semantic_compressor import SemanticCompressor


class TestSemanticCompression(unittest.TestCase):
    """
    Integration tests for the semantic compression module.
    
    These tests verify that the semantic compression pipeline works correctly
    across various scenarios, including different video types, hardware
    configurations, and edge cases.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment before all tests."""
        # Create test directory for output files
        cls.test_dir = tempfile.mkdtemp(prefix="semantic_compression_test_")
        
        # Sample video paths for different types
        cls.test_videos = {
            'tutorial': os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_tutorial.mp4'),
            'lecture': os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_lecture.mp4'),
            'demo': os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_demo.mp4'),
            'conversation': os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_conversation.mp4'),
            'silent': os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_silent.mp4'),
            'short': os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_short.mp4'),
            'low_res': os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_low_res.mp4'),
            'high_res': os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_high_res.mp4'),
        }
        
        # Create test videos if they don't exist (for CI environments)
        cls._ensure_test_videos_exist()
        
        # Initialize models
        cls.visual_model = VisualImportanceModel()
        cls.audio_model = AudioImportanceModel()
        cls.transcript_model = TranscriptImportanceModel()
        cls.fusion_model = MultimodalFusion()
        
        # Initialize compressor with default settings
        cls.compressor = SemanticCompressor(
            visual_model=cls.visual_model,
            audio_model=cls.audio_model,
            transcript_model=cls.transcript_model,
            fusion_model=cls.fusion_model,
            output_dir=cls.test_dir
        )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run."""
        # Remove test directory and files
        import shutil
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    @classmethod
    def _ensure_test_videos_exist(cls):
        """Create test videos if they don't exist."""
        # Create test directory structure if it doesn't exist
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Check each test video and create dummy videos if they don't exist
        for video_type, video_path in cls.test_videos.items():
            if not os.path.exists(video_path):
                print(f"Creating dummy test video for {video_type}")
                cls._create_dummy_video(video_path, video_type)
    
    @staticmethod
    def _create_dummy_video(path, video_type):
        """
        Create a dummy video file for testing purposes.
        
        This uses ffmpeg to create a simple video with different characteristics
        based on the video_type.
        """
        import subprocess
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Different parameters based on video type
        duration = "10" if video_type != "short" else "2"
        resolution = "640x480" if video_type != "low_res" else "320x240"
        if video_type == "high_res":
            resolution = "1920x1080"
        
        # Base command
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={resolution}:rate=30",
        ]
        
        # Add audio based on video type
        if video_type != "silent":
            cmd.extend(["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"])
            
        # Output options
        cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        
        # Add audio codec if needed
        if video_type != "silent":
            cmd.extend(["-c:a", "aac"])
            
        # Output file
        cmd.append(path)
        
        # Execute command
        try:
            subprocess.check_call(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            # Create an empty file if ffmpeg fails, so tests can still run
            print(f"Failed to create test video with ffmpeg: {e}")
            print("Creating empty file instead")
            Path(path).touch()
    
    def test_basic_compression(self):
        """
        Test basic functionality of semantic compression.
        
        This test verifies that the compressor can process a standard tutorial video
        and produce a compressed version with the expected characteristics.
        
        Expected outcome:
        - Compressed video is created
        - Compression ratio is within expected range (0.2-0.8)
        - Important segments are preserved based on importance scores
        """
        video_path = self.test_videos['tutorial']
        if not os.path.getsize(video_path) > 0:
            self.skipTest("Test video is empty or not available")
        
        output_path = os.path.join(self.test_dir, 'compressed_tutorial.mp4')
        
        # Set target acceleration
        target_acceleration = 5.0
        
        # Compress the video
        result = self.compressor.compress(
            video_path, 
            output_path=output_path,
            target_acceleration=target_acceleration
        )
        
        # Verify the compressed video exists
        self.assertTrue(os.path.exists(output_path), "Compressed video was not created")
        
        # Verify the compression ratio is within expected range
        # Expected range: between 0.1 (10x compression) and 0.5 (2x compression)
        expected_min_ratio = 1.0 / (target_acceleration * 1.5)  # Allow some variance
        expected_max_ratio = 1.0 / (target_acceleration * 0.5)  # Allow some variance
        
        self.assertGreaterEqual(result['compression_ratio'], expected_min_ratio, 
                               f"Compression ratio {result['compression_ratio']} is below expected minimum {expected_min_ratio}")
        self.assertLessEqual(result['compression_ratio'], expected_max_ratio, 
                            f"Compression ratio {result['compression_ratio']} is above expected maximum {expected_max_ratio}")
        
        # Verify that important segments were preserved
        self.assertGreaterEqual(len(result['preserved_segments']), 1, 
                               "No important segments were preserved")
        
        # Verify importance scores are consistent
        for segment in result['preserved_segments']:
            self.assertGreaterEqual(segment['importance_score'], 0.5, 
                                  "Low importance segment was preserved")
    
    def test_extreme_acceleration(self):
        """
        Test compression with extreme acceleration factors.
        
        This test verifies that the compressor can handle very high acceleration
        factors (up to 30x) while still preserving the most critical content.
        
        Expected outcome:
        - Compressed video is created even at extreme acceleration
        - Most important content is still preserved
        - Duration is reduced proportionally to acceleration factor
        """
        video_path = self.test_videos['lecture']
        if not os.path.getsize(video_path) > 0:
            self.skipTest("Test video is empty or not available")
        
        output_path = os.path.join(self.test_dir, 'extreme_compressed_lecture.mp4')
        
        # Set extreme acceleration
        target_acceleration = 30.0
        
        # Compress the video
        result = self.compressor.compress(
            video_path, 
            output_path=output_path,
            target_acceleration=target_acceleration
        )
        
        # Verify the compressed video exists
        self.assertTrue(os.path.exists(output_path), "Compressed video was not created")
        
        # Verify the compression is significant but still contains the most important content
        self.assertLessEqual(result['compression_ratio'], 0.1, 
                           "Extreme compression did not achieve significant reduction")
        
        # Verify top segments were preserved
        highest_importance = max([s['importance_score'] for s in result['preserved_segments']])
        self.assertGreaterEqual(highest_importance, 0.8, 
                              "Most important segments were not preserved")
        
        # Verify duration reduction is proportional to acceleration factor
        expected_duration_ratio = 1.0 / target_acceleration
        actual_duration_ratio = result['output_duration'] / result['input_duration']
        
        # Allow for some variance in duration reduction
        tolerance = 0.5  # 50% tolerance for extreme compression
        self.assertAlmostEqual(
            actual_duration_ratio, 
            expected_duration_ratio, 
            delta=expected_duration_ratio * tolerance,
            msg=f"Duration reduction is not proportional to acceleration factor"
        )
    
    def test_different_video_types(self):
        """
        Test compression behavior across different types of videos.
        
        This test verifies that the semantic compressor adapts appropriately
        to different video content types (tutorial, lecture, demonstration, etc.)
        
        Expected outcome:
        - All video types are compressed successfully
        - Compression characteristics vary appropriately by content type
        - Important content is preserved for each type
        """
        # Test a subset of video types
        for video_type in ['tutorial', 'demo', 'lecture', 'conversation']:
            video_path = self.test_videos[video_type]
            if not os.path.getsize(video_path) > 0:
                print(f"Skipping {video_type} test as video is not available")
                continue
                
            output_path = os.path.join(self.test_dir, f'compressed_{video_type}.mp4')
            
            # Compress the video
            result = self.compressor.compress(
                video_path, 
                output_path=output_path,
                target_acceleration=5.0
            )
            
            # Verify the compressed video exists
            self.assertTrue(os.path.exists(output_path), 
                           f"Compressed video was not created for {video_type}")
            
            # Verify that compression occurred
            self.assertLess(result['output_duration'], result['input_duration'], 
                           f"No compression occurred for {video_type}")
            
            # Verify that important segments were preserved
            self.assertGreaterEqual(len(result['preserved_segments']), 1, 
                                   f"No important segments were preserved for {video_type}")
            
            # Different video types should have different segment preservation patterns
            if video_type == 'tutorial':
                # Tutorials should preserve UI interaction segments
                ui_interactions = sum(1 for s in result['preserved_segments'] 
                                    if s.get('content_type') == 'ui_interaction')
                self.assertGreaterEqual(ui_interactions, 1, 
                                     "UI interactions not preserved in tutorial")
                
            elif video_type == 'lecture':
                # Lectures should preserve key concept segments
                key_concepts = sum(1 for s in result['preserved_segments'] 
                                 if s.get('content_type') == 'key_concept')
                self.assertGreaterEqual(key_concepts, 1, 
                                     "Key concepts not preserved in lecture")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_vs_cpu_performance(self):
        """
        Test performance difference between GPU and CPU implementations.
        
        This test measures the processing speed difference between GPU and CPU
        for the semantic compression pipeline. This test is skipped if CUDA is
        not available.
        
        Expected outcome:
        - Both GPU and CPU implementations produce similar results
        - GPU implementation is significantly faster (at least 2x)
        """
        video_path = self.test_videos['tutorial']
        if not os.path.getsize(video_path) > 0:
            self.skipTest("Test video is empty or not available")
        
        # First test with CPU
        self.compressor.use_gpu = False
        start_time = time.time()
        cpu_result = self.compressor.compress(
            video_path,
            output_path=os.path.join(self.test_dir, 'cpu_compressed.mp4'),
            target_acceleration=5.0
        )
        cpu_time = time.time() - start_time
        
        # Then test with GPU
        self.compressor.use_gpu = True
        start_time = time.time()
        gpu_result = self.compressor.compress(
            video_path,
            output_path=os.path.join(self.test_dir, 'gpu_compressed.mp4'),
            target_acceleration=5.0
        )
        gpu_time = time.time() - start_time
        
        # Verify both implementations produce similar results
        self.assertAlmostEqual(
            cpu_result['compression_ratio'], 
            gpu_result['compression_ratio'], 
            delta=0.1,
            msg="CPU and GPU implementations produced significantly different compression ratios"
        )
        
        # Verify the GPU implementation is faster
        self.assertLess(gpu_time, cpu_time, "GPU implementation was not faster than CPU")
        
        # GPU should be at least 2x faster
        speedup = cpu_time / gpu_time
        self.assertGreaterEqual(
            speedup, 2.0, 
            f"GPU speedup ({speedup:.2f}x) was less than expected (2.0x)"
        )
        
        print(f"GPU speedup: {speedup:.2f}x faster than CPU")
    
    def test_edge_cases(self):
        """
        Test handling of edge cases and unusual inputs.
        
        This test verifies that the semantic compressor handles various edge cases
        correctly, including very short videos, silent videos, and extreme resolutions.
        
        Expected outcome:
        - Compressor handles all edge cases without errors
        - Appropriate fallback mechanisms are used when needed
        - Results are reasonable given the input limitations
        """
        # Test cases to check
        edge_cases = [
            ('short', "Very short video"),
            ('silent', "Video without audio"),
            ('low_res', "Low resolution video"),
            ('high_res', "High resolution video"),
        ]

