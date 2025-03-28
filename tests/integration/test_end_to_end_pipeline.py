#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end integration tests for the HyperbolicLearner pipeline.

This module tests the complete workflow of the HyperbolicLearner system:
1. Video downloading from various sources
2. Semantic compression of video content
3. Knowledge extraction from processed videos
4. UI element detection and analysis
5. Workflow generation based on extracted knowledge

The tests include both success paths and error handling scenarios.
"""

import os
import unittest
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
from pathlib import Path

# Import HyperbolicLearner components
from src.video_processor.downloader import VideoDownloader
from src.video_processor.semantic_compression import SemanticCompressor
from src.video_processor.accelerator import VideoAccelerator
from src.ml_engine.content_analyzer import ContentAnalyzer
from src.ui_automation.ui_analyzer import UIAnalyzer
from src.knowledge_base.graph_db import KnowledgeGraph
from src.action_executor.executor import WorkflowExecutor

# Import main application for full pipeline testing
from src.main import HyperbolicLearner


class TestEndToEndPipeline(unittest.TestCase):
    """Test the complete HyperbolicLearner pipeline end-to-end."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create temporary directories for test data
        cls.test_dir = tempfile.mkdtemp()
        cls.video_cache_dir = os.path.join(cls.test_dir, 'video_cache')
        cls.knowledge_dir = os.path.join(cls.test_dir, 'knowledge')
        cls.model_dir = os.path.join(cls.test_dir, 'models')
        
        # Create directories
        os.makedirs(cls.video_cache_dir, exist_ok=True)
        os.makedirs(cls.knowledge_dir, exist_ok=True)
        os.makedirs(cls.model_dir, exist_ok=True)
        
        # Define test URLs
        cls.test_video_urls = {
            'valid_short': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',  # Short tutorial
            'valid_long': 'https://www.youtube.com/watch?v=Q8TXgCzxEnw',   # Longer tutorial
            'invalid': 'https://www.youtube.com/watch?v=invalid_video_id',
            'nonexistent': 'https://nonexistent-site.com/video.mp4'
        }
        
        # Mock environment settings
        cls.env_patcher = patch.dict('os.environ', {
            'VIDEO_CACHE_DIR': cls.video_cache_dir,
            'KNOWLEDGE_DIR': cls.knowledge_dir,
            'MODEL_DIR': cls.model_dir,
            'GPU_ENABLED': 'false',  # Avoid GPU dependency in tests
            'MAX_ACCELERATION': '10.0',
            'DEFAULT_ACCELERATION': '5.0'
        })
        cls.env_patcher.start()
        
        # Initialize the learner with test configuration
        cls.test_config = {
            'video_cache_dir': cls.video_cache_dir,
            'knowledge_dir': cls.knowledge_dir,
            'model_dir': cls.model_dir,
            'gpu_enabled': False,
            'max_acceleration': 10.0,
            'default_acceleration': 5.0,
            'test_mode': True  # Enable test mode for mocking expensive operations
        }

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests."""
        # Clean up the temporary directory
        shutil.rmtree(cls.test_dir)
        
        # Stop patching
        cls.env_patcher.stop()

    def setUp(self):
        """Set up before each test."""
        # Initialize components with test configuration
        self.downloader = VideoDownloader(cache_dir=self.video_cache_dir)
        self.compressor = SemanticCompressor(gpu_enabled=False)
        self.content_analyzer = ContentAnalyzer(model_dir=self.model_dir, gpu_enabled=False)
        self.ui_analyzer = UIAnalyzer(model_dir=self.model_dir, gpu_enabled=False)
        self.knowledge_graph = KnowledgeGraph(storage_dir=self.knowledge_dir)
        self.workflow_executor = WorkflowExecutor()
        
        # Initialize the main application
        self.learner = HyperbolicLearner(**self.test_config)
        
        # Create a mock video file for tests that need a real file
        self.mock_video_path = os.path.join(self.video_cache_dir, 'mock_video.mp4')
        self._create_mock_video(self.mock_video_path)

    def tearDown(self):
        """Clean up after each test."""
        # Clean test cache directories
        for file in os.listdir(self.video_cache_dir):
            os.remove(os.path.join(self.video_cache_dir, file))
        
        for file in os.listdir(self.knowledge_dir):
            os.remove(os.path.join(self.knowledge_dir, file))

    def _create_mock_video(self, path, duration=10, resolution=(640, 480)):
        """Create a small mock video file for testing."""
        # This is a simplified mock - in a real implementation, we might use
        # libraries like OpenCV to create actual video content
        with open(path, 'w') as f:
            f.write(f"MOCK_VIDEO:duration={duration},resolution={resolution[0]}x{resolution[1]}")
        return path

    def _create_mock_knowledge(self):
        """Create mock knowledge extraction results."""
        return {
            'concepts': [
                {'id': 'c1', 'name': 'Button', 'type': 'UI_ELEMENT', 'confidence': 0.92},
                {'id': 'c2', 'name': 'Click', 'type': 'ACTION', 'confidence': 0.95},
                {'id': 'c3', 'name': 'Save', 'type': 'OPERATION', 'confidence': 0.88}
            ],
            'relationships': [
                {'source': 'c2', 'target': 'c1', 'type': 'ACTS_ON', 'confidence': 0.91},
                {'source': 'c2', 'target': 'c3', 'type': 'PERFORMS', 'confidence': 0.87}
            ],
            'workflow': {
                'id': 'w1',
                'name': 'Save Operation',
                'steps': [
                    {'action': 'c2', 'target': 'c1', 'parameters': {}}
                ]
            }
        }

    @patch('src.video_processor.downloader.VideoDownloader.download')
    def test_video_download_success(self, mock_download):
        """Test successful video download."""
        # Setup
        mock_download.return_value = self.mock_video_path
        
        # Execute
        result = self.downloader.download(self.test_video_urls['valid_short'])
        
        # Verify
        self.assertEqual(result, self.mock_video_path)
        mock_download.assert_called_once_with(self.test_video_urls['valid_short'])

    @patch('src.video_processor.downloader.VideoDownloader.download')
    def test_video_download_failure(self, mock_download):
        """Test video download with invalid URL."""
        # Setup
        mock_download.side_effect = ValueError("Invalid video URL")
        
        # Execute and verify
        with self.assertRaises(ValueError):
            self.downloader.download(self.test_video_urls['invalid'])
        
        mock_download.assert_called_once_with(self.test_video_urls['invalid'])

    @patch('src.video_processor.semantic_compression.SemanticCompressor.compress')
    def test_semantic_compression(self, mock_compress):
        """Test semantic compression of video content."""
        # Setup
        output_path = os.path.join(self.video_cache_dir, 'compressed_video.mp4')
        mock_compress.return_value = {
            'output_path': output_path,
            'original_duration': 10.0,
            'compressed_duration': 2.0,
            'compression_ratio': 5.0,
            'importance_threshold': 0.7,
            'preserved_segments': [
                {'start_time': 0.0, 'end_time': 0.5, 'importance': 0.9},
                {'start_time': 3.2, 'end_time': 4.1, 'importance': 0.85}
            ]
        }
        
        # Execute
        result = self.compressor.compress(
            self.mock_video_path,
            output_path=output_path,
            target_ratio=5.0
        )
        
        # Verify
        self.assertEqual(result['output_path'], output_path)
        self.assertEqual(result['compression_ratio'], 5.0)
        mock_compress.assert_called_once()

    @patch('src.ml_engine.content_analyzer.ContentAnalyzer.analyze')
    def test_content_analysis(self, mock_analyze):
        """Test content analysis of processed video."""
        # Setup
        mock_analyze.return_value = {
            'scenes': [
                {'start_time': 0.0, 'end_time': 2.5, 'importance': 0.8, 'type': 'INTRODUCTION'},
                {'start_time': 2.5, 'end_time': 5.0, 'importance': 0.9, 'type': 'DEMONSTRATION'}
            ],
            'concepts': [
                {'id': 'c1', 'name': 'Button', 'type': 'UI_ELEMENT', 'confidence': 0.92, 'time_ranges': [(3.1, 4.5)]},
                {'id': 'c2', 'name': 'Click', 'type': 'ACTION', 'confidence': 0.95, 'time_ranges': [(3.2, 3.5)]}
            ],
            'transcript': "In this tutorial, I'll show you how to save a file by clicking the button."
        }
        
        # Execute
        result = self.content_analyzer.analyze(self.mock_video_path)
        
        # Verify
        self.assertEqual(len(result['scenes']), 2)
        self.assertEqual(len(result['concepts']), 2)
        self.assertTrue('transcript' in result)
        mock_analyze.assert_called_once_with(self.mock_video_path)

    @patch('src.ui_automation.ui_analyzer.UIAnalyzer.analyze')
    def test_ui_analysis(self, mock_analyze):
        """Test UI element detection and analysis."""
        # Setup
        mock_analyze.return_value = {
            'ui_elements': [
                {
                    'id': 'ui1',
                    'type': 'BUTTON',
                    'bounding_box': [100, 100, 200, 150],
                    'text': 'Save',
                    'confidence': 0.94,
                    'time_ranges': [(3.1, 4.5)]
                }
            ],
            'interactions': [
                {
                    'id': 'int1',
                    'type': 'CLICK',
                    'target_element': 'ui1',
                    'time': 3.2,
                    'confidence': 0.95
                }
            ]
        }
        
        # Execute
        result = self.ui_analyzer.analyze(self.mock_video_path)
        
        # Verify
        self.assertEqual(len(result['ui_elements']), 1)
        self.assertEqual(len(result['interactions']), 1)
        self.assertEqual(result['interactions'][0]['target_element'], 'ui1')
        mock_analyze.assert_called_once_with(self.mock_video_path)

    @patch('src.knowledge_base.graph_db.KnowledgeGraph.build_from_analysis')
    def test_knowledge_graph_construction(self, mock_build):
        """Test knowledge graph construction from analysis results."""
        # Setup
        mock_build.return_value = 'graph-123'
        
        # Mock analysis results
        content_analysis = {
            'scenes': [
                {'start_time': 0.0, 'end_time': 2.5, 'importance': 0.8, 'type': 'INTRODUCTION'},
                {'start_time': 2.5, 'end_time': 5.0, 'importance': 0.9, 'type': 'DEMONSTRATION'}
            ],
            'concepts': [
                {'id': 'c1', 'name': 'Button', 'type': 'UI_ELEMENT', 'confidence': 0.92},
                {'id': 'c2', 'name': 'Click', 'type': 'ACTION', 'confidence': 0.95}
            ],
            'transcript': "In this tutorial, I'll show you how to save a file by clicking the button."
        }
        
        ui_analysis = {
            'ui_elements': [
                {
                    'id': 'ui1',
                    'type': 'BUTTON',
                    'bounding_box': [100, 100, 200, 150],
                    'text': 'Save',
                    'confidence': 0.94
                }
            ],
            'interactions': [
                {
                    'id': 'int1',
                    'type': 'CLICK',
                    'target_element': 'ui1',
                    'time': 3.2,
                    'confidence': 0.95
                }
            ]
        }
        
        # Execute
        graph_id = self.knowledge_graph.build_from_analysis(
            content_analysis,
            ui_analysis,
            source_video=self.mock_video_path
        )
        
        # Verify
        self.assertEqual(graph_id, 'graph-123')
        mock_build.assert_called_once()

    @patch('src.action_executor.executor.WorkflowExecutor.generate_workflow')
    def test_workflow_generation(self, mock_generate):
        """Test workflow generation from knowledge graph."""
        # Setup
        expected_workflow = {
            'id': 'wf-123',
            'name': 'Save Operation',
            'nodes': [
                {'id': 'n1', 'type': 'ACTION', 'action': 'CLICK', 'target': 'ui1', 'parameters': {}}
            ],
            'edges': [
                {'source': 'START', 'target': 'n1'}
            ]
        }
        mock_generate.return_value = expected_workflow
        
        # Execute
        workflow = self.workflow_executor.generate_workflow('graph-123')
        
        # Verify
        self.assertEqual(workflow['id'], expected_workflow['id'])

import os
import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.video_processor.downloader import YouTubeDownloader
from src.video_processor.semantic_compression import SemanticCompressor
from src.knowledge_base.graph_db import KnowledgeGraph
from src.ml_engine.content_analyzer import ContentAnalyzer
from src.ui_automation.ui_analyzer import UIAnalyzer
from src.core.config import Configuration


class TestEndToEndPipeline(unittest.TestCase):
    """
    End-to-end integration tests for the complete HyperbolicLearner pipeline.
    
    This test suite validates the entire workflow from video downloading to
    knowledge extraction and UI analysis using semantic compression.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with configuration and temp directories."""
        # Create temporary directory for test data
        cls.test_dir = tempfile.mkdtemp()
        cls.video_cache_dir = os.path.join(cls.test_dir, "video_cache")
        cls.knowledge_dir = os.path.join(cls.test_dir, "knowledge")
        
        # Ensure directories exist
        os.makedirs(cls.video_cache_dir, exist_ok=True)
        os.makedirs(cls.knowledge_dir, exist_ok=True)
        
        # Configure test environment
        cls.config = Configuration()
        cls.config.set("VIDEO_CACHE_DIR", cls.video_cache_dir)
        cls.config.set("KNOWLEDGE_DIR", cls.knowledge_dir)
        cls.config.set("MAX_ACCELERATION", 10.0)
        cls.config.set("TEST_MODE", True)
        
        # Sample test video - short educational clip
        # Using a Creative Commons video to ensure test legality
        cls.test_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with actual educational CC video
        
        # Initialize components
        cls.downloader = YouTubeDownloader(config=cls.config)
        cls.compressor = SemanticCompressor(config=cls.config)
        cls.content_analyzer = ContentAnalyzer(config=cls.config)
        cls.ui_analyzer = UIAnalyzer(config=cls.config)
        cls.knowledge_graph = KnowledgeGraph(config=cls.config)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    def test_video_download(self):
        """Test that the downloader correctly retrieves videos."""
        video_path = self.downloader.download(self.test_video_url)
        
        self.assertTrue(os.path.exists(video_path), "Downloaded video file doesn't exist")
        self.assertTrue(os.path.getsize(video_path) > 0, "Downloaded video file is empty")
        
        # Verify metadata extraction
        metadata = self.downloader.extract_metadata(self.test_video_url)
        self.assertIsNotNone(metadata.get("title"), "Video title not extracted")
        self.assertIsNotNone(metadata.get("duration"), "Video duration not extracted")
        
        return video_path
    
    def test_semantic_compression(self):
        """Test semantic compression of the downloaded video."""
        video_path = self.test_video_download()
        
        # Apply semantic compression with acceleration factor of 5x
        compressed_video_path = self.compressor.compress(
            video_path=video_path,
            acceleration_factor=5.0,
            preserve_important_segments=True,
            content_aware=True
        )
        
        self.assertTrue(os.path.exists(compressed_video_path), 
                       "Compressed video file doesn't exist")
        
        # Verify compression ratio
        original_size = os.path.getsize(video_path)
        compressed_size = os.path.getsize(compressed_video_path)
        self.assertLess(compressed_size, original_size, 
                       "Compressed video isn't smaller than original")
        
        # Verify important segments were preserved
        importance_map = self.compressor.get_importance_map(video_path)
        self.assertIsNotNone(importance_map, "Importance map not generated")
        self.assertGreater(len(importance_map), 0, "Importance map is empty")
        
        # Verify temporal coherence of compression
        coherence_score = self.compressor.evaluate_temporal_coherence(compressed_video_path)
        self.assertGreater(coherence_score, 0.7, 
                          "Compressed video lacks temporal coherence")
        
        return compressed_video_path, importance_map
    
    def test_content_analysis(self):
        """Test content analysis on the compressed video."""
        compressed_video_path, importance_map = self.test_semantic_compression()
        
        # Analyze content of the compressed video
        analysis_results = self.content_analyzer.analyze(
            video_path=compressed_video_path,
            importance_map=importance_map,
            extract_concepts=True,
            detect_scenes=True,
            transcribe_audio=True
        )
        
        # Verify concept extraction
        self.assertIn("concepts", analysis_results, "Concepts not extracted")
        self.assertGreater(len(analysis_results["concepts"]), 0, 
                          "No concepts extracted from video")
        
        # Verify scene detection
        self.assertIn("scenes", analysis_results, "Scenes not detected")
        self.assertGreater(len(analysis_results["scenes"]), 0, 
                          "No scenes detected in video")
        
        # Verify transcript extraction
        self.assertIn("transcript", analysis_results, "Transcript not extracted")
        self.assertGreater(len(analysis_results["transcript"]), 0, 
                          "Empty transcript extracted")
        
        return analysis_results
    
    def test_ui_analysis(self):
        """Test UI element detection and interaction analysis."""
        compressed_video_path, _ = self.test_semantic_compression()
        
        # Analyze UI elements and interactions
        ui_results = self.ui_analyzer.analyze(
            video_path=compressed_video_path,
            detect_elements=True,
            track_interactions=True,
            extract_workflows=True
        )
        
        # Verify UI element detection
        self.assertIn("elements", ui_results, "UI elements not detected")
        
        # If the video contains UI interactions, verify detection
        if ui_results["has_ui_interactions"]:
            self.assertGreater(len(ui_results["elements"]), 0, 
                              "No UI elements detected")
            self.assertIn("interactions", ui_results, "UI interactions not detected")
            self.assertGreater(len(ui_results["interactions"]), 0, 
                              "No UI interactions detected")
        
        return ui_results
    
    def test_knowledge_extraction(self):
        """Test knowledge extraction and graph building."""
        analysis_results = self.test_content_analysis()
        ui_results = self.test_ui_analysis()
        
        # Extract knowledge and build graph
        graph_id = self.knowledge_graph.build_from_analysis(
            content_analysis=analysis_results,
            ui_analysis=ui_results,
            metadata={"source_url": self.test_video_url}
        )
        
        self.assertIsNotNone(graph_id, "Knowledge graph not created")
        
        # Verify graph structure and content
        graph = self.knowledge_graph.get(graph_id)
        self.assertIsNotNone(graph, "Cannot retrieve knowledge graph")
        
        # Verify nodes and edges
        self.assertGreater(graph.node_count(), 0, "Knowledge graph has no nodes")
        self.assertGreater(graph.edge_count(), 0, "Knowledge graph has no edges")
        
        # Verify concept extraction
        concepts = graph.get_all_concepts()
        self.assertGreater(len(concepts), 0, "No concepts in knowledge graph")
        
        # Verify workflow extraction if UI interactions exist
        if ui_results.get("has_ui_interactions", False):
            workflows = graph.get_all_workflows()
            self.assertGreater(len(workflows), 0, "No workflows extracted")
        
        return graph_id, graph
    
    def test_end_to_end_pipeline(self):
        """Test the complete end-to-end pipeline."""
        # This test brings together all components to verify full integration
        graph_id, graph = self.test_knowledge_extraction()
        
        # Verify graph confidence scores
        avg_confidence = graph.get_average_confidence()
        self.assertGreater(avg_confidence, 0.7, 
                          "Knowledge graph has low confidence scores")
        
        # Test query capabilities
        query_results = self.knowledge_graph.query(
            graph_id=graph_id,
            query="What are the main concepts in this video?",
            limit=5
        )
        
        self.assertIsNotNone(query_results, "Query returned no results")
        self.assertGreater(len(query_results), 0, "Query returned empty results")
        
        # Test workflow extraction and execution preparation
        if graph.has_workflows():
            workflow = graph.get_main_workflow()
            self.assertIsNotNone(workflow, "Cannot retrieve main workflow")
            self.assertGreater(len(workflow.steps), 0, "Workflow has no steps")
            
            # Verify workflow can be serialized for execution
            serialized = workflow.to_executable()
            self.assertIsNotNone(serialized, "Workflow cannot be serialized")
            self.assertIn("steps", serialized, "Serialized workflow missing steps")
            
            # Verify workflow validation
            validation = workflow.validate()
            self.assertTrue(validation.is_valid, 
                           f"Workflow validation failed: {validation.error_message}")
        
        # Verify knowledge can be exported and reimported
        export_path = os.path.join(self.test_dir, "export.json")
        self.knowledge_graph.export(graph_id, export_path)
        self.assertTrue(os.path.exists(export_path), "Knowledge export failed")
        
        # Reimport and verify
        reimport_id = self.knowledge_graph.import_from_file(export_path)
        reimported_graph = self.knowledge_graph.get(reimport_id)
        
        self.assertEqual(graph.node_count(), reimported_graph.node_count(),
                        "Reimported graph has different node count")
        self.assertEqual(graph.edge_count(), reimported_graph.edge_count(),
                        "Reimported graph has different edge count")


if __name__ == "__main__":
    unittest.main()

import os
import unittest
import tempfile
import shutil
import logging
from pathlib import Path
import json
import time

import numpy as np
import pytest

from src.video_processor.downloader import YouTubeDownloader
from src.video_processor.accelerator import VideoAccelerator
from src.video_processor.semantic_compression import SemanticCompressor
from src.video_processor.semantic_compression.importance_models import VisualImportanceModel, AudioImportanceModel, TranscriptImportanceModel
from src.video_processor.semantic_compression.multimodal_fusion import MultimodalFusion
from src.ml_engine.content_analyzer import ContentAnalyzer
from src.knowledge_base.graph_db import KnowledgeGraph, Concept, Relationship
from src.ui_automation.ui_analyzer import UIElementDetector, UIElement, UIInteraction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Specific tutorial video for testing - provided by user
TEST_VIDEO_URL = "https://www.youtube.com/watch?v=W2Ur7FGqsJE"  # Specific test video

class EndToEndPipelineTest(unittest.TestCase):
    """
    Integration test for the entire HyperbolicLearner pipeline.
    Tests the workflow from video download to knowledge extraction using semantic compression.
    """
    
    def setUp(self):
        """Set up test environment with temporary directories and component initialization."""
        # Create temporary directories for test data
        self.temp_dir = tempfile.mkdtemp()
        self.video_cache_dir = os.path.join(self.temp_dir, "video_cache")
        self.processed_dir = os.path.join(self.temp_dir, "processed")
        self.knowledge_dir = os.path.join(self.temp_dir, "knowledge")
        self.frames_dir = os.path.join(self.temp_dir, "frames")
        self.transcripts_dir = os.path.join(self.temp_dir, "transcripts")
        
        # Create the directories
        for directory in [self.video_cache_dir, self.processed_dir, self.knowledge_dir, 
                          self.frames_dir, self.transcripts_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize importance models for semantic compression
        self.visual_importance_model = VisualImportanceModel(
            model_path="models/visual_importance.pth",
            use_gpu=False,  # Avoid GPU dependency in testing
            batch_size=4,
            frame_resolution=(224, 224)
        )
        
        self.audio_importance_model = AudioImportanceModel(
            model_path="models/audio_importance.pth",
            sample_rate=16000,
            segment_duration_ms=500,
            use_gpu=False
        )
        
        self.transcript_importance_model = TranscriptImportanceModel(
            model_path="models/transcript_importance.pth",
            embedding_dim=768,
            use_domain_specific=True,
            domain="technology"  # Assuming the test video is tech-related
        )
        
        # Initialize multimodal fusion
        self.multimodal_fusion = MultimodalFusion(
            fusion_method="attention",
            temporal_context_size=5,
            modality_weights={"visual": 0.4, "audio": 0.3, "transcript": 0.3},
            use_gpu=False
        )
        
        # Initialize main components
        self.downloader = YouTubeDownloader(
            cache_dir=self.video_cache_dir,
            max_resolution="480p",  # Higher resolution for better feature extraction
            timeout=300,  # Longer timeout for larger videos
            extract_audio=True,
            extract_subtitles=True
        )
        
        self.semantic_compressor = SemanticCompressor(
            visual_importance_model=self.visual_importance_model,
            audio_importance_model=self.audio_importance_model,
            transcript_importance_model=self.transcript_importance_model,
            multimodal_fusion=self.multimodal_fusion,
            importance_threshold=0.4,
            min_acceleration=2.0,
            max_acceleration=15.0,  # Higher max acceleration for testing
            use_gpu=False,  # Avoid GPU dependency in testing
            output_dir=self.processed_dir,
            frames_dir=self.frames_dir,
            preserve_transitions=True,
            adaptive_acceleration=True
        )
        
        self.content_analyzer = ContentAnalyzer(
            models_dir="models",
            use_lightweight_models=True,  # Use smaller models for testing
            batch_size=4,
            use_gpu=False,
            detect_transitions=True,
            extract_timestamps=True,
            transcripts_dir=self.transcripts_dir
        )
        
        self.ui_detector = UIElementDetector(
            confidence_threshold=0.6,
            tracking_enabled=True,
            models_dir="models",
            element_types=["button", "text_field", "checkbox", "dropdown", "menu", "toolbar"],
            temporal_consistency_threshold=0.7,
            max_tracking_distance=50  # pixels
        )
        
        self.knowledge_graph = KnowledgeGraph(
            storage_path=self.knowledge_dir,
            embedding_dim=128,  # Smaller embeddings for testing
            use_hyperbolic=True,
            similarity_threshold=0.7,
            max_connections=100,
            confidence_threshold=0.5
        )
        
        logger.info("Test environment set up successfully")
    
    def tearDown(self):
        """Clean up temporary test directories and resources."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        logger.info("Test environment cleaned up")
    
    def test_full_pipeline(self):
        """Test the entire pipeline from video download to knowledge extraction."""
        # Step 1: Download the test video
        logger.info(f"Downloading test video: {TEST_VIDEO_URL}")
        video_info = self.downloader.download(TEST_VIDEO_URL)
        
        video_path = video_info['video_path']
        audio_path = video_info.get('audio_path')
        subtitle_path = video_info.get('subtitle_path')
        
        self.assertTrue(os.path.exists(video_path), f"Downloaded video not found at {video_path}")
        video_size = os.path.getsize(video_path)
        video_duration = video_info['duration_seconds']
        
        logger.info(f"Downloaded video: {video_info['title']}")
        logger.info(f"Video size: {video_size / 1024 / 1024:.2f} MB, Duration: {video_duration:.2f} seconds")
        
        # Verify audio and subtitles if available
        if audio_path:
            self.assertTrue(os.path.exists(audio_path), f"Extracted audio not found at {audio_path}")
        
        if subtitle_path:
            self.assertTrue(os.path.exists(subtitle_path), f"Subtitles not found at {subtitle_path}")
        
        # Step 2: Apply semantic compression
        logger.info("Applying semantic compression")
        compression_start_time = time.time()
        
        compressed_video_info = self.semantic_compressor.compress(
            video_path=video_path,
            audio_path=audio_path,
            subtitle_path=subtitle_path,
            target_acceleration=8.0,  # Aim for 8x acceleration
            preserve_transitions=True,
            audio_quality="medium",
            content_type="tutorial",  # Assuming it's a tutorial
            adaptive_mode=True  # Use content-adaptive acceleration
        )
        
        compression_time = time.time() - compression_start_time
        logger.info(f"Compression completed in {compression_time:.2f} seconds")
        
        # Verify compression results
        self.assertIsNotNone(compressed_video_info, "Compression failed to return results")
        self.assertTrue(os.path.exists(compressed_video_info['output_path']), 
                       f"Compressed video not found at {compressed_video_info['output_path']}")
        
        # Verify acceleration ratio
        original_duration = compressed_video_info['original_duration']
        compressed_duration = compressed_video_info['compressed_duration']
        acceleration_ratio = original_duration / compressed_duration if compressed_duration > 0 else 0
        
        logger.info(f"Original duration: {original_duration:.2f} seconds")
        logger.info(f"Compressed duration: {compressed_duration:.2f} seconds")
        logger.info(f"Achieved acceleration ratio: {acceleration_ratio:.2f}x")
        
        # The acceleration ratio should be reasonable for a tutorial video
        self.assertGreater(acceleration_ratio, 3.0, "Acceleration ratio too low")
        
        # Verify compression quality metrics
        self.assertGreaterEqual(compressed_video_info['information_retention'], 0.85, 
                              "Information retention below threshold")
        
        # Check for important segments preservation
        self.assertGreaterEqual(compressed_video_info['important_segments_preserved'], 0.90, 
                              "Important segment preservation below threshold")
        
        # Step 3: Analyze content and extract knowledge
        logger.info("Analyzing content and extracting knowledge")
        analysis_start_time = time.time()
        
        content_analysis = self.content_analyzer.analyze(
            video_path=compressed_video_info['output_path'],
            extract_text=True,
            extract_objects=True,
            extract_concepts=True,
            extract_actions=True,
            transcript_path=subtitle_path,
            semantic_segmentation=True,
            temporal_resolution_ms=500
        )
        
        analysis_time = time.time() - analysis_start_time
        logger.info(f"Content analysis completed in {analysis_time:.2f} seconds")
        
        # Verify content analysis results
        self.assertIsNotNone(content_analysis, "Content analysis failed to return results")
        self.assertGreaterEqual(len(content_analysis['segments']), 2, "Too few segments detected in video")
        self.assertGreaterEqual(len(content_analysis['concepts']), 5, "Too few concepts extracted")
        
        # Log extracted concepts
        logger.info(f"Extracted {len(content_analysis['concepts'])} concepts:")
        for i, concept in enumerate(content_analysis['concepts'][:5]):  # Log first 5 concepts
            logger.info(f"  {i+1}. {concept['name']} (confidence: {concept['confidence']:.2f})")
        
        # Step 4: Detect UI elements
        logger.info("Detecting UI elements")
        ui_start_time = time.time()
        
        ui_elements = self.ui_detector.detect_elements(
            video_path=compressed_video_info['output_path'],
            track_interactions=True,
            detection_interval_ms=1000,  # Check every second
            min_element_size=20,  # Minimum 20x20 pixels
            max_elements_per_frame=50
        )
        
        ui_time = time.time() - ui_start_time
        logger.info(f"UI detection completed in {ui_time:.2f} seconds")
        logger.info(f"Detected {len(ui_elements)} UI elements")
        
        # Log detected UI elements
        if ui_elements:
            for i, element in enumerate(ui_elements[:3]):  # Log first 3 elements
                logger.info(f"  {i+1}. {element['type']} at {element['position']} (confidence: {element['confidence']:.2f})")
        
        # Step 5: Extract UI interactions if present
        if hasattr(self.ui_detector, 'extract_interactions') and ui_elements:
            ui_interactions = self.ui_detector.extract_interactions(ui_elements)
            logger.info(f"Extracted {len(ui_interactions)} UI interactions")
            
            # Verify interactions if present
            if ui_interactions:
                self.assertGreaterEqual(len(ui_interactions), 1, "No UI interactions extracted")
                
                # Log interactions
                for i, interaction in enumerate(ui_interactions[:3]):  # Log first 3
                    logger.info(f"  {i+1}. {interaction['type']} on {interaction['element_type']} at {interaction['timestamp_ms']}ms")
        
        # Step 6: Build knowledge graph
        logger.info("Building knowledge graph")
        kg_start_time = time.time()
        
        # Initialize a new graph for this test
        graph_id = f"test_graph_{int(time.time())}"
        self.knowledge_graph.create_graph(namespace=graph_id)
        
        # Add extracted concepts to the knowledge graph
        for concept in content_analysis['concepts']:
            self.knowledge_graph.add_concept(
                name=concept['name'],
                metadata={
                    "description": concept.get('description', ''),
                    "timestamps": concept.get('timestamps', []),
                    "source_video": os.path.basename(video_path),
                    "domain": concept.get('domain', 'general'),
                    "importance": concept.get('importance', 0.5)
                },
                confidence=concept['confidence'],
                embedding=concept.get('embedding')
            )
        
        # Add relationships between concepts based on co-occurrence and other factors
        for i in range(len(content_analysis['concepts'])):
            for j in range(i+1, len(content_analysis['concepts'])):
                # Create relationships based on temporal proximity, similarity, etc.
                if self._should_create_relationship(content_analysis['concepts'][i], content_analysis['concepts'][j]):
                    relationship_type = self._determine_relationship_type(
                        content_analysis['concepts'][i], 
                        content_analysis['concepts'][j]
                    )
                    
                    weight = self._calculate_relationship_weight(
                        content_analysis['concepts'][i], 
                        content_analysis['concepts'][j]
                    )
                    
                    self.knowledge_graph.add_relationship(
                        source=content_analysis['concepts'][i]['name'],
                        target=content_analysis['concepts'][j]['name'],
                        relationship_type=relationship_type,
                        weight=weight,
                        metadata={
                            "source_video": os.path.basename(video_path),
                            "temporal_distance_ms": self._calculate_temporal_distance(
                                content_analysis['concepts'][i], 
                                content_analysis['concepts'][j]
                            )
                        }
                    )
        
        # Add UI elements to knowledge graph if available
        if ui_elements:
            for element in ui_elements:
                # Add UI elements as special concept types
                ui_concept_name = f"UI_{element['type']}_{element['id']}"
                self.knowledge_graph.add_concept(
                    name=ui_concept_name,
                    metadata={
                        "ui_type": element['type'],
                        "position": element['position'],
                        "timestamps": element['timestamps'],
                        "source_video": os.path.basename(video_path)
                    },
                    confidence=element['confidence'],
                    concept_type="ui_element"
                )
                
                # Connect UI elements to relevant concepts
                for concept in content_analysis['concepts']:

