#!/usr/bin/env python3
"""
HyperbolicLearner YouTube Learning Example

This example demonstrates how to use the HyperbolicLearner system to:
1. Download a YouTube tutorial video
2. Process the video at accelerated speeds
3. Extract knowledge and concepts from the content
4. Identify and record UI interactions for potential automation

Usage:
    python youtube_learning_example.py --url <youtube_url> [--acceleration <factor>] [--output <directory>]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import traceback
from typing import Dict, List, Optional, Tuple, Any

# Add the project root to the Python path to allow imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from core.config import SystemConfig
    from video_processor.downloader import YouTubeDownloader
    from video_processor.accelerator import VideoAccelerator
    from ui_automation.ui_analyzer import UIInteractionDetector
    from knowledge_base.graph_db import KnowledgeGraph
except ImportError as e:
    print(f"Error importing HyperbolicLearner modules: {e}")
    print("Make sure you've installed all required dependencies.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'hyperbolic_learner.log'))
    ]
)
logger = logging.getLogger('hyperbolic_example')


class HyperbolicLearningExample:
    """
    A practical example class demonstrating the HyperbolicLearner workflow
    for learning from YouTube tutorials at accelerated speeds.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the example with system configuration.
        
        Args:
            config_path: Optional path to a configuration file
        """
        try:
            logger.info("Initializing HyperbolicLearner system...")
            
            # Initialize system configuration with auto-detection of capabilities
            self.config = SystemConfig(config_path=config_path)
            
            # Initialize components with the system configuration
            self.downloader = YouTubeDownloader(self.config)
            self.accelerator = VideoAccelerator(self.config)
            self.ui_detector = UIInteractionDetector(self.config)
            self.knowledge_graph = KnowledgeGraph(self.config)
            
            logger.info("System initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize HyperbolicLearner: {e}")
            raise

    def process_tutorial(self, 
                         video_url: str, 
                         acceleration_factor: float = 5.0,
                         output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a YouTube tutorial through the complete HyperbolicLearner pipeline.
        
        Args:
            video_url: URL of the YouTube tutorial to process
            acceleration_factor: Speed factor for hyperbolic acceleration (default: 5.0)
            output_dir: Directory to save outputs (default: system temp directory)
            
        Returns:
            Dictionary containing the results of processing, including:
            - video_info: Metadata about the downloaded video
            - knowledge_entities: Extracted knowledge concepts
            - ui_interactions: Detected UI interactions
            - acceleration_metrics: Performance metrics for acceleration
        """
        results = {}
        
        try:
            # Step 1: Download the YouTube tutorial
            logger.info(f"Downloading tutorial from {video_url}...")
            video_path, video_info = self._download_video(video_url, output_dir)
            results['video_info'] = video_info
            
            # Step 2: Process the video at accelerated speed
            logger.info(f"Accelerating video at {acceleration_factor}x speed...")
            accelerated_video_path, acceleration_metrics = self._accelerate_video(
                video_path, acceleration_factor
            )
            results['acceleration_metrics'] = acceleration_metrics
            
            # Step 3: Extract knowledge from the video
            logger.info("Extracting knowledge from tutorial content...")
            knowledge_entities = self._extract_knowledge(accelerated_video_path, video_info)
            results['knowledge_entities'] = knowledge_entities
            
            # Step 4: Identify UI interactions
            logger.info("Detecting UI interactions for automation...")
            ui_interactions = self._detect_ui_interactions(accelerated_video_path)
            results['ui_interactions'] = ui_interactions
            
            # Step 5: Store everything in the knowledge graph
            logger.info("Storing extracted information in knowledge graph...")
            graph_id = self._store_in_knowledge_graph(
                video_info, knowledge_entities, ui_interactions
            )
            results['knowledge_graph_id'] = graph_id
            
            logger.info(f"Successfully processed tutorial: {video_info.get('title', video_url)}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing tutorial: {e}")
            logger.debug(traceback.format_exc())
            raise

    def _download_video(self, 
                       video_url: str, 
                       output_dir: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Download a YouTube video using the YouTubeDownloader component.
        
        Args:
            video_url: URL of the YouTube video
            output_dir: Optional directory to save the video
            
        Returns:
            Tuple containing (video_file_path, video_metadata)
        """
        try:
            # Configure download parameters
            download_params = {
                'resolution': 'highest',
                'include_audio': True,
                'output_dir': output_dir or self.config.get('temp_directory'),
                'extract_metadata': True,
                'use_cache': True  # Use cached video if previously downloaded
            }
            
            # Download the video
            video_path, metadata = self.downloader.download(video_url, **download_params)
            
            logger.info(f"Video downloaded: {metadata.get('title')} ({metadata.get('duration')} seconds)")
            logger.info(f"Saved to: {video_path}")
            
            return video_path, metadata
            
        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            raise
    
    def _accelerate_video(self, 
                         video_path: str, 
                         acceleration_factor: float) -> Tuple[str, Dict]:
        """
        Process the video at accelerated speeds while maintaining comprehension.
        
        Args:
            video_path: Path to the downloaded video
            acceleration_factor: Speed multiplier for acceleration
            
        Returns:
            Tuple containing (accelerated_video_path, acceleration_metrics)
        """
        try:
            # Configure acceleration parameters
            acceleration_params = {
                'factor': acceleration_factor,
                'preserve_pitch': True,
                'content_aware': True,  # Adjust speed based on content complexity
                'key_frame_detection': True,  # Slow down at important frames
                'remove_silence': True,  # Remove silence to further speed up
            }
            
            # Accelerate the video
            accelerated_path, metrics = self.accelerator.process(
                video_path, **acceleration_params
            )
            
            logger.info(f"Video accelerated {acceleration_factor}x")
            logger.info(f"Original duration: {metrics.get('original_duration')} seconds")
            logger.info(f"Accelerated duration: {metrics.get('accelerated_duration')} seconds")
            logger.info(f"Effective speed factor: {metrics.get('effective_factor')}")
            
            return accelerated_path, metrics
            
        except Exception as e:
            logger.error(f"Failed to accelerate video: {e}")
            raise
    
    def _extract_knowledge(self, 
                          video_path: str, 
                          video_info: Dict) -> List[Dict]:
        """
        Extract knowledge concepts and relationships from the video.
        
        Args:
            video_path: Path to the accelerated video
            video_info: Metadata about the video
            
        Returns:
            List of extracted knowledge entities
        """
        try:
            # This would use techniques like:
            # - Speech-to-text to transcribe content
            # - NLP to extract concepts and relationships
            # - Visual analysis to identify objects and text on screen
            
            # In a real implementation, this would use ML models trained for this purpose
            # For this example, we'll simulate extraction based on video metadata
            
            topic = video_info.get('title', '').split(' - ')[0]
            
            # Simulate extracted knowledge
            knowledge_entities = [
                {
                    'id': 'concept_001',
                    'type': 'concept',
                    'name': f"{topic} Basics",
                    'confidence': 0.92,
                    'timestamp': '00:01:23'
                },
                {
                    'id': 'procedure_001',
                    'type': 'procedure',
                    'name': f"Setting up {topic} environment",
                    'steps': [
                        "Install dependencies",
                        "Configure settings",
                        "Initialize system"
                    ],
                    'confidence': 0.87,
                    'timestamp': '00:03:45'
                },
                {
                    'id': 'technique_001',
                    'type': 'technique',
                    'name': f"Advanced {topic} optimization",
                    'confidence': 0.75,
                    'timestamp': '00:12:30'
                }
            ]
            
            logger.info(f"Extracted {len(knowledge_entities)} knowledge entities")
            
            return knowledge_entities
            
        except Exception as e:
            logger.error(f"Failed to extract knowledge: {e}")
            raise
    
    def _detect_ui_interactions(self, video_path: str) -> List[Dict]:
        """
        Detect UI interactions in the tutorial video.
        
        Args:
            video_path: Path to the accelerated video
            
        Returns:
            List of detected UI interactions
        """
        try:
            # Configure UI detection parameters
            detection_params = {
                'detect_clicks': True,
                'detect_typing': True,
                'detect_scrolling': True,
                'detect_menu_navigation': True,
                'extract_ui_elements': True,
                'confidence_threshold': 0.7
            }
            
            # Detect UI interactions
            ui_interactions = self.ui_detector.analyze(video_path, **detection_params)
            
            # Log detected interactions
            logger.info(f"Detected {len(ui_interactions)} UI interactions")
            for i, interaction in enumerate(ui_interactions[:3]):  # Log first 3 as examples
                logger.info(f"  Interaction {i+1}: {interaction['type']} at {interaction['timestamp']}")
                
            return ui_interactions
            
        except Exception as e:
            logger.error(f"Failed to detect UI interactions: {e}")
            raise
    
    def _store_in_knowledge_graph(self,
                                video_info: Dict,
                                knowledge_entities: List[Dict],
                                ui_interactions: List[Dict]) -> str:
        """
        Store all extracted information in the knowledge graph.
        
        Args:
            video_info: Metadata about the video
            knowledge_entities: Extracted knowledge concepts
            ui_interactions: Detected UI interactions
            
        Returns:
            ID of the knowledge graph entry
        """
        try:
            # Create a new graph transaction
            transaction = self.knowledge_graph.create_transaction()
            
            # Add video as the root node
            video_node = transaction.add_node(
                type='video',
                properties={
                    'title': video_info.get('title', 'Unknown'),
                    'url': video_info.get('url', 'Unknown'),
                    'duration': video_info.get('duration', 0),
                    'author': video_info.get('author', 'Unknown'),
                    'processed_date': self.knowledge_graph.get_current_timestamp()
                }
            )
            
            # Add knowledge entities and connect to video
            for entity in knowledge_entities:
                entity_node = transaction.add_node(
                    type=entity['type'],
                    properties=entity
                )
                transaction.add_edge(
                    source=video_node,
                    target=entity_node,
                    relationship='contains',
                    properties={'timestamp': entity.get('timestamp')}
                )
            
            # Add UI interactions and connect to video
            for interaction in ui_interactions:
                interaction_node = transaction.add_node(
                    type='ui_interaction',
                    properties=interaction
                )
                transaction.add_edge(
                    source=video_node,
                    target=interaction_node,
                    relationship='demonstrates',
                    properties={'timestamp': interaction.get('timestamp')}
                )
            
            # Commit the transaction to the knowledge graph
            graph_id = transaction.commit()
            
            logger.info(f"Knowledge stored in graph with ID: {graph_id}")
            
            return graph_id
            
        except Exception as e:
            logger.error(f"Failed to store in knowledge graph: {e}")
            raise

    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a human-readable summary of the processing results.
        
        Args:
            results: Results dictionary from process_tutorial
        """
        if not results:
            print("No results to display.")
            return
            
        print("\n" + "="*80)
        print(f"HYPERBOLIC LEARNER: TUTORIAL PROCESSING SUMMARY")
        print("="*80)
        
        # Video information
        video_info = results.get('video_info', {})
        print(f"\nTUTORIAL: {video_info.get('title', 'Unknown')}")
        print(f"AUTHOR: {video_info.get('author', 'Unknown')}")
        print(f"DURATION: {video_info.get('duration', 0)} seconds")
        
        # Acceleration metrics
        metrics = results.get('acceleration_metrics', {})
        print(f"\nACCELERATION:")
        print(f"  Original duration: {metrics.get('original_duration', 0):.2f} seconds")
        print(f"  Accelerated duration: {metrics.get('accelerated_duration', 0):.2f} seconds")
        print(f"  Time saved: {metrics.get('time_saved', 0):.2f} seconds "
              f"({metrics.get('time_saved_percentage', 0):.1f}%)")
        
        # Knowledge extraction
        entities = results.get('knowledge_entities', [])
        print(f"\nEXTRACTED KNOWLEDGE: {len(entities)} entities")
        for i, entity in enumerate(entities[:5]):  # Show first 5
            print(f"  {i+1}. [{entity.get('type', 'Unknown').upper()}] "
                  f"{entity.get('name', 'Unnamed')} "
                  f"(Confidence: {entity.get('confidence', 0):.2f})")
        if len(entities) > 5:
            print(f"  ... and {len(

