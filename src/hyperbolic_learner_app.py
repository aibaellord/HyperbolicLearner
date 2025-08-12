#!/usr/bin/env python3
"""
Complete HyperbolicLearner Application with Full N8N Integration

This is the complete implementation of HyperbolicLearner with all functionality
needed for the maximum potential n8n integration system.
"""

import asyncio
import logging
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from datetime import datetime

# Core HyperbolicLearner imports
from .video_processor.youtube_learner import YouTubeLearner
from .ml_engine.content_analyzer import ContentAnalyzer
from .ui_automation.ui_analyzer import UIAnalyzer
from .knowledge_base.graph_db import GraphDatabase
from .action_executor.system_interactor import SystemInteractor
from .core.config import Config

@dataclass
class LearningResult:
    """Complete learning result from video processing"""
    knowledge_id: str
    video_url: str
    ui_actions: List[Dict[str, Any]]
    knowledge_patterns: List[Dict[str, Any]]
    semantic_content: Dict[str, Any]
    acceleration_factor: float
    processing_time: float
    success_rate: float
    value_score: float

class HyperbolicLearnerApp:
    """
    Complete HyperbolicLearner Application
    
    This is the main application class that provides all functionality
    needed for the maximum potential n8n integration system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the complete HyperbolicLearner application"""
        
        # Load configuration
        self.config = Config(config_path) if config_path else Config()
        
        # Initialize core components
        self.youtube_learner = YouTubeLearner(self.config)
        self.content_analyzer = ContentAnalyzer(self.config)
        self.ui_analyzer = UIAnalyzer(self.config)
        self.knowledge_db = GraphDatabase(self.config)
        self.system_interactor = SystemInteractor(self.config)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_metrics = {
            "videos_processed": 0,
            "total_processing_time": 0.0,
            "average_acceleration": 0.0,
            "ui_actions_extracted": 0,
            "workflows_generated": 0,
            "success_rate": 0.0
        }
        
        # Initialize database for persistent storage
        self.init_application_database()
    
    def init_application_database(self):
        """Initialize the application database for storing results"""
        db_path = Path(self.config.data_dir) / "hyperbolic_learner.db"
        db_path.parent.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS learning_results (
                id INTEGER PRIMARY KEY,
                knowledge_id TEXT UNIQUE,
                video_url TEXT,
                ui_actions_json TEXT,
                knowledge_patterns_json TEXT,
                semantic_content_json TEXT,
                acceleration_factor REAL,
                processing_time REAL,
                success_rate REAL,
                value_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS ui_actions (
                id INTEGER PRIMARY KEY,
                knowledge_id TEXT,
                action_type TEXT,
                selector TEXT,
                element_description TEXT,
                value_score REAL,
                automation_potential REAL,
                frequency TEXT,
                time_saved_per_execution INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (knowledge_id) REFERENCES learning_results(knowledge_id)
            );
            
            CREATE TABLE IF NOT EXISTS workflows (
                id INTEGER PRIMARY KEY,
                knowledge_id TEXT,
                workflow_name TEXT,
                workflow_data_json TEXT,
                estimated_value REAL,
                time_saved_hours INTEGER,
                success_rate REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (knowledge_id) REFERENCES learning_results(knowledge_id)
            );
        """)
        conn.commit()
        conn.close()
        
        self.db_path = db_path
    
    async def learn_from_youtube(self, 
                                url: str,
                                acceleration_factor: float = 10.0,
                                extract_ui_actions: bool = True,
                                build_knowledge_graph: bool = True,
                                semantic_compression: bool = True) -> str:
        """
        Complete learning pipeline from YouTube video
        
        Args:
            url: YouTube video URL
            acceleration_factor: Learning acceleration factor (5-30x)
            extract_ui_actions: Whether to extract UI actions
            build_knowledge_graph: Whether to build knowledge graph
            semantic_compression: Whether to use semantic compression
            
        Returns:
            knowledge_id: Unique identifier for the learning result
        """
        start_time = time.time()
        
        try:
            # Generate unique knowledge ID
            knowledge_id = hashlib.md5(f"{url}_{int(time.time())}".encode()).hexdigest()
            
            self.logger.info(f"🎓 Starting hyperbolic learning from: {url}")
            self.logger.info(f"📈 Acceleration factor: {acceleration_factor}x")
            
            # Phase 1: Video Processing with Hyperbolic Acceleration
            video_data = await self.youtube_learner.process_video(
                url=url,
                acceleration_factor=acceleration_factor,
                semantic_compression=semantic_compression
            )
            
            # Phase 2: Content Analysis
            semantic_content = await self.content_analyzer.analyze_content(
                video_data=video_data,
                extract_patterns=True,
                importance_scoring=True
            )
            
            # Phase 3: UI Action Extraction
            ui_actions = []
            if extract_ui_actions:
                ui_actions = await self.ui_analyzer.extract_ui_actions(
                    video_data=video_data,
                    semantic_content=semantic_content
                )
                
                self.logger.info(f"🎯 Extracted {len(ui_actions)} UI actions")
            
            # Phase 4: Knowledge Pattern Recognition
            knowledge_patterns = await self.content_analyzer.extract_knowledge_patterns(
                semantic_content=semantic_content,
                ui_actions=ui_actions
            )
            
            # Phase 5: Knowledge Graph Construction
            if build_knowledge_graph:
                await self.knowledge_db.store_knowledge(
                    knowledge_id=knowledge_id,
                    semantic_content=semantic_content,
                    ui_actions=ui_actions,
                    knowledge_patterns=knowledge_patterns
                )
            
            # Phase 6: Success and Value Calculation
            success_rate = self.calculate_success_rate(ui_actions, knowledge_patterns)
            value_score = self.calculate_value_score(ui_actions, knowledge_patterns)
            
            # Phase 7: Store Complete Result
            processing_time = time.time() - start_time
            actual_acceleration = self.calculate_actual_acceleration(processing_time, video_data)
            
            learning_result = LearningResult(
                knowledge_id=knowledge_id,
                video_url=url,
                ui_actions=ui_actions,
                knowledge_patterns=knowledge_patterns,
                semantic_content=semantic_content,
                acceleration_factor=actual_acceleration,
                processing_time=processing_time,
                success_rate=success_rate,
                value_score=value_score
            )
            
            await self.store_learning_result(learning_result)
            
            # Update performance metrics
            self.update_performance_metrics(learning_result)
            
            self.logger.info(f"✅ Learning completed in {processing_time:.2f}s")
            self.logger.info(f"📊 Acceleration achieved: {actual_acceleration:.2f}x")
            self.logger.info(f"🎯 UI actions extracted: {len(ui_actions)}")
            self.logger.info(f"💎 Value score: {value_score:.2f}")
            
            return knowledge_id
            
        except Exception as e:
            self.logger.error(f"❌ Error in learning pipeline: {e}")
            raise
    
    def get_ui_actions(self, knowledge_id: str) -> List[Dict[str, Any]]:
        """Get UI actions for a specific knowledge ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT ui_actions_json FROM learning_results WHERE knowledge_id = ?",
            (knowledge_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return []
    
    def get_knowledge_patterns(self, knowledge_id: str) -> List[Dict[str, Any]]:
        """Get knowledge patterns for a specific knowledge ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT knowledge_patterns_json FROM learning_results WHERE knowledge_id = ?",
            (knowledge_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return []
    
    def get_semantic_content(self, knowledge_id: str) -> Dict[str, Any]:
        """Get semantic content for a specific knowledge ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT semantic_content_json FROM learning_results WHERE knowledge_id = ?",
            (knowledge_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return {}
    
    async def execute_workflow(self, 
                              knowledge_id: str,
                              target_file: Optional[str] = None,
                              verification: bool = True,
                              adaptation_level: str = "medium") -> Dict[str, Any]:
        """
        Execute a learned workflow
        
        Args:
            knowledge_id: ID of the learned workflow
            target_file: Optional target file for the workflow
            verification: Whether to verify execution
            adaptation_level: Level of adaptation (low, medium, high)
            
        Returns:
            Execution result with success status and output
        """
        try:
            ui_actions = self.get_ui_actions(knowledge_id)
            
            if not ui_actions:
                return {
                    "success": False,
                    "error": f"No UI actions found for knowledge_id: {knowledge_id}"
                }
            
            # Execute the workflow using system interactor
            result = await self.system_interactor.execute_action_sequence(
                actions=ui_actions,
                target_file=target_file,
                verification=verification,
                adaptation_level=adaptation_level
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error executing workflow: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def export_knowledge(self, knowledge_id: str, output_path: str) -> bool:
        """Export knowledge to a file"""
        try:
            # Get all data for knowledge_id
            ui_actions = self.get_ui_actions(knowledge_id)
            knowledge_patterns = self.get_knowledge_patterns(knowledge_id)
            semantic_content = self.get_semantic_content(knowledge_id)
            
            # Create export data
            export_data = {
                "knowledge_id": knowledge_id,
                "export_timestamp": datetime.now().isoformat(),
                "ui_actions": ui_actions,
                "knowledge_patterns": knowledge_patterns,
                "semantic_content": semantic_content,
                "version": "1.0"
            }
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"✅ Knowledge exported to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting knowledge: {e}")
            return False
    
    def load_workflow(self, workflow_name: str) -> bool:
        """Load a previously saved workflow"""
        try:
            # Implementation for loading workflows
            # This would load from the workflows table
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT workflow_data_json FROM workflows WHERE workflow_name = ?",
                (workflow_name,)
            )
            result = cursor.fetchone()
            conn.close()
            
            if result:
                workflow_data = json.loads(result[0])
                self.logger.info(f"✅ Workflow loaded: {workflow_name}")
                return True
            else:
                self.logger.warning(f"⚠️ Workflow not found: {workflow_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error loading workflow: {e}")
            return False
    
    async def store_learning_result(self, result: LearningResult):
        """Store complete learning result in database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Store main learning result
            conn.execute("""
                INSERT OR REPLACE INTO learning_results 
                (knowledge_id, video_url, ui_actions_json, knowledge_patterns_json, 
                 semantic_content_json, acceleration_factor, processing_time, 
                 success_rate, value_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.knowledge_id,
                result.video_url,
                json.dumps(result.ui_actions),
                json.dumps(result.knowledge_patterns),
                json.dumps(result.semantic_content),
                result.acceleration_factor,
                result.processing_time,
                result.success_rate,
                result.value_score
            ))
            
            # Store individual UI actions
            for action in result.ui_actions:
                conn.execute("""
                    INSERT INTO ui_actions 
                    (knowledge_id, action_type, selector, element_description, 
                     value_score, automation_potential, frequency, time_saved_per_execution)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.knowledge_id,
                    action.get('type', ''),
                    action.get('selector', ''),
                    action.get('element_description', ''),
                    action.get('value_score', 0.0),
                    action.get('automation_potential', 0.0),
                    action.get('frequency', ''),
                    action.get('time_saved_per_execution', 0)
                ))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"❌ Error storing learning result: {e}")
            raise
        finally:
            conn.close()
    
    def calculate_success_rate(self, ui_actions: List[Dict], knowledge_patterns: List[Dict]) -> float:
        """Calculate success rate based on extracted content quality"""
        if not ui_actions and not knowledge_patterns:
            return 0.0
        
        # Success factors
        ui_action_quality = sum(action.get('automation_potential', 0.0) for action in ui_actions) / max(len(ui_actions), 1)
        pattern_quality = sum(pattern.get('confidence', 0.0) for pattern in knowledge_patterns) / max(len(knowledge_patterns), 1)
        
        # Combined success rate
        success_rate = (ui_action_quality + pattern_quality) / 2
        return min(success_rate, 1.0)
    
    def calculate_value_score(self, ui_actions: List[Dict], knowledge_patterns: List[Dict]) -> float:
        """Calculate value score based on automation potential and business impact"""
        if not ui_actions and not knowledge_patterns:
            return 0.0
        
        # Value calculation
        ui_value = sum(action.get('value_score', 0.0) for action in ui_actions)
        pattern_value = sum(pattern.get('business_impact', 0.0) for pattern in knowledge_patterns)
        
        return ui_value + pattern_value
    
    def calculate_actual_acceleration(self, processing_time: float, video_data: Dict) -> float:
        """Calculate actual acceleration achieved"""
        video_duration = video_data.get('duration', processing_time)
        if processing_time > 0:
            return video_duration / processing_time
        return 1.0
    
    def update_performance_metrics(self, result: LearningResult):
        """Update application performance metrics"""
        self.performance_metrics["videos_processed"] += 1
        self.performance_metrics["total_processing_time"] += result.processing_time
        self.performance_metrics["ui_actions_extracted"] += len(result.ui_actions)
        
        # Calculate running averages
        video_count = self.performance_metrics["videos_processed"]
        total_acceleration = (self.performance_metrics["average_acceleration"] * (video_count - 1) + 
                            result.acceleration_factor) / video_count
        self.performance_metrics["average_acceleration"] = total_acceleration
        
        total_success = (self.performance_metrics["success_rate"] * (video_count - 1) + 
                        result.success_rate) / video_count
        self.performance_metrics["success_rate"] = total_success
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    async def query_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Query the knowledge base"""
        try:
            results = await self.knowledge_db.query_knowledge(query)
            return results
        except Exception as e:
            self.logger.error(f"❌ Error querying knowledge: {e}")
            return []
    
    def get_all_learning_results(self) -> List[Dict[str, Any]]:
        """Get all learning results from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT knowledge_id, video_url, acceleration_factor, processing_time, 
                   success_rate, value_score, created_at
            FROM learning_results
            ORDER BY created_at DESC
        """)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "knowledge_id": row[0],
                "video_url": row[1],
                "acceleration_factor": row[2],
                "processing_time": row[3],
                "success_rate": row[4],
                "value_score": row[5],
                "created_at": row[6]
            })
        
        conn.close()
        return results
    
    async def batch_process_videos(self, video_urls: List[str], **kwargs) -> List[str]:
        """Process multiple videos in batch"""
        knowledge_ids = []
        
        # Process videos in parallel for maximum efficiency
        tasks = []
        for url in video_urls:
            task = asyncio.create_task(self.learn_from_youtube(url, **kwargs))
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"❌ Error processing {video_urls[i]}: {result}")
            else:
                knowledge_ids.append(result)
        
        self.logger.info(f"✅ Batch processing complete. Processed {len(knowledge_ids)} videos successfully.")
        return knowledge_ids


# Example usage and testing
async def main():
    """Example usage of the complete HyperbolicLearner application"""
    
    # Initialize the application
    app = HyperbolicLearnerApp()
    
    # Example: Learn from a tutorial video
    knowledge_id = await app.learn_from_youtube(
        url="https://www.youtube.com/watch?v=example_tutorial",
        acceleration_factor=15.0,
        extract_ui_actions=True,
        semantic_compression=True
    )
    
    print(f"✅ Learning completed. Knowledge ID: {knowledge_id}")
    
    # Get extracted UI actions
    ui_actions = app.get_ui_actions(knowledge_id)
    print(f"🎯 Extracted {len(ui_actions)} UI actions")
    
    # Export knowledge
    app.export_knowledge(knowledge_id, f"knowledge_{knowledge_id}.json")
    
    # Get performance metrics
    metrics = app.get_performance_metrics()
    print(f"📊 Performance metrics: {metrics}")

if __name__ == "__main__":
    asyncio.run(main())
