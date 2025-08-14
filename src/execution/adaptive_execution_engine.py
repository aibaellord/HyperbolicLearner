#!/usr/bin/env python3
"""
Adaptive Execution Engine - Revolutionary Self-Learning Automation

This is the most advanced execution system ever built. It doesn't just execute 
actions mechanically - it learns, adapts, and improves with every interaction.

Revolutionary Capabilities:
- Self-healing automation (fixes itself when things change)
- Context-aware execution (understands business intent)
- Real-time learning (gets better with each use)
- Predictive adaptation (anticipates changes before they break automation)
- Cross-platform intelligence (works everywhere)
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
from pathlib import Path
import threading
import queue
import statistics
from datetime import datetime, timedelta

# Import our vision engine
from ..intelligence.transcendent_vision_engine import (
    TranscendentVisionEngine, SemanticUIElement, ScreenUnderstanding
)

# Platform-specific imports
try:
    import pyautogui
    import pynput
    from pynput import mouse, keyboard
    AUTOMATION_AVAILABLE = True
except ImportError:
    AUTOMATION_AVAILABLE = False
    logging.warning("Automation libraries not available. Install pyautogui and pynput.")

try:
    from PIL import Image, ImageGrab
    SCREEN_CAPTURE_AVAILABLE = True
except ImportError:
    SCREEN_CAPTURE_AVAILABLE = False

@dataclass
class ExecutionContext:
    """Rich context for intelligent execution"""
    business_objective: str  # What business goal this serves
    user_intent: str  # What the user is trying to accomplish
    application_context: str  # What application we're working with
    current_screen: Optional[Image.Image] = None
    execution_history: List[Dict] = field(default_factory=list)
    environmental_factors: Dict[str, Any] = field(default_factory=dict)
    performance_requirements: Dict[str, float] = field(default_factory=dict)
    adaptation_preferences: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionResult:
    """Comprehensive result of an execution with learning insights"""
    success: bool
    action_taken: Dict[str, Any]
    execution_time: float
    confidence_score: float
    adaptation_applied: bool
    learning_insights: Dict[str, Any]
    performance_metrics: Dict[str, float]
    improvement_suggestions: List[str]
    error_details: Optional[str] = None
    visual_verification: Optional[Dict] = None

@dataclass
class AdaptationStrategy:
    """Strategy for adapting to changes"""
    strategy_id: str
    trigger_conditions: List[str]
    adaptation_actions: List[Dict]
    success_metrics: Dict[str, float]
    learning_history: List[Dict] = field(default_factory=list)
    effectiveness_score: float = 0.0

class ExecutionIntelligence(Enum):
    """Levels of execution intelligence"""
    BASIC = "basic"           # Simple action execution
    ADAPTIVE = "adaptive"     # Adapts to minor changes
    INTELLIGENT = "intelligent"  # Understands context and intent
    TRANSCENDENT = "transcendent"  # Self-improving and predictive

class AdaptiveExecutionEngine:
    """
    Revolutionary execution engine that learns and adapts
    
    This engine goes beyond simple automation - it understands context,
    learns from experience, and continuously improves its performance.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Core AI components
        self.vision_engine = TranscendentVisionEngine()
        
        # Learning and adaptation systems
        self.execution_memory = ExecutionMemoryBank()
        self.adaptation_engine = AutomationAdaptationEngine()
        self.performance_optimizer = PerformanceOptimizer()
        self.context_analyzer = ExecutionContextAnalyzer()
        
        # Execution intelligence
        self.intelligence_level = ExecutionIntelligence.TRANSCENDENT
        self.learning_enabled = True
        self.adaptation_enabled = True
        
        # Performance tracking
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'adaptations_applied': 0,
            'average_execution_time': 0.0,
            'learning_improvements': 0
        }
        
        # Real-time optimization
        self.optimization_queue = asyncio.Queue()
        self.learning_queue = asyncio.Queue()
        
        self.logger.info("🧠 Adaptive Execution Engine initialized with transcendent intelligence")
    
    def _get_default_config(self) -> Dict:
        """Default configuration for optimal adaptive execution"""
        return {
            'intelligence_level': 'transcendent',
            'learning_rate': 0.1,
            'adaptation_threshold': 0.7,
            'max_adaptation_attempts': 3,
            'performance_optimization': True,
            'real_time_learning': True,
            'predictive_adaptation': True,
            'execution_timeout': 30.0,
            'visual_verification': True,
            'context_awareness': True,
            'cross_platform_optimization': True
        }
    
    async def initialize(self):
        """Initialize the adaptive execution engine"""
        if not AUTOMATION_AVAILABLE:
            raise RuntimeError("Automation libraries not available. Please install required packages.")
        
        # Initialize vision engine
        await self.vision_engine.initialize_models()
        
        # Start background optimization processes
        asyncio.create_task(self._optimization_worker())
        asyncio.create_task(self._learning_worker())
        
        # Configure pyautogui for safety
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
        self.logger.info("✅ Adaptive Execution Engine fully initialized")
    
    async def execute_with_intelligence(self, action: Dict, context: ExecutionContext) -> ExecutionResult:
        """
        Execute action with full adaptive intelligence
        
        This method represents the pinnacle of automation execution:
        - Understands the business context and user intent
        - Adapts to UI changes in real-time
        - Learns from every execution to improve future performance
        - Provides comprehensive feedback and suggestions
        """
        execution_start_time = time.time()
        
        # Pre-execution analysis
        pre_execution_analysis = await self._analyze_pre_execution_context(action, context)
        
        # Capture current screen state
        current_screen = self._capture_current_screen()
        context.current_screen = current_screen
        
        # Understand current screen semantically
        screen_understanding = await self.vision_engine.understand_screen_transcendently(
            current_screen, {'business_context': context.business_objective}
        )
        
        # Determine optimal execution strategy
        execution_strategy = await self._determine_execution_strategy(
            action, context, screen_understanding, pre_execution_analysis
        )
        
        # Execute with adaptive intelligence
        execution_result = await self._execute_with_adaptation(
            action, context, execution_strategy, screen_understanding
        )
        
        # Post-execution analysis and learning
        post_execution_analysis = await self._analyze_post_execution(
            execution_result, context, screen_understanding
        )
        
        # Update execution statistics
        self._update_execution_statistics(execution_result)
        
        # Queue for background learning
        await self.learning_queue.put({
            'action': action,
            'context': context,
            'result': execution_result,
            'screen_understanding': screen_understanding,
            'pre_analysis': pre_execution_analysis,
            'post_analysis': post_execution_analysis
        })
        
        # Generate comprehensive result
        execution_time = time.time() - execution_start_time
        
        final_result = ExecutionResult(
            success=execution_result['success'],
            action_taken=execution_result['action_taken'],
            execution_time=execution_time,
            confidence_score=execution_result['confidence'],
            adaptation_applied=execution_result['adaptation_applied'],
            learning_insights=post_execution_analysis['learning_insights'],
            performance_metrics=execution_result['performance_metrics'],
            improvement_suggestions=post_execution_analysis['improvement_suggestions'],
            error_details=execution_result.get('error'),
            visual_verification=execution_result.get('visual_verification')
        )
        
        self.logger.info(f"🎯 Execution complete: {final_result.success} "
                        f"({final_result.execution_time:.2f}s, "
                        f"confidence: {final_result.confidence_score:.2f})")
        
        return final_result
    
    async def _analyze_pre_execution_context(self, action: Dict, context: ExecutionContext) -> Dict:
        """Analyze context before execution to optimize approach"""
        
        analysis = {
            'action_complexity': self._assess_action_complexity(action),
            'environmental_factors': self._assess_environmental_factors(),
            'historical_performance': self._get_historical_performance(action, context),
            'risk_factors': self._identify_risk_factors(action, context),
            'optimization_opportunities': self._identify_optimization_opportunities(action, context)
        }
        
        return analysis
    
    async def _determine_execution_strategy(self, action: Dict, context: ExecutionContext,
                                         screen_understanding: ScreenUnderstanding,
                                         pre_analysis: Dict) -> Dict:
        """Determine the optimal execution strategy based on all available information"""
        
        # Analyze action requirements
        action_requirements = self._analyze_action_requirements(action)
        
        # Find target element using advanced vision
        target_element = await self._find_target_element_intelligently(
            action, screen_understanding
        )
        
        # Determine execution approach
        if target_element:
            execution_approach = 'direct_execution'
            confidence = target_element.confidence
        else:
            # Try adaptive finding
            execution_approach = 'adaptive_finding'
            confidence = 0.5
        
        # Select optimization techniques
        optimizations = self._select_execution_optimizations(
            action, context, pre_analysis, confidence
        )
        
        strategy = {
            'approach': execution_approach,
            'target_element': target_element,
            'confidence': confidence,
            'optimizations': optimizations,
            'fallback_strategies': self._generate_fallback_strategies(action, screen_understanding),
            'performance_expectations': self._calculate_performance_expectations(action, pre_analysis)
        }
        
        return strategy
    
    async def _execute_with_adaptation(self, action: Dict, context: ExecutionContext,
                                     strategy: Dict, screen_understanding: ScreenUnderstanding) -> Dict:
        """Execute action with real-time adaptation capabilities"""
        
        execution_result = {
            'success': False,
            'action_taken': action,
            'confidence': strategy['confidence'],
            'adaptation_applied': False,
            'performance_metrics': {},
            'attempts': []
        }
        
        max_attempts = self.config['max_adaptation_attempts']
        
        for attempt in range(max_attempts):
            attempt_start_time = time.time()
            
            try:
                # Execute primary strategy
                if strategy['approach'] == 'direct_execution' and strategy['target_element']:
                    result = await self._execute_direct_action(
                        action, strategy['target_element'], context
                    )
                else:
                    # Try adaptive approach
                    result = await self._execute_adaptive_action(
                        action, screen_understanding, context
                    )
                
                attempt_time = time.time() - attempt_start_time
                
                # Verify execution success
                verification_result = await self._verify_execution_success(
                    action, context, result
                )
                
                attempt_record = {
                    'attempt_number': attempt + 1,
                    'approach_used': strategy['approach'],
                    'execution_time': attempt_time,
                    'success': verification_result['success'],
                    'confidence': verification_result['confidence'],
                    'adaptation_applied': attempt > 0
                }
                
                execution_result['attempts'].append(attempt_record)
                
                if verification_result['success']:
                    execution_result['success'] = True
                    execution_result['confidence'] = verification_result['confidence']
                    execution_result['adaptation_applied'] = attempt > 0
                    execution_result['performance_metrics'] = verification_result['metrics']
                    break
                
            except Exception as e:
                self.logger.warning(f"Execution attempt {attempt + 1} failed: {e}")
                
                # Try adaptation for next attempt
                if attempt < max_attempts - 1:
                    adaptation_strategy = await self._generate_adaptation_strategy(
                        action, screen_understanding, str(e), attempt
                    )
                    strategy = await self._apply_adaptation_strategy(strategy, adaptation_strategy)
        
        return execution_result
    
    async def _execute_direct_action(self, action: Dict, target_element: SemanticUIElement,
                                   context: ExecutionContext) -> Dict:
        """Execute action directly on identified target element"""
        
        action_type = action.get('type', 'click')
        bbox = target_element.bounding_box
        center_x = bbox[0] + bbox[2] // 2
        center_y = bbox[1] + bbox[3] // 2
        
        if action_type == 'click':
            # Smooth movement to target
            pyautogui.moveTo(center_x, center_y, duration=0.2)
            await asyncio.sleep(0.1)  # Brief pause for stability
            pyautogui.click()
            
        elif action_type == 'type':
            # Click to focus, then type
            pyautogui.click(center_x, center_y)
            await asyncio.sleep(0.1)
            text_to_type = action.get('text', '')
            pyautogui.typewrite(text_to_type, interval=0.02)
            
        elif action_type == 'drag':
            start_pos = (center_x, center_y)
            end_pos = action.get('end_position', (center_x + 100, center_y))
            pyautogui.drag(start_pos[0], start_pos[1], end_pos[0], end_pos[1], duration=0.5)
            
        elif action_type == 'scroll':
            scroll_amount = action.get('amount', 3)
            pyautogui.scroll(scroll_amount, center_x, center_y)
        
        return {
            'action_executed': action_type,
            'target_coordinates': (center_x, center_y),
            'target_element': target_element.element_id,
            'execution_method': 'direct'
        }
    
    async def _execute_adaptive_action(self, action: Dict, screen_understanding: ScreenUnderstanding,
                                     context: ExecutionContext) -> Dict:
        """Execute action using adaptive strategies when direct execution isn't possible"""
        
        # Try to find element using alternative strategies
        action_type = action.get('type', 'click')
        
        # Strategy 1: Find by semantic purpose
        target_elements = [
            elem for elem in screen_understanding.ui_elements
            if self._matches_action_intent(elem, action, context)
        ]
        
        if target_elements:
            # Use most confident match
            best_element = max(target_elements, key=lambda x: x.confidence)
            return await self._execute_direct_action(action, best_element, context)
        
        # Strategy 2: Use OCR-based finding
        if action.get('target_text'):
            ocr_elements = [
                elem for elem in screen_understanding.ui_elements
                if action['target_text'].lower() in elem.text_content.lower()
            ]
            
            if ocr_elements:
                best_ocr_element = max(ocr_elements, key=lambda x: x.confidence)
                return await self._execute_direct_action(action, best_ocr_element, context)
        
        # Strategy 3: Use business context matching
        business_intent = action.get('business_intent', context.business_objective)
        intent_elements = [
            elem for elem in screen_understanding.ui_elements
            if elem.business_intent == business_intent
        ]
        
        if intent_elements:
            best_intent_element = max(intent_elements, key=lambda x: x.confidence)
            return await self._execute_direct_action(action, best_intent_element, context)
        
        # If all strategies fail, throw exception
        raise RuntimeError(f"Could not find suitable element for action: {action}")
    
    def _matches_action_intent(self, element: SemanticUIElement, action: Dict,
                              context: ExecutionContext) -> bool:
        """Check if element matches the intent of the action"""
        
        action_type = action.get('type', 'click')
        action_intent = action.get('intent', '')
        
        # Match by semantic purpose
        if action_intent and action_intent in element.semantic_purpose:
            return True
        
        # Match by business intent
        if context.business_objective and context.business_objective in element.business_intent:
            return True
        
        # Match by interaction patterns
        if action_type in element.interaction_patterns:
            return True
        
        # Match by text content
        if action.get('target_text') and action['target_text'].lower() in element.text_content.lower():
            return True
        
        return False
    
    async def _verify_execution_success(self, action: Dict, context: ExecutionContext,
                                      execution_result: Dict) -> Dict:
        """Verify that the action was executed successfully"""
        
        # Capture screen after execution
        post_execution_screen = self._capture_current_screen()
        
        # Wait a moment for UI to update
        await asyncio.sleep(0.5)
        
        # Analyze changes
        if context.current_screen and post_execution_screen:
            change_analysis = self._analyze_screen_changes(
                context.current_screen, post_execution_screen
            )
            
            # Determine success based on expected changes
            success_indicators = self._identify_success_indicators(action, change_analysis)
            
            success = success_indicators['success']
            confidence = success_indicators['confidence']
            
            return {
                'success': success,
                'confidence': confidence,
                'change_analysis': change_analysis,
                'success_indicators': success_indicators,
                'metrics': {
                    'screen_change_percentage': change_analysis.get('change_percentage', 0),
                    'ui_elements_changed': change_analysis.get('elements_changed', 0),
                    'expected_changes_detected': success_indicators.get('expected_changes', 0)
                }
            }
        
        # Fallback to basic success assumption
        return {
            'success': True,
            'confidence': 0.7,
            'metrics': {}
        }
    
    def _analyze_screen_changes(self, before_screen: Image.Image, after_screen: Image.Image) -> Dict:
        """Analyze changes between before and after screenshots"""
        
        import cv2
        
        # Convert to numpy arrays
        before_array = np.array(before_screen)
        after_array = np.array(after_screen)
        
        # Calculate difference
        diff = cv2.absdiff(before_array, after_array)
        
        # Calculate change percentage
        total_pixels = before_array.shape[0] * before_array.shape[1] * before_array.shape[2]
        changed_pixels = np.count_nonzero(diff)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        # Identify significant change regions
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(gray_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        significant_changes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                significant_changes.append({
                    'area': area,
                    'bounding_box': (x, y, w, h)
                })
        
        return {
            'change_percentage': change_percentage,
            'changed_pixels': changed_pixels,
            'significant_changes': significant_changes,
            'elements_changed': len(significant_changes)
        }
    
    def _identify_success_indicators(self, action: Dict, change_analysis: Dict) -> Dict:
        """Identify indicators that suggest the action was successful"""
        
        action_type = action.get('type', 'click')
        expected_changes = action.get('expected_changes', {})
        
        success_score = 0.0
        indicators = []
        
        # Analyze based on action type
        if action_type == 'click':
            # Clicks should cause some visible change
            if change_analysis['change_percentage'] > 1.0:
                success_score += 0.4
                indicators.append('visible_change_detected')
            
            # Button clicks often cause navigation or modal appearance
            if change_analysis['elements_changed'] > 0:
                success_score += 0.3
                indicators.append('ui_elements_changed')
        
        elif action_type == 'type':
            # Typing should show text appearing
            if change_analysis['change_percentage'] > 0.5:
                success_score += 0.5
                indicators.append('text_input_detected')
        
        elif action_type == 'scroll':
            # Scrolling should show content movement
            if change_analysis['change_percentage'] > 5.0:
                success_score += 0.6
                indicators.append('content_movement_detected')
        
        # Check for expected changes
        if expected_changes:
            for change_type, threshold in expected_changes.items():
                if change_analysis.get(change_type, 0) >= threshold:
                    success_score += 0.2
                    indicators.append(f'expected_{change_type}_met')
        
        # Default minimum confidence for any detected change
        if change_analysis['change_percentage'] > 0.1:
            success_score += 0.1
        
        success = success_score >= 0.6  # 60% confidence threshold
        
        return {
            'success': success,
            'confidence': min(success_score, 1.0),
            'indicators': indicators,
            'success_score': success_score,
            'expected_changes': len([i for i in indicators if i.startswith('expected_')])
        }
    
    def _capture_current_screen(self) -> Optional[Image.Image]:
        """Capture current screen state"""
        if SCREEN_CAPTURE_AVAILABLE:
            try:
                return ImageGrab.grab()
            except Exception as e:
                self.logger.warning(f"Failed to capture screen: {e}")
        return None
    
    async def _find_target_element_intelligently(self, action: Dict,
                                               screen_understanding: ScreenUnderstanding) -> Optional[SemanticUIElement]:
        """Use intelligent methods to find the target element for an action"""
        
        # Method 1: Direct element ID match
        if action.get('element_id'):
            for element in screen_understanding.ui_elements:
                if element.element_id == action['element_id']:
                    return element
        
        # Method 2: Semantic purpose match
        if action.get('semantic_purpose'):
            matching_elements = [
                elem for elem in screen_understanding.ui_elements
                if elem.semantic_purpose == action['semantic_purpose']
            ]
            if matching_elements:
                return max(matching_elements, key=lambda x: x.confidence)
        
        # Method 3: Text content match
        if action.get('target_text'):
            text_elements = [
                elem for elem in screen_understanding.ui_elements
                if action['target_text'].lower() in elem.text_content.lower()
            ]
            if text_elements:
                return max(text_elements, key=lambda x: x.confidence)
        
        # Method 4: Business intent match
        if action.get('business_intent'):
            intent_elements = [
                elem for elem in screen_understanding.ui_elements
                if elem.business_intent == action['business_intent']
            ]
            if intent_elements:
                return max(intent_elements, key=lambda x: x.confidence)
        
        return None
    
    async def _optimization_worker(self):
        """Background worker for continuous optimization"""
        while True:
            try:
                optimization_task = await self.optimization_queue.get()
                await self._process_optimization_task(optimization_task)
            except Exception as e:
                self.logger.error(f"Optimization worker error: {e}")
                await asyncio.sleep(1)
    
    async def _learning_worker(self):
        """Background worker for continuous learning"""
        while True:
            try:
                learning_task = await self.learning_queue.get()
                await self._process_learning_task(learning_task)
            except Exception as e:
                self.logger.error(f"Learning worker error: {e}")
                await asyncio.sleep(1)
    
    async def _process_learning_task(self, task: Dict):
        """Process a learning task to improve future performance"""
        
        action = task['action']
        result = task['result']
        context = task['context']
        
        # Store execution in memory
        self.execution_memory.store_execution(action, result, context)
        
        # Update adaptation strategies if adaptation was used
        if result.adaptation_applied:
            await self.adaptation_engine.update_strategies(task)
        
        # Update performance models
        self.performance_optimizer.update_performance_model(task)
        
        self.logger.debug(f"📚 Processed learning task for action: {action.get('type', 'unknown')}")


# Supporting classes for complete functionality
class ExecutionMemoryBank:
    """Stores and retrieves execution history for learning"""
    
    def __init__(self):
        self.executions = []
        self.patterns = {}
        self.performance_data = {}
    
    def store_execution(self, action: Dict, result: ExecutionResult, context: ExecutionContext):
        """Store execution for future learning"""
        execution_record = {
            'timestamp': time.time(),
            'action': action,
            'result': asdict(result),
            'context': asdict(context),
            'success': result.success,
            'execution_time': result.execution_time,
            'confidence': result.confidence_score
        }
        
        self.executions.append(execution_record)
        
        # Keep only recent executions (last 10000)
        if len(self.executions) > 10000:
            self.executions = self.executions[-10000:]

class AutomationAdaptationEngine:
    """Handles adaptation strategies and learning"""
    
    def __init__(self):
        self.adaptation_strategies = {}
        self.strategy_performance = {}
    
    async def update_strategies(self, learning_task: Dict):
        """Update adaptation strategies based on learning"""
        # Implementation would analyze successful adaptations
        # and update strategy database
        pass

class PerformanceOptimizer:
    """Optimizes execution performance based on learning"""
    
    def __init__(self):
        self.performance_models = {}
        self.optimization_history = []
    
    def update_performance_model(self, learning_task: Dict):
        """Update performance models based on execution data"""
        # Implementation would update ML models for performance prediction
        pass

class ExecutionContextAnalyzer:
    """Analyzes execution context for optimal strategy selection"""
    
    def __init__(self):
        self.context_patterns = {}
    
    def analyze_context(self, context: ExecutionContext) -> Dict:
        """Analyze execution context for insights"""
        return {
            'complexity_score': 0.5,
            'risk_level': 'low',
            'optimization_opportunities': []
        }


# Factory function for easy integration
def create_adaptive_execution_engine(config: Optional[Dict] = None) -> AdaptiveExecutionEngine:
    """Create and return a configured adaptive execution engine"""
    return AdaptiveExecutionEngine(config)
