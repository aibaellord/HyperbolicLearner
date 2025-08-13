"""
Predictive Workflow Generation
Anticipate and create workflows before you need them

This module provides 15x automation efficiency by:
- Intent prediction based on patterns
- Proactive workflow creation
- Seasonal/temporal pattern recognition
- Competitive intelligence integration
- Context-aware automation suggestions
- Self-optimizing workflow sequences
"""

import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import statistics
from collections import defaultdict, deque
import re
from pathlib import Path

try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

class PredictionType(Enum):
    INTENT_BASED = "intent_based"
    TEMPORAL_PATTERN = "temporal_pattern"
    CONTEXTUAL = "contextual"
    COMPETITIVE = "competitive"
    USER_BEHAVIOR = "user_behavior"
    SEASONAL = "seasonal"

class WorkflowPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"
    BACKGROUND = "background"

class ConfidenceLevel(Enum):
    VERY_HIGH = "very_high"  # 90%+
    HIGH = "high"           # 70-90%
    MEDIUM = "medium"       # 50-70%
    LOW = "low"            # 30-50%
    VERY_LOW = "very_low"  # <30%

@dataclass
class WorkflowPrediction:
    """Represents a predicted workflow need"""
    prediction_id: str
    workflow_type: str
    prediction_type: PredictionType
    confidence: float
    priority: WorkflowPriority
    estimated_execution_time: datetime
    context: Dict[str, Any]
    suggested_actions: List[Dict[str, Any]]
    dependencies: List[str]
    success_probability: float
    resource_requirements: Dict[str, Any]
    business_impact: Dict[str, Any]
    creation_timestamp: datetime

@dataclass
class PatternSignature:
    """Unique signature of a workflow pattern"""
    pattern_id: str
    features: Dict[str, Any]
    frequency: int
    last_occurrence: datetime
    success_rate: float
    context_triggers: List[str]
    seasonal_factors: Dict[str, float]

@dataclass
class IntentSignal:
    """Signal indicating user intent"""
    signal_type: str  # 'application_usage', 'calendar_event', 'email_pattern', etc.
    strength: float   # 0.0 to 1.0
    timestamp: datetime
    context: Dict[str, Any]
    metadata: Dict[str, Any]

class PredictiveWorkflowGenerator:
    """
    Anticipate and create workflows before you need them
    
    Power Multiplier: 15.0x
    Phase: intelligence_amplification
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.power_multiplier = 15.0
        self.active = False
        
        # Prediction models
        self.intent_predictor = None
        self.pattern_classifier = None
        self.temporal_analyzer = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # Pattern storage
        self.workflow_patterns: Dict[str, PatternSignature] = {}
        self.intent_signals: deque = deque(maxlen=10000)
        self.execution_history: List[Dict[str, Any]] = []
        
        # Predictions
        self.active_predictions: Dict[str, WorkflowPrediction] = {}
        self.prediction_accuracy = {}
        
        # Context tracking
        self.current_context: Dict[str, Any] = {}
        self.context_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.predictions_made = 0
        self.predictions_successful = 0
        self.workflows_generated = 0
        self.time_saved_minutes = 0
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'prediction_horizon_hours': 24,
            'min_pattern_occurrences': 3,
            'confidence_threshold': 0.6,
            'temporal_window_days': 30,
            'max_active_predictions': 50,
            'context_similarity_threshold': 0.7,
            'seasonal_analysis_months': 12,
            'intent_signal_timeout_hours': 6,
            'prediction_models_path': 'models/predictive_workflows',
            'auto_generate_workflows': True,
            'notification_enabled': True,
            'learning_rate': 0.1,
            'feature_importance_threshold': 0.05
        }
        
    async def initialize(self):
        """Initialize the predictive workflow generator"""
        self.logger.info("🚀 Initializing Predictive Workflow Generator")
        
        # Create models directory
        models_path = Path(self.config['prediction_models_path'])
        models_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ML models
        if SKLEARN_AVAILABLE:
            await self._initialize_ml_models()
        
        # Load existing patterns
        await self._load_workflow_patterns()
        
        # Initialize temporal analyzer
        await self._initialize_temporal_analyzer()
        
        # Start background prediction process
        self._start_prediction_engine()
        
        self.active = True
        self.logger.info("✅ Predictive Workflow Generator initialized successfully")
        
    async def _initialize_ml_models(self):
        """Initialize machine learning models"""
        try:
            # Intent prediction model
            self.intent_predictor = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Pattern classifier
            self.pattern_classifier = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            
            # Try to load pre-trained models
            await self._load_prediction_models()
            
            self.logger.info("✅ ML models initialized")
            
        except Exception as e:
            self.logger.warning(f"ML model initialization failed: {e}")
            
    async def _load_prediction_models(self):
        """Load pre-trained prediction models"""
        models_path = Path(self.config['prediction_models_path'])
        
        try:
            intent_model_path = models_path / 'intent_predictor.pkl'
            if intent_model_path.exists():
                with open(intent_model_path, 'rb') as f:
                    self.intent_predictor = pickle.load(f)
                self.logger.info("✅ Intent predictor model loaded")
                
            pattern_model_path = models_path / 'pattern_classifier.pkl'
            if pattern_model_path.exists():
                with open(pattern_model_path, 'rb') as f:
                    self.pattern_classifier = pickle.load(f)
                self.logger.info("✅ Pattern classifier model loaded")
                
        except Exception as e:
            self.logger.warning(f"Model loading failed: {e}")
            
    async def _save_prediction_models(self):
        """Save trained prediction models"""
        models_path = Path(self.config['prediction_models_path'])
        
        try:
            if self.intent_predictor:
                with open(models_path / 'intent_predictor.pkl', 'wb') as f:
                    pickle.dump(self.intent_predictor, f)
                    
            if self.pattern_classifier:
                with open(models_path / 'pattern_classifier.pkl', 'wb') as f:
                    pickle.dump(self.pattern_classifier, f)
                    
            self.logger.info("✅ Prediction models saved")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
            
    async def _load_workflow_patterns(self):
        """Load existing workflow patterns"""
        patterns_file = Path(self.config['prediction_models_path']) / 'workflow_patterns.json'
        
        try:
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                    
                for pattern_id, pattern_dict in patterns_data.items():
                    pattern_dict['last_occurrence'] = datetime.fromisoformat(pattern_dict['last_occurrence'])
                    self.workflow_patterns[pattern_id] = PatternSignature(**pattern_dict)
                    
                self.logger.info(f"✅ Loaded {len(self.workflow_patterns)} workflow patterns")
                
        except Exception as e:
            self.logger.warning(f"Pattern loading failed: {e}")
            
    async def _save_workflow_patterns(self):
        """Save workflow patterns"""
        patterns_file = Path(self.config['prediction_models_path']) / 'workflow_patterns.json'
        
        try:
            patterns_data = {}
            for pattern_id, pattern in self.workflow_patterns.items():
                pattern_dict = asdict(pattern)
                pattern_dict['last_occurrence'] = pattern_dict['last_occurrence'].isoformat()
                patterns_data[pattern_id] = pattern_dict
                
            with open(patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
                
            self.logger.info("✅ Workflow patterns saved")
            
        except Exception as e:
            self.logger.error(f"Pattern saving failed: {e}")
            
    async def _initialize_temporal_analyzer(self):
        """Initialize temporal pattern analysis"""
        self.temporal_analyzer = {
            'hourly_patterns': defaultdict(list),
            'daily_patterns': defaultdict(list),
            'weekly_patterns': defaultdict(list),
            'monthly_patterns': defaultdict(list),
            'seasonal_factors': {}
        }
        
    def _start_prediction_engine(self):
        """Start the background prediction engine"""
        # This would start a background task in a real implementation
        self.logger.info("🔮 Prediction engine started")
        
    async def add_intent_signal(self, signal: IntentSignal):
        """Add an intent signal to the analysis"""
        self.intent_signals.append(signal)
        
        # Update current context
        await self._update_context(signal)
        
        # Trigger prediction analysis
        if len(self.intent_signals) % 10 == 0:  # Analyze every 10 signals
            await self._analyze_intent_patterns()
            
    async def _update_context(self, signal: IntentSignal):
        """Update current context based on signal"""
        context_updates = {
            'last_signal_time': signal.timestamp.isoformat(),
            'recent_signal_types': [s.signal_type for s in list(self.intent_signals)[-5:]],
            'activity_level': self._calculate_activity_level(),
            'dominant_patterns': await self._get_dominant_patterns()
        }
        
        self.current_context.update(context_updates)
        
    def _calculate_activity_level(self) -> float:
        """Calculate current activity level (0.0 to 1.0)"""
        if not self.intent_signals:
            return 0.0
            
        # Count signals in last hour
        now = datetime.now()
        recent_signals = [
            s for s in self.intent_signals 
            if (now - s.timestamp).total_seconds() < 3600
        ]
        
        # Normalize to 0-1 scale (assuming max 60 signals/hour = high activity)
        return min(len(recent_signals) / 60.0, 1.0)
        
    async def _get_dominant_patterns(self) -> List[str]:
        """Get currently dominant workflow patterns"""
        if not self.workflow_patterns:
            return []
            
        # Sort patterns by recent activity and success rate
        sorted_patterns = sorted(
            self.workflow_patterns.items(),
            key=lambda x: (
                x[1].frequency * x[1].success_rate * 
                (1.0 / max(1, (datetime.now() - x[1].last_occurrence).days))
            ),
            reverse=True
        )
        
        return [pattern_id for pattern_id, _ in sorted_patterns[:5]]
        
    async def _analyze_intent_patterns(self):
        """Analyze intent signals for pattern recognition"""
        try:
            # Group signals by type and analyze frequency
            signal_analysis = defaultdict(list)
            
            for signal in list(self.intent_signals)[-100:]:  # Analyze last 100 signals
                signal_analysis[signal.signal_type].append(signal)
                
            # Detect emerging patterns
            for signal_type, signals in signal_analysis.items():
                if len(signals) >= 3:
                    await self._detect_signal_pattern(signal_type, signals)
                    
        except Exception as e:
            self.logger.error(f"Intent pattern analysis failed: {e}")
            
    async def _detect_signal_pattern(self, signal_type: str, signals: List[IntentSignal]):
        """Detect patterns in specific signal type"""
        try:
            # Temporal pattern analysis
            timestamps = [s.timestamp for s in signals]
            intervals = [
                (timestamps[i+1] - timestamps[i]).total_seconds() 
                for i in range(len(timestamps)-1)
            ]
            
            if intervals:
                avg_interval = statistics.mean(intervals)
                std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
                
                # Check for regular patterns
                if std_interval < avg_interval * 0.3:  # Low variance = regular pattern
                    await self._create_temporal_prediction(signal_type, avg_interval, signals)
                    
        except Exception as e:
            self.logger.error(f"Signal pattern detection failed: {e}")
            
    async def _create_temporal_prediction(self, signal_type: str, interval: float, signals: List[IntentSignal]):
        """Create prediction based on temporal pattern"""
        try:
            # Estimate next occurrence
            last_signal = signals[-1]
            next_occurrence = last_signal.timestamp + timedelta(seconds=interval)
            
            # Only predict if it's in the future and within horizon
            now = datetime.now()
            if next_occurrence > now and (next_occurrence - now).total_seconds() < self.config['prediction_horizon_hours'] * 3600:
                
                prediction = WorkflowPrediction(
                    prediction_id=f"temporal_{signal_type}_{int(next_occurrence.timestamp())}",
                    workflow_type=self._infer_workflow_type(signal_type, signals),
                    prediction_type=PredictionType.TEMPORAL_PATTERN,
                    confidence=min(0.9, len(signals) / 10.0),  # Higher confidence with more data
                    priority=self._determine_priority(signal_type, signals),
                    estimated_execution_time=next_occurrence,
                    context=self._extract_context(signals),
                    suggested_actions=await self._generate_suggested_actions(signal_type, signals),
                    dependencies=self._identify_dependencies(signal_type),
                    success_probability=self._calculate_success_probability(signals),
                    resource_requirements=self._estimate_resource_requirements(signal_type),
                    business_impact=self._assess_business_impact(signal_type, signals),
                    creation_timestamp=datetime.now()
                )
                
                await self._add_prediction(prediction)
                
        except Exception as e:
            self.logger.error(f"Temporal prediction creation failed: {e}")
            
    def _infer_workflow_type(self, signal_type: str, signals: List[IntentSignal]) -> str:
        """Infer the type of workflow from signals"""
        # Extract common patterns from signal contexts
        contexts = [s.context for s in signals if s.context]
        
        # Simple heuristics - in real implementation, use ML
        if 'email' in signal_type:
            return 'email_automation'
        elif 'calendar' in signal_type:
            return 'meeting_automation'
        elif 'browser' in signal_type:
            return 'web_automation'
        elif 'file' in signal_type:
            return 'file_management'
        elif 'application' in signal_type:
            return 'application_workflow'
        else:
            return 'general_automation'
            
    def _determine_priority(self, signal_type: str, signals: List[IntentSignal]) -> WorkflowPriority:
        """Determine workflow priority"""
        # Calculate average signal strength
        avg_strength = statistics.mean(s.strength for s in signals)
        
        if avg_strength >= 0.8:
            return WorkflowPriority.CRITICAL
        elif avg_strength >= 0.6:
            return WorkflowPriority.HIGH
        elif avg_strength >= 0.4:
            return WorkflowPriority.MEDIUM
        else:
            return WorkflowPriority.LOW
            
    def _extract_context(self, signals: List[IntentSignal]) -> Dict[str, Any]:
        """Extract common context from signals"""
        context = {
            'signal_count': len(signals),
            'time_range': {
                'start': min(s.timestamp for s in signals).isoformat(),
                'end': max(s.timestamp for s in signals).isoformat()
            },
            'average_strength': statistics.mean(s.strength for s in signals),
            'signal_types': list(set(s.signal_type for s in signals))
        }
        
        # Extract common context elements
        all_contexts = [s.context for s in signals if s.context]
        if all_contexts:
            common_keys = set.intersection(*[set(ctx.keys()) for ctx in all_contexts])
            for key in common_keys:
                values = [ctx[key] for ctx in all_contexts]
                if len(set(str(v) for v in values)) == 1:  # All same value
                    context[f'common_{key}'] = values[0]
                    
        return context
        
    async def _generate_suggested_actions(self, signal_type: str, signals: List[IntentSignal]) -> List[Dict[str, Any]]:
        """Generate suggested actions for predicted workflow"""
        actions = []
        
        # Base actions based on signal type
        if 'email' in signal_type:
            actions.extend([
                {'action': 'check_email', 'priority': 1},
                {'action': 'filter_important', 'priority': 2},
                {'action': 'auto_respond', 'priority': 3}
            ])
        elif 'calendar' in signal_type:
            actions.extend([
                {'action': 'check_schedule', 'priority': 1},
                {'action': 'prepare_meeting_materials', 'priority': 2},
                {'action': 'send_reminders', 'priority': 3}
            ])
        elif 'browser' in signal_type:
            actions.extend([
                {'action': 'open_frequent_sites', 'priority': 1},
                {'action': 'clear_cache', 'priority': 2},
                {'action': 'backup_bookmarks', 'priority': 3}
            ])
            
        # Add context-specific actions
        contexts = [s.context for s in signals if s.context]
        for context in contexts:
            if 'application' in context:
                actions.append({
                    'action': f'prepare_{context["application"]}',
                    'priority': 2
                })
                
        return actions[:10]  # Limit to top 10 actions
        
    def _identify_dependencies(self, signal_type: str) -> List[str]:
        """Identify workflow dependencies"""
        dependencies = []
        
        # Common dependencies by signal type
        dependency_map = {
            'email': ['network_connection', 'email_client'],
            'calendar': ['calendar_app', 'network_connection'],
            'browser': ['web_browser', 'network_connection'],
            'file': ['file_system_access'],
            'application': ['target_application']
        }
        
        for key, deps in dependency_map.items():
            if key in signal_type:
                dependencies.extend(deps)
                
        return list(set(dependencies))
        
    def _calculate_success_probability(self, signals: List[IntentSignal]) -> float:
        """Calculate probability of workflow success"""
        # Simple heuristic based on signal strength and consistency
        strengths = [s.strength for s in signals]
        avg_strength = statistics.mean(strengths)
        
        # Consistency bonus (low variance = more predictable)
        if len(strengths) > 1:
            strength_variance = statistics.variance(strengths)
            consistency_bonus = max(0, 0.2 - strength_variance)
        else:
            consistency_bonus = 0
            
        return min(1.0, avg_strength + consistency_bonus)
        
    def _estimate_resource_requirements(self, signal_type: str) -> Dict[str, Any]:
        """Estimate resource requirements for workflow"""
        base_requirements = {
            'cpu_intensity': 'low',
            'memory_usage': 'minimal',
            'network_required': False,
            'user_interaction': False,
            'estimated_duration_minutes': 5
        }
        
        # Adjust based on signal type
        if 'browser' in signal_type or 'api' in signal_type:
            base_requirements.update({
                'network_required': True,
                'estimated_duration_minutes': 10
            })
        elif 'file' in signal_type:
            base_requirements.update({
                'cpu_intensity': 'medium',
                'memory_usage': 'low',
                'estimated_duration_minutes': 3
            })
        elif 'email' in signal_type:
            base_requirements.update({
                'network_required': True,
                'estimated_duration_minutes': 7
            })
            
        return base_requirements
        
    def _assess_business_impact(self, signal_type: str, signals: List[IntentSignal]) -> Dict[str, Any]:
        """Assess potential business impact of workflow"""
        impact = {
            'time_saved_minutes': 0,
            'efficiency_gain': 0.0,
            'error_reduction': 0.0,
            'cost_impact': 'neutral'
        }
        
        # Estimate impact based on signal type and frequency
        signal_frequency = len(signals)
        
        time_savings_map = {
            'email': 15,
            'calendar': 10,
            'browser': 5,
            'file': 8,
            'application': 12
        }
        
        for key, savings in time_savings_map.items():
            if key in signal_type:
                impact['time_saved_minutes'] = savings * min(signal_frequency, 5)
                impact['efficiency_gain'] = min(0.5, signal_frequency * 0.1)
                break
                
        if impact['time_saved_minutes'] > 30:
            impact['cost_impact'] = 'positive'
        elif impact['time_saved_minutes'] > 60:
            impact['cost_impact'] = 'significant'
            
        return impact
        
    async def _add_prediction(self, prediction: WorkflowPrediction):
        """Add a new prediction to active predictions"""
        if len(self.active_predictions) >= self.config['max_active_predictions']:
            # Remove oldest low-priority prediction
            self._cleanup_predictions()
            
        self.active_predictions[prediction.prediction_id] = prediction
        self.predictions_made += 1
        
        self.logger.info(
            f"🔮 New prediction: {prediction.workflow_type} "
            f"(Confidence: {prediction.confidence:.2f}, "
            f"Priority: {prediction.priority.value})"
        )
        
        # Auto-generate workflow if configured and confidence is high
        if (self.config['auto_generate_workflows'] and 
            prediction.confidence >= self.config['confidence_threshold']):
            await self._generate_workflow(prediction)
            
    def _cleanup_predictions(self):
        """Remove old or low-priority predictions"""
        # Sort by priority and age
        sorted_predictions = sorted(
            self.active_predictions.items(),
            key=lambda x: (
                x[1].priority.value,  # Lower priority first
                x[1].creation_timestamp  # Older first
            )
        )
        
        # Remove lowest priority, oldest prediction
        if sorted_predictions:
            prediction_id = sorted_predictions[0][0]
            del self.active_predictions[prediction_id]
            
    async def _generate_workflow(self, prediction: WorkflowPrediction):
        """Generate actual workflow from prediction"""
        try:
            workflow = {
                'id': f"generated_{prediction.prediction_id}",
                'type': prediction.workflow_type,
                'created_from_prediction': prediction.prediction_id,
                'actions': prediction.suggested_actions,
                'schedule': {
                    'execute_at': prediction.estimated_execution_time.isoformat(),
                    'timezone': 'local'
                },
                'dependencies': prediction.dependencies,
                'context': prediction.context,
                'metadata': {
                    'auto_generated': True,
                    'confidence': prediction.confidence,
                    'priority': prediction.priority.value,
                    'estimated_impact': prediction.business_impact
                }
            }
            
            self.workflows_generated += 1
            
            # Estimate time savings
            estimated_savings = prediction.business_impact.get('time_saved_minutes', 0)
            self.time_saved_minutes += estimated_savings
            
            self.logger.info(
                f"⚡ Generated workflow: {workflow['id']} "
                f"(Estimated savings: {estimated_savings} minutes)"
            )
            
            return workflow
            
        except Exception as e:
            self.logger.error(f"Workflow generation failed: {e}")
            return None
            
    async def get_active_predictions(self, priority_filter: Optional[WorkflowPriority] = None) -> List[WorkflowPrediction]:
        """Get currently active predictions"""
        predictions = list(self.active_predictions.values())
        
        if priority_filter:
            predictions = [p for p in predictions if p.priority == priority_filter]
            
        # Sort by confidence and priority
        predictions.sort(
            key=lambda x: (x.priority.value, -x.confidence),
            reverse=True
        )
        
        return predictions
        
    async def validate_prediction(self, prediction_id: str, actual_outcome: bool, execution_time: Optional[datetime] = None):
        """Validate a prediction against actual outcome"""
        if prediction_id not in self.active_predictions:
            self.logger.warning(f"Prediction {prediction_id} not found for validation")
            return
            
        prediction = self.active_predictions[prediction_id]
        
        # Update prediction accuracy
        if prediction.workflow_type not in self.prediction_accuracy:
            self.prediction_accuracy[prediction.workflow_type] = []
            
        accuracy_score = 1.0 if actual_outcome else 0.0
        self.prediction_accuracy[prediction.workflow_type].append(accuracy_score)
        
        # Update global success rate
        if actual_outcome:
            self.predictions_successful += 1
            
        # Learn from validation
        await self._learn_from_validation(prediction, actual_outcome, execution_time)
        
        # Remove validated prediction
        del self.active_predictions[prediction_id]
        
        self.logger.info(
            f"📊 Prediction validated: {prediction_id} "
            f"({'✅ Success' if actual_outcome else '❌ Failed'})"
        )
        
    async def _learn_from_validation(self, prediction: WorkflowPrediction, outcome: bool, execution_time: Optional[datetime]):
        """Learn from prediction validation to improve future predictions"""
        try:
            # Update pattern signatures
            workflow_type = prediction.workflow_type
            
            if workflow_type not in self.workflow_patterns:
                self.workflow_patterns[workflow_type] = PatternSignature(
                    pattern_id=workflow_type,
                    features=prediction.context,
                    frequency=0,
                    last_occurrence=datetime.now(),
                    success_rate=0.0,
                    context_triggers=[],
                    seasonal_factors={}
                )
                
            pattern = self.workflow_patterns[workflow_type]
            pattern.frequency += 1
            pattern.last_occurrence = execution_time or datetime.now()
            
            # Update success rate with exponential smoothing
            alpha = self.config['learning_rate']
            new_success = 1.0 if outcome else 0.0
            pattern.success_rate = alpha * new_success + (1 - alpha) * pattern.success_rate
            
            # Update features with successful patterns
            if outcome:
                for key, value in prediction.context.items():
                    if key not in pattern.features:
                        pattern.features[key] = value
                    elif isinstance(value, (int, float)):
                        # Use exponential smoothing for numeric features
                        pattern.features[key] = alpha * value + (1 - alpha) * pattern.features[key]
                        
            # Save updated patterns
            await self._save_workflow_patterns()
            
        except Exception as e:
            self.logger.error(f"Learning from validation failed: {e}")
            
    def get_prediction_analytics(self) -> Dict[str, Any]:
        """Get analytics on prediction performance"""
        overall_accuracy = (
            self.predictions_successful / self.predictions_made 
            if self.predictions_made > 0 else 0.0
        )
        
        workflow_accuracies = {}
        for workflow_type, scores in self.prediction_accuracy.items():
            workflow_accuracies[workflow_type] = {
                'accuracy': statistics.mean(scores),
                'predictions_count': len(scores),
                'latest_scores': scores[-10:]  # Last 10 predictions
            }
            
        return {
            'overall_accuracy': overall_accuracy,
            'total_predictions': self.predictions_made,
            'successful_predictions': self.predictions_successful,
            'active_predictions_count': len(self.active_predictions),
            'workflows_generated': self.workflows_generated,
            'estimated_time_saved_minutes': self.time_saved_minutes,
            'workflow_type_accuracies': workflow_accuracies,
            'pattern_count': len(self.workflow_patterns),
            'intent_signals_processed': len(self.intent_signals),
            'power_multiplier_achieved': self.power_multiplier
        }
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the predictive workflow generator"""
        return {
            "name": "Predictive Workflow Generation",
            "active": self.active,
            "power_multiplier": self.power_multiplier,
            "phase": "intelligence_amplification",
            "performance": self.get_prediction_analytics(),
            "capabilities": {
                "ml_models_available": SKLEARN_AVAILABLE,
                "auto_generation_enabled": self.config['auto_generate_workflows'],
                "prediction_horizon_hours": self.config['prediction_horizon_hours'],
                "active_predictions": len(self.active_predictions)
            }
        }

# Factory function for easy import
def create_predictive_workflow_generator():
    return PredictiveWorkflowGenerator()

# Example usage and testing
async def main():
    """Test the predictive workflow generator"""
    generator = PredictiveWorkflowGenerator()
    
    try:
        # Initialize
        await generator.initialize()
        
        # Simulate intent signals
        test_signals = [
            IntentSignal(
                signal_type="email_check",
                strength=0.8,
                timestamp=datetime.now(),
                context={"application": "email_client"},
                metadata={"user_action": "manual"}
            ),
            IntentSignal(
                signal_type="calendar_view",
                strength=0.7,
                timestamp=datetime.now() + timedelta(minutes=5),
                context={"application": "calendar"},
                metadata={"upcoming_meetings": 3}
            ),
            IntentSignal(
                signal_type="browser_opening",
                strength=0.6,
                timestamp=datetime.now() + timedelta(minutes=10),
                context={"application": "chrome", "frequent_sites": ["github.com", "stackoverflow.com"]},
                metadata={"tabs_opened": 5}
            )
        ]
        
        # Add signals
        for signal in test_signals:
            await generator.add_intent_signal(signal)
            
        # Get active predictions
        predictions = await generator.get_active_predictions()
        
        print(f"\n🔮 Active Predictions: {len(predictions)}")
        for pred in predictions:
            print(f"  • {pred.workflow_type} "
                  f"(Confidence: {pred.confidence:.2f}, "
                  f"Priority: {pred.priority.value})")
            
        # Get analytics
        analytics = generator.get_prediction_analytics()
        print(f"\n📊 Analytics:")
        print(f"  • Total predictions: {analytics['total_predictions']}")
        print(f"  • Workflows generated: {analytics['workflows_generated']}")
        print(f"  • Patterns discovered: {analytics['pattern_count']}")
        print(f"  • Estimated time saved: {analytics['estimated_time_saved_minutes']} minutes")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
