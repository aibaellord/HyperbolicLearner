"""
Real-Time Screen Intelligence Monitor
Continuously analyzes screen content for learning opportunities

This module provides 5x learning acceleration by monitoring:
- UI patterns and workflows
- Application usage patterns  
- Visual content recognition
- Automation opportunities
- Real-time screen changes
"""

import asyncio
import logging
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageGrab
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import threading
from dataclasses import dataclass, asdict
import base64
import io

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    VISION_MODEL_AVAILABLE = True
except ImportError:
    VISION_MODEL_AVAILABLE = False

@dataclass
class ScreenEvent:
    """Represents a significant screen event"""
    timestamp: datetime
    event_type: str  # 'click', 'type', 'scroll', 'window_change', 'content_change'
    coordinates: Optional[Tuple[int, int]]
    text_content: str
    image_description: str
    confidence: float
    automation_opportunity: Optional[Dict[str, Any]]

@dataclass
class UIElement:
    """Represents a detected UI element"""
    element_type: str  # 'button', 'input', 'text', 'image', 'menu'
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    text: str
    description: str
    clickable: bool
    automatable: bool
    confidence: float

class RealTimeScreenIntelligence:
    """
    Continuously analyze screen content for learning opportunities
    
    Power Multiplier: 5.0x
    Phase: intelligence_amplification
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.power_multiplier = 5.0
        self.active = False
        
        # Screen monitoring
        self.monitor_thread = None
        self.stop_monitoring = False
        self.current_screen = None
        self.previous_screen = None
        
        # AI Models
        self.vision_processor = None
        self.vision_model = None
        
        # Event tracking
        self.screen_events: List[ScreenEvent] = []
        self.ui_elements: List[UIElement] = []
        self.automation_patterns: Dict[str, List[ScreenEvent]] = {}
        
        # Performance tracking
        self.screenshots_analyzed = 0
        self.events_detected = 0
        self.automation_opportunities = 0
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'monitor_interval': 0.5,  # seconds
            'ocr_enabled': True,
            'vision_model_enabled': VISION_MODEL_AVAILABLE,
            'ui_detection_enabled': True,
            'change_threshold': 0.05,  # 5% pixel difference
            'max_events_memory': 1000,
            'automation_pattern_min_occurrences': 3,
            'screenshot_quality': 'medium',  # 'low', 'medium', 'high'
            'save_screenshots': False,
            'screenshot_dir': 'screenshots'
        }
        
    async def initialize(self):
        """Initialize the screen intelligence monitor"""
        self.logger.info("🚀 Initializing Real-Time Screen Intelligence")
        
        # Initialize vision models if available and enabled
        if self.config['vision_model_enabled'] and VISION_MODEL_AVAILABLE:
            try:
                self.logger.info("Loading vision model...")
                self.vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.logger.info("✅ Vision model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load vision model: {e}")
                self.config['vision_model_enabled'] = False
        
        # Test screen capture
        try:
            test_screenshot = self._capture_screen()
            if test_screenshot is None:
                raise Exception("Failed to capture screen")
            self.logger.info(f"✅ Screen capture working - Resolution: {test_screenshot.size}")
        except Exception as e:
            self.logger.error(f"Screen capture failed: {e}")
            raise
            
        self.active = True
        self.logger.info("✅ Real-Time Screen Intelligence initialized successfully")
        
    def start_monitoring(self):
        """Start continuous screen monitoring"""
        if not self.active:
            raise RuntimeError("Must initialize before starting monitoring")
            
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.logger.warning("Monitoring already active")
            return
            
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("🔍 Screen monitoring started")
        
    def stop_monitoring_process(self):
        """Stop screen monitoring"""
        self.stop_monitoring = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("⏹️ Screen monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop - runs in separate thread"""
        self.logger.info("🔄 Screen monitoring loop started")
        
        while not self.stop_monitoring:
            try:
                # Capture current screen
                current_screen = self._capture_screen()
                if current_screen is None:
                    continue
                    
                # Check for significant changes
                if self._has_significant_change(current_screen):
                    # Analyze the screen
                    asyncio.run(self._analyze_screen(current_screen))
                    
                # Update screen state
                self.previous_screen = self.current_screen
                self.current_screen = current_screen
                self.screenshots_analyzed += 1
                
                # Sleep until next capture
                asyncio.run(asyncio.sleep(self.config['monitor_interval']))
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                asyncio.run(asyncio.sleep(1))  # Brief pause on error
                
        self.logger.info("🔄 Screen monitoring loop ended")
        
    def _capture_screen(self) -> Optional[Image.Image]:
        """Capture current screen"""
        try:
            screenshot = ImageGrab.grab()
            
            # Resize based on quality setting
            if self.config['screenshot_quality'] == 'low':
                screenshot = screenshot.resize((screenshot.width // 4, screenshot.height // 4))
            elif self.config['screenshot_quality'] == 'medium':
                screenshot = screenshot.resize((screenshot.width // 2, screenshot.height // 2))
                
            return screenshot
        except Exception as e:
            self.logger.error(f"Failed to capture screen: {e}")
            return None
            
    def _has_significant_change(self, current_screen: Image.Image) -> bool:
        """Check if there's a significant change from previous screen"""
        if self.current_screen is None:
            return True
            
        try:
            # Convert to numpy arrays for comparison
            current_array = np.array(current_screen.resize((100, 100)))
            previous_array = np.array(self.current_screen.resize((100, 100)))
            
            # Calculate difference
            diff = np.mean(np.abs(current_array.astype(float) - previous_array.astype(float)))
            normalized_diff = diff / 255.0
            
            return normalized_diff > self.config['change_threshold']
            
        except Exception as e:
            self.logger.error(f"Error comparing screens: {e}")
            return True  # Assume change on error
            
    async def _analyze_screen(self, screen: Image.Image):
        """Analyze a screen capture for intelligence"""
        try:
            analysis_results = {}
            
            # OCR text extraction
            if self.config['ocr_enabled']:
                text_content = await self._extract_text(screen)
                analysis_results['text'] = text_content
            
            # Visual scene description
            if self.config['vision_model_enabled']:
                scene_description = await self._describe_scene(screen)
                analysis_results['description'] = scene_description
            
            # UI element detection
            if self.config['ui_detection_enabled']:
                ui_elements = await self._detect_ui_elements(screen)
                analysis_results['ui_elements'] = ui_elements
                self.ui_elements.extend(ui_elements)
            
            # Detect automation opportunities
            automation_opportunities = await self._detect_automation_opportunities(analysis_results)
            
            # Create screen event
            event = ScreenEvent(
                timestamp=datetime.now(),
                event_type='content_change',
                coordinates=None,
                text_content=analysis_results.get('text', ''),
                image_description=analysis_results.get('description', ''),
                confidence=0.8,
                automation_opportunity=automation_opportunities
            )
            
            # Store event
            self._add_screen_event(event)
            
            # Update automation patterns
            if automation_opportunities:
                await self._update_automation_patterns(event)
                
            self.events_detected += 1
            
        except Exception as e:
            self.logger.error(f"Error analyzing screen: {e}")
            
    async def _extract_text(self, image: Image.Image) -> str:
        """Extract text from screen using OCR"""
        try:
            # Use pytesseract for OCR
            text = pytesseract.image_to_string(image, config='--psm 6')
            return text.strip()
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            return ""
            
    async def _describe_scene(self, image: Image.Image) -> str:
        """Describe the screen scene using vision model"""
        if not self.vision_model:
            return ""
            
        try:
            # Process image through vision model
            inputs = self.vision_processor(image, return_tensors="pt")
            out = self.vision_model.generate(**inputs, max_length=50)
            description = self.vision_processor.decode(out[0], skip_special_tokens=True)
            return description
        except Exception as e:
            self.logger.error(f"Scene description failed: {e}")
            return ""
            
    async def _detect_ui_elements(self, image: Image.Image) -> List[UIElement]:
        """Detect UI elements in the screen"""
        elements = []
        
        try:
            # Convert to OpenCV format for processing
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Simple edge detection for UI elements
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours (potential UI elements)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size (avoid tiny elements)
                if w > 20 and h > 10:
                    # Extract text from element region
                    element_region = image.crop((x, y, x + w, y + h))
                    element_text = ""
                    try:
                        element_text = pytesseract.image_to_string(element_region, config='--psm 8').strip()
                    except:
                        pass
                    
                    # Determine element type heuristically
                    element_type = self._classify_ui_element(element_text, w, h)
                    
                    element = UIElement(
                        element_type=element_type,
                        bounds=(x, y, w, h),
                        text=element_text,
                        description=f"{element_type} at ({x},{y})",
                        clickable=element_type in ['button', 'link', 'menu'],
                        automatable=True,
                        confidence=0.7
                    )
                    
                    elements.append(element)
                    
        except Exception as e:
            self.logger.error(f"UI element detection failed: {e}")
            
        return elements[:20]  # Limit to prevent overwhelming
        
    def _classify_ui_element(self, text: str, width: int, height: int) -> str:
        """Classify UI element type based on text and dimensions"""
        text_lower = text.lower()
        
        # Button indicators
        if any(word in text_lower for word in ['click', 'submit', 'ok', 'cancel', 'save', 'send']):
            return 'button'
        
        # Input field indicators
        if height < 50 and width > 100:
            if any(word in text_lower for word in ['enter', 'type', 'search', '@', '.com']):
                return 'input'
        
        # Menu indicators
        if 'menu' in text_lower or '▼' in text or '>' in text:
            return 'menu'
            
        # Link indicators
        if any(word in text_lower for word in ['http', 'www', 'link', 'more']):
            return 'link'
            
        # Default classification
        if text.strip():
            return 'text'
        else:
            return 'image'
            
    async def _detect_automation_opportunities(self, analysis: Dict) -> Optional[Dict[str, Any]]:
        """Detect potential automation opportunities"""
        opportunities = []
        
        # Look for repetitive patterns
        ui_elements = analysis.get('ui_elements', [])
        text_content = analysis.get('text', '')
        
        # Form automation opportunities
        if any(elem.element_type == 'input' for elem in ui_elements):
            if any(word in text_content.lower() for word in ['form', 'register', 'login', 'signup']):
                opportunities.append({
                    'type': 'form_automation',
                    'description': 'Detected form that could be automated',
                    'confidence': 0.8,
                    'elements': [elem for elem in ui_elements if elem.element_type in ['input', 'button']]
                })
        
        # Data entry opportunities
        if text_content and any(elem.element_type == 'input' for elem in ui_elements):
            opportunities.append({
                'type': 'data_entry',
                'description': 'Data entry workflow detected',
                'confidence': 0.7,
                'potential_fields': len([e for e in ui_elements if e.element_type == 'input'])
            })
        
        # Navigation automation
        if any(elem.element_type in ['button', 'link', 'menu'] for elem in ui_elements):
            opportunities.append({
                'type': 'navigation',
                'description': 'Navigation workflow possible',
                'confidence': 0.6,
                'clickable_elements': len([e for e in ui_elements if e.clickable])
            })
        
        if opportunities:
            self.automation_opportunities += 1
            return {
                'opportunities': opportunities,
                'total_score': sum(op['confidence'] for op in opportunities),
                'priority': 'high' if len(opportunities) > 1 else 'medium'
            }
        
        return None
        
    def _add_screen_event(self, event: ScreenEvent):
        """Add a screen event to memory"""
        self.screen_events.append(event)
        
        # Maintain memory limit
        if len(self.screen_events) > self.config['max_events_memory']:
            self.screen_events = self.screen_events[-self.config['max_events_memory']:]
            
    async def _update_automation_patterns(self, event: ScreenEvent):
        """Update automation patterns based on new event"""
        if not event.automation_opportunity:
            return
            
        for opportunity in event.automation_opportunity.get('opportunities', []):
            pattern_key = opportunity['type']
            
            if pattern_key not in self.automation_patterns:
                self.automation_patterns[pattern_key] = []
                
            self.automation_patterns[pattern_key].append(event)
            
            # Maintain pattern memory
            if len(self.automation_patterns[pattern_key]) > 20:
                self.automation_patterns[pattern_key] = self.automation_patterns[pattern_key][-20:]
        
    async def get_automation_recommendations(self) -> List[Dict[str, Any]]:
        """Get current automation recommendations"""
        recommendations = []
        
        for pattern_type, events in self.automation_patterns.items():
            if len(events) >= self.config['automation_pattern_min_occurrences']:
                # Calculate pattern strength
                avg_confidence = sum(e.automation_opportunity.get('total_score', 0) for e in events) / len(events)
                
                recommendation = {
                    'pattern_type': pattern_type,
                    'occurrences': len(events),
                    'confidence': avg_confidence,
                    'priority': 'high' if len(events) > 5 else 'medium',
                    'description': f"Detected {len(events)} instances of {pattern_type}",
                    'automation_potential': min(avg_confidence * len(events) / 10, 1.0),
                    'last_seen': events[-1].timestamp.isoformat(),
                    'sample_elements': [
                        {
                            'text': event.text_content[:100],
                            'timestamp': event.timestamp.isoformat()
                        }
                        for event in events[-3:]  # Last 3 examples
                    ]
                }
                
                recommendations.append(recommendation)
        
        # Sort by automation potential
        recommendations.sort(key=lambda x: x['automation_potential'], reverse=True)
        return recommendations
        
    async def get_current_screen_analysis(self) -> Dict[str, Any]:
        """Get analysis of current screen"""
        if not self.current_screen:
            return {'error': 'No screen captured yet'}
            
        # Analyze current screen
        await self._analyze_screen(self.current_screen)
        
        # Get latest event
        latest_event = self.screen_events[-1] if self.screen_events else None
        
        return {
            'screen_resolution': self.current_screen.size,
            'latest_analysis': {
                'text_content': latest_event.text_content if latest_event else '',
                'scene_description': latest_event.image_description if latest_event else '',
                'automation_opportunities': latest_event.automation_opportunity if latest_event else None
            },
            'ui_elements_detected': len(self.ui_elements),
            'automation_patterns': len(self.automation_patterns),
            'total_events': len(self.screen_events)
        }
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'screenshots_analyzed': self.screenshots_analyzed,
            'events_detected': self.events_detected,
            'automation_opportunities_found': self.automation_opportunities,
            'active_patterns': len(self.automation_patterns),
            'power_multiplier_achieved': self.power_multiplier,
            'monitoring_active': not self.stop_monitoring,
            'memory_usage': {
                'screen_events': len(self.screen_events),
                'ui_elements': len(self.ui_elements),
                'automation_patterns': sum(len(events) for events in self.automation_patterns.values())
            }
        }
        
    async def export_intelligence_data(self, filepath: str):
        """Export collected intelligence data"""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'configuration': self.config,
            'performance_stats': self.get_performance_stats(),
            'automation_patterns': {
                pattern: [
                    {
                        'timestamp': event.timestamp.isoformat(),
                        'event_type': event.event_type,
                        'text_content': event.text_content[:200],  # Truncate for size
                        'confidence': event.confidence,
                        'automation_opportunity': event.automation_opportunity
                    }
                    for event in events
                ]
                for pattern, events in self.automation_patterns.items()
            },
            'recommendations': await self.get_automation_recommendations()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        self.logger.info(f"✅ Intelligence data exported to {filepath}")
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the screen intelligence monitor"""
        return {
            "name": "Real-Time Screen Intelligence",
            "active": self.active,
            "monitoring": not self.stop_monitoring if hasattr(self, 'stop_monitoring') else False,
            "power_multiplier": self.power_multiplier,
            "phase": "intelligence_amplification",
            "performance": self.get_performance_stats(),
            "capabilities": {
                "ocr_enabled": self.config['ocr_enabled'],
                "vision_model_enabled": self.config['vision_model_enabled'],
                "ui_detection_enabled": self.config['ui_detection_enabled']
            }
        }

# Factory function for easy import
def create_real_time_screen_intelligence():
    return RealTimeScreenIntelligence()

# Example usage and testing
async def main():
    """Test the screen intelligence monitor"""
    monitor = RealTimeScreenIntelligence()
    
    try:
        # Initialize
        await monitor.initialize()
        
        # Start monitoring for a short period
        monitor.start_monitoring()
        print("🔍 Monitoring screen for 30 seconds...")
        await asyncio.sleep(30)
        
        # Stop monitoring
        monitor.stop_monitoring_process()
        
        # Get results
        recommendations = await monitor.get_automation_recommendations()
        current_analysis = await monitor.get_current_screen_analysis()
        stats = monitor.get_performance_stats()
        
        print("\n📊 Results:")
        print(f"Screenshots analyzed: {stats['screenshots_analyzed']}")
        print(f"Events detected: {stats['events_detected']}")
        print(f"Automation opportunities: {stats['automation_opportunities_found']}")
        print(f"Patterns discovered: {len(recommendations)}")
        
        if recommendations:
            print("\n🎯 Top Automation Recommendations:")
            for rec in recommendations[:3]:
                print(f"  • {rec['pattern_type']}: {rec['description']} "
                      f"(Confidence: {rec['confidence']:.2f})")
        
        # Export data
        await monitor.export_intelligence_data("screen_intelligence_export.json")
        
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
        monitor.stop_monitoring_process()
    except Exception as e:
        print(f"❌ Error: {e}")
        monitor.stop_monitoring_process()

if __name__ == "__main__":
    asyncio.run(main())
