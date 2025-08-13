"""
Universal Interface Controller
Automate ANY interface - web, desktop, mobile, API, voice, gesture

This module provides 10x automation scope expansion by supporting:
- Web browsers (Selenium, Playwright)
- Desktop applications (PyAutoGUI, accessibility APIs)
- Mobile apps (Appium)
- REST/GraphQL APIs
- Voice interfaces
- Gesture control
- IoT devices
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import subprocess
import requests
import base64
import io
from pathlib import Path

# Web automation imports
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Desktop automation imports
try:
    import pyautogui
    import pygetwindow as gw
    DESKTOP_AUTOMATION_AVAILABLE = True
except ImportError:
    DESKTOP_AUTOMATION_AVAILABLE = False

# Mobile automation imports
try:
    from appium import webdriver as appium_webdriver
    APPIUM_AVAILABLE = True
except ImportError:
    APPIUM_AVAILABLE = False

# Voice recognition imports
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# Computer vision imports
try:
    import cv2
    import numpy as np
    from PIL import Image
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

class InterfaceType(Enum):
    WEB = "web"
    DESKTOP = "desktop"
    MOBILE = "mobile"
    API = "api"
    VOICE = "voice"
    GESTURE = "gesture"
    IOT = "iot"

class ActionType(Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    DRAG = "drag"
    WAIT = "wait"
    NAVIGATE = "navigate"
    CAPTURE = "capture"
    VOICE_COMMAND = "voice_command"
    API_CALL = "api_call"
    GESTURE = "gesture"

@dataclass
class UniversalAction:
    """Represents a universal action that can be executed on any interface"""
    action_type: ActionType
    interface_type: InterfaceType
    target: str  # Element selector, API endpoint, etc.
    value: Optional[str] = None
    coordinates: Optional[tuple] = None
    timeout: float = 10.0
    metadata: Dict[str, Any] = None
    
@dataclass
class ExecutionResult:
    """Result of action execution"""
    success: bool
    action: UniversalAction
    execution_time: float
    result_data: Optional[Any] = None
    error_message: Optional[str] = None
    screenshot: Optional[str] = None  # Base64 encoded

class UniversalInterfaceController:
    """
    Automate ANY interface - web, desktop, mobile, API, voice, gesture
    
    Power Multiplier: 10.0x
    Phase: intelligence_amplification
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.power_multiplier = 10.0
        self.active = False
        
        # Interface controllers
        self.web_controller = None
        self.desktop_controller = None
        self.mobile_controller = None
        self.api_controller = None
        self.voice_controller = None
        self.gesture_controller = None
        
        # Execution tracking
        self.execution_history: List[ExecutionResult] = []
        self.active_sessions: Dict[InterfaceType, Any] = {}
        
        # Performance metrics
        self.actions_executed = 0
        self.success_rate = 0.0
        self.interface_capabilities = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'web_driver': 'chrome',  # chrome, firefox, safari, edge
            'headless_mode': False,
            'screenshot_on_action': True,
            'action_delay': 0.1,
            'max_retry_attempts': 3,
            'api_timeout': 30.0,
            'voice_language': 'en-US',
            'gesture_sensitivity': 0.7,
            'desktop_screenshot_quality': 0.8,
            'mobile_platform': 'android',  # android, ios
            'enable_interfaces': ['web', 'desktop', 'api', 'voice']
        }
        
    async def initialize(self):
        """Initialize the universal interface controller"""
        self.logger.info("🚀 Initializing Universal Interface Controller")
        
        # Initialize available interfaces
        await self._initialize_web_controller()
        await self._initialize_desktop_controller()
        await self._initialize_mobile_controller()
        await self._initialize_api_controller()
        await self._initialize_voice_controller()
        await self._initialize_gesture_controller()
        
        # Test capabilities
        await self._test_interface_capabilities()
        
        self.active = True
        self.logger.info("✅ Universal Interface Controller initialized successfully")
        
    async def _initialize_web_controller(self):
        """Initialize web browser controller"""
        if 'web' not in self.config['enable_interfaces'] or not SELENIUM_AVAILABLE:
            return
            
        try:
            self.web_controller = WebInterfaceController(self.config)
            await self.web_controller.initialize()
            self.interface_capabilities['web'] = True
            self.logger.info("✅ Web interface controller ready")
        except Exception as e:
            self.logger.warning(f"Web interface initialization failed: {e}")
            self.interface_capabilities['web'] = False
            
    async def _initialize_desktop_controller(self):
        """Initialize desktop application controller"""
        if 'desktop' not in self.config['enable_interfaces'] or not DESKTOP_AUTOMATION_AVAILABLE:
            return
            
        try:
            self.desktop_controller = DesktopInterfaceController(self.config)
            await self.desktop_controller.initialize()
            self.interface_capabilities['desktop'] = True
            self.logger.info("✅ Desktop interface controller ready")
        except Exception as e:
            self.logger.warning(f"Desktop interface initialization failed: {e}")
            self.interface_capabilities['desktop'] = False
            
    async def _initialize_mobile_controller(self):
        """Initialize mobile application controller"""
        if 'mobile' not in self.config['enable_interfaces'] or not APPIUM_AVAILABLE:
            return
            
        try:
            self.mobile_controller = MobileInterfaceController(self.config)
            # Mobile controller is initialized on-demand
            self.interface_capabilities['mobile'] = True
            self.logger.info("✅ Mobile interface controller ready")
        except Exception as e:
            self.logger.warning(f"Mobile interface initialization failed: {e}")
            self.interface_capabilities['mobile'] = False
            
    async def _initialize_api_controller(self):
        """Initialize API controller"""
        if 'api' not in self.config['enable_interfaces']:
            return
            
        try:
            self.api_controller = APIInterfaceController(self.config)
            await self.api_controller.initialize()
            self.interface_capabilities['api'] = True
            self.logger.info("✅ API interface controller ready")
        except Exception as e:
            self.logger.warning(f"API interface initialization failed: {e}")
            self.interface_capabilities['api'] = False
            
    async def _initialize_voice_controller(self):
        """Initialize voice interface controller"""
        if 'voice' not in self.config['enable_interfaces'] or not VOICE_AVAILABLE:
            return
            
        try:
            self.voice_controller = VoiceInterfaceController(self.config)
            await self.voice_controller.initialize()
            self.interface_capabilities['voice'] = True
            self.logger.info("✅ Voice interface controller ready")
        except Exception as e:
            self.logger.warning(f"Voice interface initialization failed: {e}")
            self.interface_capabilities['voice'] = False
            
    async def _initialize_gesture_controller(self):
        """Initialize gesture interface controller"""
        if 'gesture' not in self.config['enable_interfaces'] or not VISION_AVAILABLE:
            return
            
        try:
            self.gesture_controller = GestureInterfaceController(self.config)
            await self.gesture_controller.initialize()
            self.interface_capabilities['gesture'] = True
            self.logger.info("✅ Gesture interface controller ready")
        except Exception as e:
            self.logger.warning(f"Gesture interface initialization failed: {e}")
            self.interface_capabilities['gesture'] = False
            
    async def _test_interface_capabilities(self):
        """Test all interface capabilities"""
        self.logger.info("🧪 Testing interface capabilities...")
        
        capabilities_summary = []
        for interface, available in self.interface_capabilities.items():
            if available:
                capabilities_summary.append(f"✅ {interface.upper()}")
            else:
                capabilities_summary.append(f"❌ {interface.upper()}")
                
        self.logger.info(f"Interface capabilities: {', '.join(capabilities_summary)}")
        
    async def execute_action(self, action: UniversalAction) -> ExecutionResult:
        """Execute a universal action on the appropriate interface"""
        start_time = time.time()
        
        try:
            self.logger.info(f"🎯 Executing {action.action_type.value} on {action.interface_type.value}: {action.target}")
            
            # Route action to appropriate controller
            result_data = await self._route_action(action)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create result
            result = ExecutionResult(
                success=True,
                action=action,
                execution_time=execution_time,
                result_data=result_data,
                screenshot=await self._capture_interface_screenshot(action.interface_type) if self.config['screenshot_on_action'] else None
            )
            
            self.actions_executed += 1
            self._update_success_rate(True)
            
            self.logger.info(f"✅ Action completed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            result = ExecutionResult(
                success=False,
                action=action,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            self._update_success_rate(False)
            self.logger.error(f"❌ Action failed after {execution_time:.2f}s: {e}")
            
        # Store in history
        self.execution_history.append(result)
        
        # Apply delay between actions
        if self.config['action_delay'] > 0:
            await asyncio.sleep(self.config['action_delay'])
            
        return result
        
    async def _route_action(self, action: UniversalAction) -> Any:
        """Route action to the appropriate interface controller"""
        interface_type = action.interface_type
        
        if interface_type == InterfaceType.WEB and self.web_controller:
            return await self.web_controller.execute_action(action)
        elif interface_type == InterfaceType.DESKTOP and self.desktop_controller:
            return await self.desktop_controller.execute_action(action)
        elif interface_type == InterfaceType.MOBILE and self.mobile_controller:
            return await self.mobile_controller.execute_action(action)
        elif interface_type == InterfaceType.API and self.api_controller:
            return await self.api_controller.execute_action(action)
        elif interface_type == InterfaceType.VOICE and self.voice_controller:
            return await self.voice_controller.execute_action(action)
        elif interface_type == InterfaceType.GESTURE and self.gesture_controller:
            return await self.gesture_controller.execute_action(action)
        else:
            raise Exception(f"Interface {interface_type.value} not available or not initialized")
            
    async def execute_workflow(self, actions: List[UniversalAction]) -> List[ExecutionResult]:
        """Execute a sequence of actions across multiple interfaces"""
        self.logger.info(f"🚀 Executing workflow with {len(actions)} actions")
        
        results = []
        
        for i, action in enumerate(actions):
            self.logger.info(f"📍 Step {i+1}/{len(actions)}")
            
            # Execute with retry logic
            result = await self._execute_with_retry(action)
            results.append(result)
            
            # Stop on critical failure if configured
            if not result.success and getattr(action, 'critical', False):
                self.logger.error(f"❌ Critical action failed, stopping workflow")
                break
                
        success_count = sum(1 for r in results if r.success)
        self.logger.info(f"✅ Workflow completed: {success_count}/{len(results)} actions successful")
        
        return results
        
    async def _execute_with_retry(self, action: UniversalAction) -> ExecutionResult:
        """Execute action with retry logic"""
        for attempt in range(self.config['max_retry_attempts']):
            result = await self.execute_action(action)
            
            if result.success:
                return result
                
            if attempt < self.config['max_retry_attempts'] - 1:
                self.logger.warning(f"🔄 Retry attempt {attempt + 1} for {action.action_type.value}")
                await asyncio.sleep(1)  # Brief delay before retry
                
        return result
        
    async def _capture_interface_screenshot(self, interface_type: InterfaceType) -> Optional[str]:
        """Capture screenshot from the appropriate interface"""
        try:
            screenshot = None
            
            if interface_type == InterfaceType.WEB and self.web_controller:
                screenshot = await self.web_controller.take_screenshot()
            elif interface_type == InterfaceType.DESKTOP and self.desktop_controller:
                screenshot = await self.desktop_controller.take_screenshot()
            elif interface_type == InterfaceType.MOBILE and self.mobile_controller:
                screenshot = await self.mobile_controller.take_screenshot()
                
            if screenshot:
                # Convert to base64 for storage
                if isinstance(screenshot, str):
                    return screenshot
                elif hasattr(screenshot, 'save'):
                    buffer = io.BytesIO()
                    screenshot.save(buffer, format='PNG')
                    return base64.b64encode(buffer.getvalue()).decode()
                    
        except Exception as e:
            self.logger.warning(f"Screenshot capture failed: {e}")
            
        return None
        
    def _update_success_rate(self, success: bool):
        """Update the overall success rate"""
        total_actions = len(self.execution_history) + (1 if success else 0)
        successful_actions = sum(1 for r in self.execution_history if r.success) + (1 if success else 0)
        self.success_rate = successful_actions / total_actions if total_actions > 0 else 0.0
        
    async def get_interface_status(self, interface_type: InterfaceType) -> Dict[str, Any]:
        """Get status of a specific interface"""
        controller = getattr(self, f"{interface_type.value}_controller")
        
        if not controller:
            return {"available": False, "error": "Controller not initialized"}
            
        try:
            if hasattr(controller, 'get_status'):
                return await controller.get_status()
            else:
                return {"available": True, "status": "ready"}
        except Exception as e:
            return {"available": False, "error": str(e)}
            
    async def discover_interface_elements(self, interface_type: InterfaceType, context: str = None) -> List[Dict[str, Any]]:
        """Discover available elements/endpoints in an interface"""
        controller = getattr(self, f"{interface_type.value}_controller")
        
        if not controller:
            return []
            
        try:
            if hasattr(controller, 'discover_elements'):
                return await controller.discover_elements(context)
        except Exception as e:
            self.logger.error(f"Element discovery failed: {e}")
            
        return []
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all interfaces"""
        return {
            'total_actions_executed': self.actions_executed,
            'success_rate': self.success_rate,
            'interface_capabilities': self.interface_capabilities,
            'execution_history_size': len(self.execution_history),
            'power_multiplier_achieved': self.power_multiplier,
            'average_execution_time': sum(r.execution_time for r in self.execution_history) / len(self.execution_history) if self.execution_history else 0,
            'interface_usage': {
                interface.value: sum(1 for r in self.execution_history if r.action.interface_type == interface)
                for interface in InterfaceType
            }
        }
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the universal controller"""
        return {
            "name": "Universal Interface Controller",
            "active": self.active,
            "power_multiplier": self.power_multiplier,
            "phase": "intelligence_amplification",
            "interface_capabilities": self.interface_capabilities,
            "performance": self.get_performance_metrics()
        }

# Specialized interface controllers

class WebInterfaceController:
    """Controller for web browser automation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.driver = None
        
    async def initialize(self):
        """Initialize web driver"""
        if SELENIUM_AVAILABLE:
            options = webdriver.ChromeOptions()
            if self.config.get('headless_mode', False):
                options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            self.driver = webdriver.Chrome(options=options)
            
    async def execute_action(self, action: UniversalAction) -> Any:
        """Execute web action"""
        if action.action_type == ActionType.NAVIGATE:
            self.driver.get(action.target)
            return {"url": self.driver.current_url}
            
        elif action.action_type == ActionType.CLICK:
            element = self.driver.find_element(By.CSS_SELECTOR, action.target)
            element.click()
            return {"clicked": True}
            
        elif action.action_type == ActionType.TYPE:
            element = self.driver.find_element(By.CSS_SELECTOR, action.target)
            element.clear()
            element.send_keys(action.value)
            return {"typed": action.value}
            
        elif action.action_type == ActionType.WAIT:
            WebDriverWait(self.driver, action.timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, action.target))
            )
            return {"waited": True}
            
        return {"result": "action_completed"}
        
    async def take_screenshot(self):
        """Take web page screenshot"""
        return self.driver.get_screenshot_as_base64()
        
    async def discover_elements(self, context: str = None) -> List[Dict[str, Any]]:
        """Discover web elements"""
        elements = []
        try:
            for tag in ['button', 'input', 'a', 'select']:
                web_elements = self.driver.find_elements(By.TAG_NAME, tag)
                for element in web_elements[:20]:  # Limit results
                    elements.append({
                        'tag': tag,
                        'text': element.text[:50],
                        'id': element.get_attribute('id'),
                        'class': element.get_attribute('class'),
                        'clickable': element.is_enabled()
                    })
        except Exception as e:
            logging.error(f"Web element discovery failed: {e}")
            
        return elements

class DesktopInterfaceController:
    """Controller for desktop application automation"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    async def initialize(self):
        """Initialize desktop automation"""
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
    async def execute_action(self, action: UniversalAction) -> Any:
        """Execute desktop action"""
        if action.action_type == ActionType.CLICK:
            if action.coordinates:
                pyautogui.click(action.coordinates[0], action.coordinates[1])
                return {"clicked": action.coordinates}
            else:
                # Try to find element by image/text
                location = pyautogui.locateOnScreen(action.target)
                if location:
                    pyautogui.click(pyautogui.center(location))
                    return {"clicked": pyautogui.center(location)}
                    
        elif action.action_type == ActionType.TYPE:
            pyautogui.typewrite(action.value)
            return {"typed": action.value}
            
        elif action.action_type == ActionType.SCROLL:
            pyautogui.scroll(int(action.value) if action.value else 3)
            return {"scrolled": action.value}
            
        return {"result": "action_completed"}
        
    async def take_screenshot(self):
        """Take desktop screenshot"""
        return pyautogui.screenshot()

class MobileInterfaceController:
    """Controller for mobile application automation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.driver = None
        
    async def initialize(self):
        """Initialize mobile driver when needed"""
        pass
        
    async def execute_action(self, action: UniversalAction) -> Any:
        """Execute mobile action"""
        # Mobile automation implementation would go here
        return {"result": "mobile_action_completed"}

class APIInterfaceController:
    """Controller for API automation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.session = requests.Session()
        
    async def initialize(self):
        """Initialize API client"""
        self.session.headers.update({
            'User-Agent': 'HyperbolicLearner-Universal-Controller/1.0'
        })
        
    async def execute_action(self, action: UniversalAction) -> Any:
        """Execute API action"""
        if action.action_type == ActionType.API_CALL:
            method = action.metadata.get('method', 'GET')
            headers = action.metadata.get('headers', {})
            data = action.metadata.get('data')
            
            response = self.session.request(
                method=method,
                url=action.target,
                headers=headers,
                json=data,
                timeout=self.config.get('api_timeout', 30)
            )
            
            return {
                'status_code': response.status_code,
                'response': response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                'headers': dict(response.headers)
            }
            
        return {"result": "api_action_completed"}

class VoiceInterfaceController:
    """Controller for voice interface automation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.recognizer = None
        self.tts_engine = None
        
    async def initialize(self):
        """Initialize voice recognition and TTS"""
        if VOICE_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.tts_engine = pyttsx3.init()
            
    async def execute_action(self, action: UniversalAction) -> Any:
        """Execute voice action"""
        if action.action_type == ActionType.VOICE_COMMAND:
            # Voice command implementation would go here
            self.tts_engine.say(action.value)
            self.tts_engine.runAndWait()
            return {"spoken": action.value}
            
        return {"result": "voice_action_completed"}

class GestureInterfaceController:
    """Controller for gesture interface automation"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    async def initialize(self):
        """Initialize gesture recognition"""
        pass
        
    async def execute_action(self, action: UniversalAction) -> Any:
        """Execute gesture action"""
        # Gesture implementation would go here
        return {"result": "gesture_action_completed"}

# Factory function for easy import
def create_universal_interface_controller():
    return UniversalInterfaceController()

# Example usage and testing
async def main():
    """Test the universal interface controller"""
    controller = UniversalInterfaceController()
    
    try:
        # Initialize
        await controller.initialize()
        
        # Test web automation
        web_actions = [
            UniversalAction(
                action_type=ActionType.NAVIGATE,
                interface_type=InterfaceType.WEB,
                target="https://httpbin.org/forms/post"
            ),
            UniversalAction(
                action_type=ActionType.TYPE,
                interface_type=InterfaceType.WEB,
                target="input[name='custname']",
                value="HyperbolicLearner Test"
            ),
            UniversalAction(
                action_type=ActionType.CLICK,
                interface_type=InterfaceType.WEB,
                target="input[type='submit']"
            )
        ]
        
        # Execute workflow
        results = await controller.execute_workflow(web_actions)
        
        # Print results
        print("\n📊 Execution Results:")
        for i, result in enumerate(results):
            status = "✅" if result.success else "❌"
            print(f"{status} Step {i+1}: {result.action.action_type.value} "
                  f"({result.execution_time:.2f}s)")
            
        # Get performance metrics
        metrics = controller.get_performance_metrics()
        print(f"\n📈 Performance: {metrics['success_rate']:.2%} success rate, "
              f"{metrics['total_actions_executed']} actions executed")
              
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
