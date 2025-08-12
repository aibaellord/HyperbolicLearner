"""
N8N Workflow Automation Integration for HyperbolicLearner

This module provides seamless integration between HyperbolicLearner's AI capabilities
and N8N's workflow automation platform, creating a powerful automation ecosystem.
"""

import asyncio
import json
import requests
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import os
import subprocess
import time

@dataclass
class WorkflowExecution:
    """Represents an n8n workflow execution result"""
    workflow_id: str
    execution_id: str
    status: str
    started_at: datetime
    finished_at: Optional[datetime]
    data: Dict[str, Any]
    error: Optional[str] = None

class N8NIntegrationManager:
    """
    Manages integration between HyperbolicLearner and N8N workflows
    
    This class provides:
    - Automated workflow creation based on learned UI patterns
    - Execution of n8n workflows triggered by HyperbolicLearner events
    - Bidirectional data flow between systems
    - Intelligent workflow optimization based on success rates
    """
    
    def __init__(self, 
                 n8n_host: str = "localhost",
                 n8n_port: int = 5678,
                 api_key: Optional[str] = None):
        self.n8n_host = n8n_host
        self.n8n_port = n8n_port
        self.api_key = api_key
        self.base_url = f"http://{n8n_host}:{n8n_port}/api/v1"
        self.logger = logging.getLogger(__name__)
        
        # Headers for API requests
        self.headers = {
            'Content-Type': 'application/json',
        }
        if api_key:
            self.headers['X-N8N-API-KEY'] = api_key
    
    async def start_n8n_server(self) -> bool:
        """Start n8n server if not already running"""
        try:
            # Check if n8n is already running
            response = requests.get(f"{self.base_url}/workflows", timeout=5)
            if response.status_code == 200:
                self.logger.info("N8N server is already running")
                return True
        except requests.exceptions.RequestException:
            pass
        
        try:
            # Start n8n in the background
            self.logger.info("Starting N8N server...")
            process = subprocess.Popen(
                ['n8n', 'start', '--tunnel'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(os.environ, N8N_PORT=str(self.n8n_port))
            )
            
            # Wait for server to start
            for _ in range(30):  # Wait up to 30 seconds
                try:
                    response = requests.get(f"{self.base_url}/workflows", timeout=2)
                    if response.status_code == 200:
                        self.logger.info("N8N server started successfully")
                        return True
                except requests.exceptions.RequestException:
                    pass
                time.sleep(1)
            
            self.logger.error("Failed to start N8N server")
            return False
            
        except Exception as e:
            self.logger.error(f"Error starting N8N server: {e}")
            return False
    
    def create_workflow_from_ui_actions(self, 
                                       ui_actions: List[Dict[str, Any]], 
                                       workflow_name: str) -> Optional[str]:
        """
        Create an n8n workflow based on UI actions learned by HyperbolicLearner
        
        Args:
            ui_actions: List of UI actions extracted from video tutorials
            workflow_name: Name for the new workflow
            
        Returns:
            Workflow ID if successful, None otherwise
        """
        
        # Convert HyperbolicLearner UI actions to n8n workflow nodes
        nodes = self._convert_ui_actions_to_nodes(ui_actions)
        
        workflow_data = {
            "name": workflow_name,
            "nodes": nodes,
            "connections": self._create_node_connections(nodes),
            "active": True,
            "settings": {
                "executionOrder": "v1"
            },
            "tags": ["hyperbolic-learner", "auto-generated"]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/workflows",
                headers=self.headers,
                json=workflow_data
            )
            
            if response.status_code == 201:
                workflow = response.json()
                workflow_id = workflow.get('id')
                self.logger.info(f"Created workflow '{workflow_name}' with ID: {workflow_id}")
                return workflow_id
            else:
                self.logger.error(f"Failed to create workflow: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating workflow: {e}")
            return None
    
    def _convert_ui_actions_to_nodes(self, ui_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert HyperbolicLearner UI actions to n8n workflow nodes"""
        nodes = []
        
        # Add trigger node
        nodes.append({
            "parameters": {},
            "name": "Manual Trigger",
            "type": "n8n-nodes-base.manualTrigger",
            "typeVersion": 1,
            "position": [240, 300]
        })
        
        # Convert each UI action to appropriate n8n nodes
        for i, action in enumerate(ui_actions):
            node_position = [240 + (i + 1) * 200, 300]
            
            if action.get('type') == 'click':
                nodes.append(self._create_click_node(action, i + 1, node_position))
            elif action.get('type') == 'type':
                nodes.append(self._create_type_node(action, i + 1, node_position))
            elif action.get('type') == 'wait':
                nodes.append(self._create_wait_node(action, i + 1, node_position))
            elif action.get('type') == 'screenshot':
                nodes.append(self._create_screenshot_node(action, i + 1, node_position))
        
        return nodes
    
    def _create_click_node(self, action: Dict[str, Any], index: int, position: List[int]) -> Dict[str, Any]:
        """Create a click action node for n8n"""
        return {
            "parameters": {
                "functionCode": f"""
                // Click action learned from HyperbolicLearner
                const element = await page.waitForSelector('{action.get('selector', '')}');
                await element.click();
                console.log('Clicked element: {action.get('element_description', 'Unknown')}');
                return {{'success': true, 'action': 'click'}};
                """
            },
            "name": f"Click_{index}",
            "type": "n8n-nodes-base.function",
            "typeVersion": 1,
            "position": position
        }
    
    def _create_type_node(self, action: Dict[str, Any], index: int, position: List[int]) -> Dict[str, Any]:
        """Create a type action node for n8n"""
        return {
            "parameters": {
                "functionCode": f"""
                // Type action learned from HyperbolicLearner
                const element = await page.waitForSelector('{action.get('selector', '')}');
                await element.type('{action.get('text', '')}');
                console.log('Typed text: {action.get('text', '')}');
                return {{'success': true, 'action': 'type'}};
                """
            },
            "name": f"Type_{index}",
            "type": "n8n-nodes-base.function",
            "typeVersion": 1,
            "position": position
        }
    
    def _create_wait_node(self, action: Dict[str, Any], index: int, position: List[int]) -> Dict[str, Any]:
        """Create a wait action node for n8n"""
        wait_time = action.get('duration', 1000)
        return {
            "parameters": {
                "amount": wait_time,
                "unit": "milliseconds"
            },
            "name": f"Wait_{index}",
            "type": "n8n-nodes-base.wait",
            "typeVersion": 1,
            "position": position
        }
    
    def _create_screenshot_node(self, action: Dict[str, Any], index: int, position: List[int]) -> Dict[str, Any]:
        """Create a screenshot action node for n8n"""
        return {
            "parameters": {
                "functionCode": """
                // Take screenshot for verification
                const screenshot = await page.screenshot({encoding: 'base64'});
                console.log('Screenshot taken');
                return {'success': true, 'action': 'screenshot', 'data': screenshot};
                """
            },
            "name": f"Screenshot_{index}",
            "type": "n8n-nodes-base.function",
            "typeVersion": 1,
            "position": position
        }
    
    def _create_node_connections(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create connections between workflow nodes"""
        connections = {}
        
        for i in range(len(nodes) - 1):
            current_node = nodes[i]['name']
            next_node = nodes[i + 1]['name']
            
            connections[current_node] = {
                "main": [[{"node": next_node, "type": "main", "index": 0}]]
            }
        
        return connections
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any] = None) -> WorkflowExecution:
        """Execute an n8n workflow with optional input data"""
        try:
            execution_data = {
                "workflowData": {"id": workflow_id}
            }
            
            if input_data:
                execution_data["runData"] = input_data
            
            response = requests.post(
                f"{self.base_url}/executions",
                headers=self.headers,
                json=execution_data
            )
            
            if response.status_code == 201:
                execution = response.json()
                execution_id = execution.get('id')
                
                # Wait for execution to complete
                return await self._wait_for_execution(execution_id)
            else:
                self.logger.error(f"Failed to execute workflow: {response.text}")
                return WorkflowExecution(
                    workflow_id=workflow_id,
                    execution_id="",
                    status="failed",
                    started_at=datetime.now(),
                    finished_at=datetime.now(),
                    data={},
                    error=f"Failed to start execution: {response.text}"
                )
                
        except Exception as e:
            self.logger.error(f"Error executing workflow: {e}")
            return WorkflowExecution(
                workflow_id=workflow_id,
                execution_id="",
                status="error",
                started_at=datetime.now(),
                finished_at=datetime.now(),
                data={},
                error=str(e)
            )
    
    async def _wait_for_execution(self, execution_id: str) -> WorkflowExecution:
        """Wait for workflow execution to complete"""
        start_time = datetime.now()
        
        while True:
            try:
                response = requests.get(
                    f"{self.base_url}/executions/{execution_id}",
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    execution = response.json()
                    status = execution.get('finished', False)
                    
                    if status or execution.get('stoppedAt'):
                        return WorkflowExecution(
                            workflow_id=execution.get('workflowId', ''),
                            execution_id=execution_id,
                            status="success" if execution.get('finished') else "failed",
                            started_at=start_time,
                            finished_at=datetime.now(),
                            data=execution.get('data', {}),
                            error=execution.get('error')
                        )
                    
                    # Wait before checking again
                    await asyncio.sleep(1)
                else:
                    self.logger.error(f"Failed to get execution status: {response.text}")
                    break
                    
            except Exception as e:
                self.logger.error(f"Error waiting for execution: {e}")
                break
        
        return WorkflowExecution(
            workflow_id="",
            execution_id=execution_id,
            status="error",
            started_at=start_time,
            finished_at=datetime.now(),
            data={},
            error="Failed to track execution"
        )
    
    def create_hyperbolic_learner_webhook(self) -> Optional[str]:
        """Create a webhook workflow that can receive data from HyperbolicLearner"""
        webhook_workflow = {
            "name": "HyperbolicLearner Webhook Handler",
            "nodes": [
                {
                    "parameters": {
                        "httpMethod": "POST",
                        "path": "hyperbolic-learner",
                        "responseMode": "responseNode",
                        "options": {}
                    },
                    "name": "Webhook",
                    "type": "n8n-nodes-base.webhook",
                    "typeVersion": 1,
                    "position": [240, 300],
                    "webhookId": "hyperbolic-learner-webhook"
                },
                {
                    "parameters": {
                        "functionCode": """
                        // Process data from HyperbolicLearner
                        const data = $input.all()[0].json;
                        console.log('Received data from HyperbolicLearner:', data);
                        
                        // You can add custom processing logic here
                        // For example, trigger different workflows based on data type
                        
                        return [{
                            json: {
                                received: true,
                                timestamp: new Date().toISOString(),
                                processed_data: data
                            }
                        }];
                        """
                    },
                    "name": "Process HyperbolicLearner Data",
                    "type": "n8n-nodes-base.function",
                    "typeVersion": 1,
                    "position": [460, 300]
                },
                {
                    "parameters": {
                        "respondWith": "json",
                        "responseBody": "={{ $json }}"
                    },
                    "name": "Respond",
                    "type": "n8n-nodes-base.respondToWebhook",
                    "typeVersion": 1,
                    "position": [680, 300]
                }
            ],
            "connections": {
                "Webhook": {
                    "main": [[{"node": "Process HyperbolicLearner Data", "type": "main", "index": 0}]]
                },
                "Process HyperbolicLearner Data": {
                    "main": [[{"node": "Respond", "type": "main", "index": 0}]]
                }
            },
            "active": True,
            "settings": {
                "executionOrder": "v1"
            },
            "tags": ["hyperbolic-learner", "webhook"]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/workflows",
                headers=self.headers,
                json=webhook_workflow
            )
            
            if response.status_code == 201:
                workflow = response.json()
                workflow_id = workflow.get('id')
                self.logger.info(f"Created webhook workflow with ID: {workflow_id}")
                return workflow_id
            else:
                self.logger.error(f"Failed to create webhook workflow: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating webhook workflow: {e}")
            return None
    
    def send_to_webhook(self, data: Dict[str, Any], webhook_path: str = "hyperbolic-learner") -> bool:
        """Send data to n8n webhook from HyperbolicLearner"""
        try:
            webhook_url = f"http://{self.n8n_host}:{self.n8n_port}/webhook/{webhook_path}"
            
            response = requests.post(webhook_url, json=data, timeout=30)
            
            if response.status_code == 200:
                self.logger.info("Successfully sent data to n8n webhook")
                return True
            else:
                self.logger.error(f"Failed to send data to webhook: {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending data to webhook: {e}")
            return False


# Integration helper functions for HyperbolicLearner components

class HyperbolicN8NBridge:
    """Bridge class to integrate N8N with HyperbolicLearner's existing components"""
    
    def __init__(self, hyperbolic_app, n8n_manager: N8NIntegrationManager):
        self.hyperbolic_app = hyperbolic_app
        self.n8n_manager = n8n_manager
        self.logger = logging.getLogger(__name__)
    
    async def auto_create_workflows_from_learning(self, video_url: str) -> List[str]:
        """
        Automatically create n8n workflows from video learning sessions
        
        This method:
        1. Uses HyperbolicLearner to extract UI actions from video
        2. Creates corresponding n8n workflows
        3. Returns list of created workflow IDs
        """
        workflow_ids = []
        
        try:
            # Learn from video using HyperbolicLearner
            self.logger.info(f"Learning from video: {video_url}")
            knowledge_id = await self.hyperbolic_app.learn_from_youtube(
                url=video_url,
                extract_ui_actions=True,
                semantic_compression=True
            )
            
            # Extract UI actions from learned knowledge
            ui_actions = self.hyperbolic_app.get_ui_actions(knowledge_id)
            
            if ui_actions:
                # Group related actions into workflows
                workflow_groups = self._group_actions_into_workflows(ui_actions)
                
                for i, action_group in enumerate(workflow_groups):
                    workflow_name = f"Learned_Workflow_{knowledge_id}_{i+1}"
                    workflow_id = self.n8n_manager.create_workflow_from_ui_actions(
                        action_group['actions'], 
                        workflow_name
                    )
                    
                    if workflow_id:
                        workflow_ids.append(workflow_id)
                        self.logger.info(f"Created workflow: {workflow_name} ({workflow_id})")
            
            return workflow_ids
            
        except Exception as e:
            self.logger.error(f"Error creating workflows from learning: {e}")
            return workflow_ids
    
    def _group_actions_into_workflows(self, ui_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group UI actions into logical workflow sequences"""
        # This is a simplified grouping - you can enhance this with ML-based clustering
        workflows = []
        current_workflow = {"actions": [], "context": ""}
        
        for action in ui_actions:
            # Simple heuristic: group actions by application or major UI changes
            if (action.get('application') != current_workflow.get('context') and 
                current_workflow['actions']):
                workflows.append(current_workflow)
                current_workflow = {"actions": [action], "context": action.get('application', '')}
            else:
                current_workflow['actions'].append(action)
                current_workflow['context'] = action.get('application', '')
        
        if current_workflow['actions']:
            workflows.append(current_workflow)
        
        return workflows


# Example usage and integration
async def main():
    """Example of how to use the N8N integration with HyperbolicLearner"""
    
    # Initialize the integration
    n8n_manager = N8NIntegrationManager()
    
    # Start n8n server
    await n8n_manager.start_n8n_server()
    
    # Create webhook for receiving data from HyperbolicLearner
    webhook_id = n8n_manager.create_hyperbolic_learner_webhook()
    
    # Example: Create workflow from UI actions (would come from HyperbolicLearner)
    sample_ui_actions = [
        {
            "type": "click",
            "selector": "#login-button",
            "element_description": "Login button",
            "application": "web_browser"
        },
        {
            "type": "type", 
            "selector": "#username",
            "text": "user@example.com",
            "application": "web_browser"
        },
        {
            "type": "wait",
            "duration": 2000,
            "application": "web_browser"
        }
    ]
    
    workflow_id = n8n_manager.create_workflow_from_ui_actions(
        sample_ui_actions,
        "Sample_Login_Workflow"
    )
    
    if workflow_id:
        print(f"Created workflow with ID: {workflow_id}")
        
        # Execute the workflow
        execution_result = await n8n_manager.execute_workflow(workflow_id)
        print(f"Execution result: {execution_result.status}")


if __name__ == "__main__":
    asyncio.run(main())
