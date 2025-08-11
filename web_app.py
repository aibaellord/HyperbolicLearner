#!/usr/bin/env python3
"""
CodeTutor AI Web Interface
==========================

Simple Flask web app to demonstrate CodeTutor AI capabilities
"""

import sys
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
import logging

# Add our modules
sys.path.append('src')
sys.path.append('.')

from codetutor_mvp import CodeTutorAI

# Setup Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global AI instance
ai = None

def initialize_ai():
    """Initialize the AI system"""
    global ai
    if ai is None:
        logger.info("Initializing CodeTutor AI...")
        ai = CodeTutorAI()
        logger.info("CodeTutor AI initialized successfully!")

@app.route('/')
def index():
    """Main page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CodeTutor AI - Learn Programming with AI</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
            .section { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .demo-area { background: white; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
            textarea { width: 100%; height: 120px; font-family: monospace; padding: 10px; border: 1px solid #ccc; border-radius: 4px; }
            button { background: #667eea; color: white; border: none; padding: 12px 24px; border-radius: 4px; cursor: pointer; font-size: 16px; }
            button:hover { background: #5a6fd8; }
            .result { background: #e8f5e8; border: 1px solid #4caf50; padding: 15px; border-radius: 4px; margin-top: 15px; }
            .status { display: flex; justify-content: space-around; text-align: center; }
            .status-item { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 CodeTutor AI</h1>
            <p>AI that learns programming from tutorials and teaches you interactively</p>
        </div>

        <div class="section">
            <h2>🎓 Learn from Tutorial Text</h2>
            <div class="demo-area">
                <p>Paste tutorial text below and watch the AI learn programming concepts:</p>
                <textarea id="tutorial-text" placeholder="Paste your tutorial text here...">In Python, you can create loops using 'for' and 'while'. A for loop iterates over a sequence: for i in range(5): print(i). A while loop continues until a condition is false: while x < 10: x += 1.</textarea>
                <br><br>
                <label for="language-select">Programming Language:</label>
                <select id="language-select">
                    <option value="python">Python</option>
                    <option value="javascript">JavaScript</option>
                </select>
                <br><br>
                <button onclick="learnFromTutorial()">🎓 Learn from Tutorial</button>
                <div id="learning-result"></div>
            </div>
        </div>

        <div class="section">
            <h2>💡 Code Explanation Generator</h2>
            <div class="demo-area">
                <p>Enter code below and get AI-generated explanations:</p>
                <textarea id="code-input" placeholder="Enter your code here...">def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)</textarea>
                <br><br>
                <button onclick="explainCode()">💡 Explain Code</button>
                <div id="explanation-result"></div>
            </div>
        </div>

        <div class="section">
            <h2>📊 AI Status</h2>
            <div class="demo-area">
                <button onclick="getStatus()">📊 Get AI Status</button>
                <div id="status-result"></div>
            </div>
        </div>

        <script>
            async function learnFromTutorial() {
                const text = document.getElementById('tutorial-text').value;
                const language = document.getElementById('language-select').value;
                const button = event.target;
                
                button.disabled = true;
                button.innerHTML = '🔄 Learning...';
                
                try {
                    const response = await fetch('/learn', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            tutorial_text: text,
                            language: language
                        })
                    });
                    
                    const result = await response.json();
                    
                    let html = `
                        <div class="result">
                            <h4>✅ Learning Complete!</h4>
                            <p><strong>Concepts Learned:</strong> ${result.concepts_learned.length}</p>
                            <p><strong>Learning Quality:</strong> ${(result.learning_quality * 100).toFixed(1)}%</p>
                            <p><strong>Processing Time:</strong> ${(result.processing_time * 1000).toFixed(1)}ms</p>
                            <p><strong>Consciousness Level:</strong> ${(result.consciousness_level * 100).toFixed(2)}%</p>
                            <details>
                                <summary>View Learned Concepts</summary>
                                <pre>${JSON.stringify(result.concepts_learned, null, 2)}</pre>
                            </details>
                        </div>
                    `;
                    
                    document.getElementById('learning-result').innerHTML = html;
                } catch (error) {
                    document.getElementById('learning-result').innerHTML = `
                        <div style="background: #ffebee; border: 1px solid #f44336; padding: 15px; border-radius: 4px;">
                            <strong>Error:</strong> ${error.message}
                        </div>
                    `;
                }
                
                button.disabled = false;
                button.innerHTML = '🎓 Learn from Tutorial';
            }

            async function explainCode() {
                const code = document.getElementById('code-input').value;
                const button = event.target;
                
                button.disabled = true;
                button.innerHTML = '🔄 Analyzing...';
                
                try {
                    const response = await fetch('/explain', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            code: code,
                            language: 'python'
                        })
                    });
                    
                    const result = await response.json();
                    
                    let html = `
                        <div class="result">
                            <h4>💡 Code Explanation</h4>
                            <p><strong>Explanation:</strong></p>
                            <p>${result.explanation}</p>
                            <p><strong>Complexity Score:</strong> ${(result.complexity_score * 100).toFixed(1)}%</p>
                            <p><strong>Explanation Quality:</strong> ${(result.explanation_quality * 100).toFixed(1)}%</p>
                            <p><strong>Concepts Identified:</strong> ${result.concepts_identified.join(', ') || 'None detected'}</p>
                        </div>
                    `;
                    
                    document.getElementById('explanation-result').innerHTML = html;
                } catch (error) {
                    document.getElementById('explanation-result').innerHTML = `
                        <div style="background: #ffebee; border: 1px solid #f44336; padding: 15px; border-radius: 4px;">
                            <strong>Error:</strong> ${error.message}
                        </div>
                    `;
                }
                
                button.disabled = false;
                button.innerHTML = '💡 Explain Code';
            }

            async function getStatus() {
                const button = event.target;
                
                button.disabled = true;
                button.innerHTML = '🔄 Loading...';
                
                try {
                    const response = await fetch('/status');
                    const status = await response.json();
                    
                    let html = `
                        <div class="result">
                            <div class="status">
                                <div class="status-item">
                                    <h4>${(status.consciousness_level * 100).toFixed(2)}%</h4>
                                    <p>Consciousness Level</p>
                                </div>
                                <div class="status-item">
                                    <h4>${status.lessons_learned}</h4>
                                    <p>Lessons Learned</p>
                                </div>
                                <div class="status-item">
                                    <h4>${status.languages_known.length}</h4>
                                    <p>Languages Known</p>
                                </div>
                                <div class="status-item">
                                    <h4>${status.evolution_generation}</h4>
                                    <p>Evolution Generation</p>
                                </div>
                                <div class="status-item">
                                    <h4>${status.total_concepts}</h4>
                                    <p>Total Concepts</p>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    document.getElementById('status-result').innerHTML = html;
                } catch (error) {
                    document.getElementById('status-result').innerHTML = `
                        <div style="background: #ffebee; border: 1px solid #f44336; padding: 15px; border-radius: 4px;">
                            <strong>Error:</strong> ${error.message}
                        </div>
                    `;
                }
                
                button.disabled = false;
                button.innerHTML = '📊 Get AI Status';
            }
        </script>
    </body>
    </html>
    """

@app.route('/learn', methods=['POST'])
def learn():
    """Learn from tutorial text"""
    try:
        initialize_ai()
        data = request.get_json()
        
        result = ai.learn_from_tutorial_text(
            data['tutorial_text'], 
            data.get('language', 'python')
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /learn: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/explain', methods=['POST'])
def explain():
    """Explain code"""
    try:
        initialize_ai()
        data = request.get_json()
        
        result = ai.generate_code_explanation(
            data['code'], 
            data.get('language', 'python')
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /explain: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Get AI status"""
    try:
        initialize_ai()
        result = ai.get_status()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in /status: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting CodeTutor AI Web Interface...")
    print("📍 Open your browser to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
