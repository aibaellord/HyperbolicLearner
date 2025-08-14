#!/bin/bash
# HyperbolicLearner Web UI Launch Script

echo "🚀 Starting HyperbolicLearner Web UI..."
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if Flask is available
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask..."
    python3 -m pip install flask
fi

# Navigate to the project directory
cd "$(dirname "$0")"

echo "🌐 Web UI will be available at:"
echo "   Local:    http://localhost:5001"
echo "   Network:  http://$(ipconfig getifaddr en0 2>/dev/null || hostname -I | cut -d' ' -f1):5001"
echo ""
echo "✨ Features:"
echo "   📊 Real-time system monitoring"
echo "   🤖 Interactive automation controls"
echo "   💼 Business opportunity generator"
echo "   🌟 System capability overview"
echo "   📜 Live activity logging"
echo "   ⚡ Quick actions and testing"
echo ""
echo "🛡️ Browser safety: Only ONE Chrome instance will be used"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================="

# Start the web UI
python3 simple_web_ui.py
