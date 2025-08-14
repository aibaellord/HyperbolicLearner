#!/bin/bash
# HyperbolicLearner Browser Cleanup Script

echo "🧹 Cleaning up HyperbolicLearner browsers..."

# Kill chromedriver
pkill -f chromedriver 2>/dev/null

# Kill automated Chrome instances
ps aux | grep -i chrome | grep -E "(automation|webdriver|test-type)" | awk '{print $2}' | xargs kill -9 2>/dev/null

echo "✅ Cleanup complete"
