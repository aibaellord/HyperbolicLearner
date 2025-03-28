# HyperbolicLearner Quick Start Guide

This guide provides step-by-step instructions to get HyperbolicLearner up and running on your system.

## Prerequisites

Before installing HyperbolicLearner, ensure you have the following:

* Python 3.8 or higher
* Git
* FFmpeg (for video processing)
* At least 8GB of RAM
* 5GB of free disk space
* CUDA-compatible GPU (recommended for optimal performance)

### Operating System Requirements

* **Linux**: Ubuntu 20.04+ or similar distribution
* **macOS**: 10.15 (Catalina) or higher
* **Windows**: Windows 10/11 with WSL2 recommended

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/HyperbolicLearner.git
cd HyperbolicLearner
```

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
# For Linux/macOS
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Optional GPU Support

For accelerated performance with GPU:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

### 5. Configure the System

```bash
python -m src.core.setup
```

## Basic Usage

### Learning from a YouTube Tutorial

```python
from hyperboliclearner import HyperbolicLearner

# Initialize the system
learner = HyperbolicLearner()

# Learn from a YouTube tutorial
knowledge = learner.learn_from_url("https://www.youtube.com/watch?v=example_tutorial")

# Save the acquired knowledge
knowledge.save("tutorial_knowledge")

# Print a summary of what was learned
print(knowledge.summary())
```

### Executing Learned Actions

```python
from hyperboliclearner import KnowledgeExecutor

# Load previously acquired knowledge
executor = KnowledgeExecutor("tutorial_knowledge")

# Execute the learned sequence of actions
result = executor.execute()

# Print the results
print(f"Execution completed with status: {result.status}")
```

### Content Analysis Example

```python
from hyperboliclearner import ContentAnalyzer

# Initialize the analyzer
analyzer = ContentAnalyzer()

# Analyze a YouTube video for content quality
quality_score = analyzer.evaluate_video_quality("https://www.youtube.com/watch?v=example_video")

print(f"Video quality score: {quality_score}/10")
print(f"Key insights: {analyzer.extract_key_insights()}")
```

## Command Line Interface

HyperbolicLearner also provides a command-line interface for quick operations:

```bash
# Download and learn from a YouTube video
python -m hyperboliclearner learn https://www.youtube.com/watch?v=example_tutorial

# Execute previously learned knowledge
python -m hyperboliclearner execute tutorial_knowledge

# Analyze video quality
python -m hyperboliclearner analyze https://www.youtube.com/watch?v=example_video
```

## Configuration Options

HyperbolicLearner can be configured through the `config.json` file or environment variables:

```json
{
  "download_quality": "high",
  "acceleration_factor": 10,
  "use_gpu": true,
  "knowledge_db_path": "./knowledge_db",
  "log_level": "info"
}
```

Or using environment variables:

```bash
export HYPERBOLIC_DOWNLOAD_QUALITY=high
export HYPERBOLIC_ACCELERATION_FACTOR=10
export HYPERBOLIC_USE_GPU=true
```

## Troubleshooting

### Common Issues

#### Video Download Fails

```
Error: Unable to download video from YouTube
```

**Solution**: Check your internet connection and verify the video URL is correct and the video is publicly accessible.

#### GPU Acceleration Not Working

```
Warning: GPU acceleration unavailable, falling back to CPU
```

**Solution**: Ensure you have a CUDA-compatible GPU and have installed the GPU version of dependencies. Verify with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

#### Knowledge Base Corruption

```
Error: Knowledge base file is corrupted or incompatible
```

**Solution**: Try recovering from a backup or recreate the knowledge base:

```bash
python -m hyperboliclearner repair-kb tutorial_knowledge
```

#### Missing FFmpeg

```
Error: FFmpeg not found in system path
```

**Solution**: Install FFmpeg using your package manager:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Getting Help

If you encounter issues not covered here:

1. Check the logs in the `logs/` directory
2. Visit our [GitHub Issues](https://github.com/yourusername/HyperbolicLearner/issues)
3. Run the diagnostic tool: `python -m hyperboliclearner diagnose`

## Advanced Features

* **Batch Processing**: Process multiple videos in sequence
* **Knowledge Merging**: Combine knowledge from different sources
* **Content Filtering**: Automatically skip low-quality segments
* **Export/Import**: Share knowledge bases with other users

For more detailed information, refer to the full documentation in the `docs/` directory.

