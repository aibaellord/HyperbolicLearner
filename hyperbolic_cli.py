#!/usr/bin/env python3
"""
HyperbolicLearner CLI - Command Line Interface for the HyperbolicLearner system

This script provides an easy-to-use command line interface to interact with the
HyperbolicLearner system, allowing users to learn from videos, execute workflows,
query the knowledge base, and manage the system.

Usage:
  hyperbolic_cli.py learn [options] <url>
  hyperbolic_cli.py execute [options] <workflow_id>
  hyperbolic_cli.py query [options] <query>
  hyperbolic_cli.py list workflows|videos|knowledge
  hyperbolic_cli.py manage [options]

Examples:
  hyperbolic_cli.py learn https://www.youtube.com/watch?v=example --speed 5.0
  hyperbolic_cli.py execute workflow_12345 --verify
  hyperbolic_cli.py query "how to create a Docker container"
"""

import argparse
import sys
import os
import textwrap
from datetime import datetime
import logging

try:
    from colorama import init, Fore, Back, Style
    COLORAMA_AVAILABLE = True
    init()  # Initialize colorama
except ImportError:
    COLORAMA_AVAILABLE = False

# Path handling to ensure proper imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.config import ConfigManager
    from video_processor.downloader import YouTubeDownloader
    from video_processor.accelerator import VideoAccelerator
    from ui_automation.ui_analyzer import UIAnalyzer
    from knowledge_base.graph_db import KnowledgeGraph
    from ml_engine.content_analyzer import ContentAnalyzer
    from action_executor.executor import ActionExecutor
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperbolic_learner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HyperbolicCLI")

class ColoredFormatter:
    """Utility class to handle colored output formatting"""
    
    @staticmethod
    def success(message):
        """Format success messages"""
        if COLORAMA_AVAILABLE:
            return f"{Fore.GREEN}{message}{Style.RESET_ALL}"
        return f"SUCCESS: {message}"
    
    @staticmethod
    def error(message):
        """Format error messages"""
        if COLORAMA_AVAILABLE:
            return f"{Fore.RED}{message}{Style.RESET_ALL}"
        return f"ERROR: {message}"
    
    @staticmethod
    def warning(message):
        """Format warning messages"""
        if COLORAMA_AVAILABLE:
            return f"{Fore.YELLOW}{message}{Style.RESET_ALL}"
        return f"WARNING: {message}"
    
    @staticmethod
    def info(message):
        """Format info messages"""
        if COLORAMA_AVAILABLE:
            return f"{Fore.CYAN}{message}{Style.RESET_ALL}"
        return f"INFO: {message}"
    
    @staticmethod
    def highlight(message):
        """Format highlighted messages"""
        if COLORAMA_AVAILABLE:
            return f"{Fore.MAGENTA}{message}{Style.RESET_ALL}"
        return f">>> {message}"


class HyperbolicLearnerCLI:
    """Main CLI handler for HyperbolicLearner system"""
    
    def __init__(self):
        """Initialize the CLI interface"""
        self.parser = self._create_argument_parser()
        
        if not MODULES_AVAILABLE:
            logger.error(f"Failed to import required modules: {IMPORT_ERROR}")
            print(ColoredFormatter.error(
                "Failed to import required modules. Make sure you've installed all dependencies.\n"
                f"Error: {IMPORT_ERROR}\n"
                "Run: pip install -r requirements.txt"
            ))
        else:
            try:
                self.config = ConfigManager()
                self.config.load()
                self.downloader = YouTubeDownloader(self.config)
                self.accelerator = VideoAccelerator(self.config)
                self.ui_analyzer = UIAnalyzer(self.config)
                self.knowledge_graph = KnowledgeGraph(self.config)
                self.content_analyzer = ContentAnalyzer(self.config)
                self.action_executor = ActionExecutor(self.config)
            except Exception as e:
                logger.error(f"Failed to initialize components: {str(e)}")
                print(ColoredFormatter.error(
                    f"Failed to initialize components: {str(e)}"
                ))
                self.initialized = False
            else:
                self.initialized = True
    
    def _create_argument_parser(self):
        """Create and configure the argument parser"""
        parser = argparse.ArgumentParser(
            description=ColoredFormatter.highlight(
                "HyperbolicLearner CLI - Learn from videos and automate workflows"
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent("""
            Examples:
              hyperbolic_cli.py learn https://www.youtube.com/watch?v=example --speed 5.0
              hyperbolic_cli.py execute workflow_12345 --verify
              hyperbolic_cli.py query "how to create a Docker container"
            """)
        )
        
        # Create subparsers for different commands
        subparsers = parser.add_subparsers(dest='command', help='Command to execute')
        
        # Learn command
        learn_parser = subparsers.add_parser('learn', help='Learn from a video')
        learn_parser.add_argument('url', help='URL of the video to learn from')
        learn_parser.add_argument('--speed', type=float, default=1.0, 
                                help='Acceleration factor for learning (1.0-30.0)')
        learn_parser.add_argument('--quality', choices=['low', 'medium', 'high'], default='medium',
                                help='Video quality to download')
        learn_parser.add_argument('--output', '-o', help='Output file for generated workflow')
        learn_parser.add_argument('--force', '-f', action='store_true', 
                                help='Force re-download if video exists')
        learn_parser.add_argument('--segment', '-s', 
                                help='Time segment to process (format: start-end in seconds)')
        
        # Execute command
        execute_parser = subparsers.add_parser('execute', help='Execute a learned workflow')
        execute_parser.add_argument('workflow_id', help='ID of the workflow to execute')
        execute_parser.add_argument('--verify', '-v', action='store_true',
                                  help='Verify each action before executing')
        execute_parser.add_argument('--delay', '-d', type=float, default=0.5,
                                  help='Delay between actions in seconds')
        execute_parser.add_argument('--dry-run', action='store_true',
                                  help='Simulate execution without actually performing actions')
        
        # Query command
        query_parser = subparsers.add_parser('query', help='Query the knowledge base')
        query_parser.add_argument('query_text', help='Query text to search for')
        query_parser.add_argument('--format', '-f', choices=['text', 'json', 'yaml'], default='text',
                                help='Output format')
        query_parser.add_argument('--limit', '-l', type=int, default=10,
                                help='Maximum number of results to return')
        
        # List command
        list_parser = subparsers.add_parser('list', help='List available resources')
        list_parser.add_argument('resource_type', choices=['workflows', 'videos', 'knowledge'],
                               help='Type of resource to list')
        list_parser.add_argument('--filter', help='Filter results by keyword')
        list_parser.add_argument('--sort', choices=['date', 'name', 'size', 'relevance'],
                               default='date', help='Sort order')
        list_parser.add_argument('--limit', '-l', type=int, default=20,
                               help='Maximum number of items to display')
        
        # Manage command
        manage_parser = subparsers.add_parser('manage', help='Manage the system')
        manage_parser.add_argument('--clean-cache', action='store_true',
                                 help='Clean the video cache')
        manage_parser.add_argument('--update-models', action='store_true',
                                 help='Update machine learning models')
        manage_parser.add_argument('--backup', help='Backup knowledge base to specified file')
        manage_parser.add_argument('--restore', help='Restore knowledge base from specified file')
        manage_parser.add_argument('--stats', action='store_true',
                                 help='Show system statistics')
        
        # Version information
        parser.add_argument('--version', action='store_true', help='Show version information')
        
        # Verbose output
        parser.add_argument('--verbose', '-v', action='count', default=0,
                          help='Increase verbosity (can be used multiple times)')
        
        return parser
    
    def run(self):
        """Parse arguments and execute the requested command"""
        args = self.parser.parse_args()
        
        # Set logging level based on verbosity
        if args.verbose == 1:
            logging.getLogger().setLevel(logging.INFO)
        elif args.verbose >= 2:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Show version information if requested
        if args.version:
            self._show_version()
            return 0
        
        # Dispatch to the appropriate command handler
        if args.command == 'learn':
            return self._handle_learn(args)
        elif args.command == 'execute':
            return self._handle_execute(args)
        elif args.command == 'query':
            return self._handle_query(args)
        elif args.command == 'list':
            return self._handle_list(args)
        elif args.command == 'manage':
            return self._handle_manage(args)
        else:
            # No command specified, show help
            self.parser.print_help()
            return 1
    
    def _show_version(self):
        """Display version information"""
        version = "1.0.0"  # Set this to your actual version
        
        banner = """
╭─────────────────────────────────────────────────╮
│                                                 │
│       HyperbolicLearner - Version {version:<8}       │
│                                                 │
│     "Learn faster, automate better"             │
│                                                 │
╰─────────────────────────────────────────────────╯
        """
        print(ColoredFormatter.highlight(banner.format(version=version)))
        
        # Show component information if available
        if MODULES_AVAILABLE and self.initialized:
            capabilities = [
                ("Video Download", self.config.is_feature_enabled("video_download")),
                ("Video Acceleration", self.config.is_feature_enabled("video_acceleration")),
                ("UI Analysis", self.config.is_feature_enabled("ui_analysis")),
                ("Knowledge Graph", self.config.is_feature_enabled("knowledge_graph")),
                ("Action Execution", self.config.is_feature_enabled("action_execution")),
                ("Machine Learning", self.config.is_feature_enabled("machine_learning"))
            ]
            
            print(ColoredFormatter.info("System Capabilities:"))
            for capability, enabled in capabilities:
                status = "✓" if enabled else "✗"
                color_func = ColoredFormatter.success if enabled else ColoredFormatter.error
                print(f"  {color_func(status)} {capability}")
            
            print()
            print(ColoredFormatter.info("System Information:"))
            print(f"  Python Version: {sys.version.split()[0]}")
            print(f"  Platform: {sys.platform}")
            
            # Try to get GPU information if available
            if self.config.has_gpu_support():
                print(f"  GPU Support: {ColoredFormatter.success('Available')}")
                print(f"  GPU Devices: {', '.join(self.config.get_available_gpus())}")
            else:
                print(f"  GPU Support: {ColoredFormatter.warning('Not Available')}")
    
    def _handle_learn(self, args):
        """Handle the 'learn' command"""
        if not self._check_initialization():
            return 1
        
        try:
            print(ColoredFormatter.info(f"Learning from video: {args.url}"))
            print(f"Acceleration factor: {ColoredFormatter.highlight(str(args.speed))}x")
            
            # Parse time segment if provided
            start_time, end_time = None, None
            if args.segment:
                try:
                    parts = args.segment.split('-')
                    start_time = float(parts[0])
                    end_time = float(parts[1]) if len(parts) > 1 else None
                except ValueError:
                    print(ColoredFormatter.error(
                        "Invalid time segment format. Use 'start-end' in seconds."
                    ))
                    return 1
            
            # Download the video
            print(ColoredFormatter.info("Downloading video..."))
            video_path = self.downloader.download(
                args.url, 
                quality=args.quality,
                force=args.force
            )
            print(ColoredFormatter.success(f"Downloaded video to: {video_path}"))
            
            # Accelerate and process the video
            print(ColoredFormatter.info(f"Processing video at {args.speed}x speed..."))
            processed_path = self.accelerator.process(
                video_path, 
                speed_factor=args.speed,
                start_time=start_time,
                end_time=end_time
            )
            print(ColoredFormatter.success("Video processing complete"))
            
            # Analyze content and extract knowledge
            print(ColoredFormatter.info("Analyzing content and extracting knowledge..."))
            content_data = self.content_analyzer.analyze(processed_path)
            print(ColoredFormatter.success(f"Identified {len(content_data['scenes'])} key scenes"))
            print(ColoredFormatter.success(f"Extracted {len(content_data['concepts'])} concepts"))
            
            # Analyze UI interactions
            print(ColoredFormatter.info("Detecting UI interactions..."))
            ui_actions = self.ui_analyzer.analyze(processed_path)
            print(ColoredFormatter.success(f"Detected {len(ui_actions)} UI interactions"))
            
            # Store in knowledge graph
            print(ColoredFormatter.info("Storing in knowledge graph..."))
            workflow_id = self.knowledge_graph.add_workflow(
                source_url=args.url,
                content_data=content_data,
                ui_actions=ui_actions
            )
            
            # Success message
            print()
            print

