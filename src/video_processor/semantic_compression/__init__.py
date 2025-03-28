"""
Semantic Compression Module for HyperbolicLearner
================================================

This module provides advanced tools for content-aware video acceleration through
semantic importance analysis and multimodal fusion techniques. The system identifies
and preserves the most informative parts of video content while intelligently
accelerating or eliminating less important segments, resulting in hyperbolic learning efficiency.

Core Capabilities:
-----------------
- Multi-level content importance detection across visual, audio, and textual modalities
- Dynamic acceleration profiles based on semantic density and user learning preferences
- Preservation of critical educational moments while compressing redundant content
- Cross-modal alignment to maintain context across accelerated segments
- Adaptive processing based on content type (lectures, tutorials, demonstrations, etc.)

Key Components:
--------------
- Importance Models: Neural networks that analyze visual, audio, and transcript content
  to identify important segments using domain-specific heuristics
  
- Multimodal Fusion: Advanced attention mechanisms that combine importance scores
  across modalities with temporal context awareness and content-type sensitivity
  
- Semantic Compressor: Intelligent acceleration engine that applies variable speedup
  based on semantic importance while preserving content integrity and learning value

Technical Features:
-----------------
- GPU-optimized processing pipeline for real-time performance
- Configurable importance thresholds and acceleration factors
- Pre-trained models for common educational domains
- Streaming-compatible processing for handling large videos
- Detailed metrics and visualizations of compression decisions

Usage Examples:
--------------
Basic usage:
    >>> from video_processor.semantic_compression import create_compressor
    >>> 
    >>> compressor = create_compressor()
    >>> compressed_video = compressor.compress(
    ...     video_path="path/to/tutorial.mp4",
    ...     acceleration_factor=5.0
    ... )
    >>> compressed_video.save("accelerated_tutorial.mp4")

Advanced configuration:
    >>> from video_processor.semantic_compression import (
    ...     SemanticCompressor, CompressionConfig, ImportanceModelFactory,
    ...     ModalityWeights, AccelerationProfile
    ... )
    >>> 
    >>> # Create custom configuration
    >>> config = CompressionConfig(
    ...     min_acceleration=1.0,
    ...     max_acceleration=30.0,
    ...     importance_threshold=0.65,
    ...     preserve_transitions=True,
    ...     audio_quality='high'
    ... )
    >>> 
    >>> # Create domain-specific importance models
    >>> models = ImportanceModelFactory.create_models(
    ...     domain="programming_tutorials",
    ...     visual_model="attention_resnet50",
    ...     audio_model="wav2vec_classifier",
    ...     transcript_model="domain_bert"
    ... )
    >>> 
    >>> # Configure modality weights for programming content
    >>> weights = ModalityWeights(
    ...     visual=0.6,    # Higher weight for visual (code examples)
    ...     audio=0.3,     # Medium weight for audio explanations
    ...     transcript=0.7 # High weight for technical terms in transcript
    ... )
    >>> 
    >>> # Create a personalized acceleration profile
    >>> profile = AccelerationProfile.from_user_history(
    ...     user_id="user123",
    ...     content_type="programming_tutorial"
    ... )
    >>> 
    >>> # Create the compressor with all customizations
    >>> compressor = SemanticCompressor(
    ...     config=config,
    ...     importance_models=models,
    ...     modality_weights=weights,
    ...     acceleration_profile=profile,
    ...     use_gpu=True
    ... )
    >>> 
    >>> # Process video with progress tracking
    >>> compressed_video = compressor.compress(
    ...     video_path="path/to/python_course.mp4",
    ...     callback=lambda progress: print(f"Processing: {progress*100:.1f}%")
    ... )
    >>> 
    >>> # Generate and save compression report
    >>> report = compressed_video.generate_report()
    >>> report.save("compression_report.pdf")
    >>> 
    >>> # Save processed video
    >>> compressed_video.save("accelerated_python_course.mp4")

Integration with knowledge extraction:
    >>> from video_processor.semantic_compression import SemanticCompressor
    >>> from knowledge_base.extraction import KnowledgeExtractor
    >>> 
    >>> # Create pipeline components
    >>> compressor = SemanticCompressor()
    >>> extractor = KnowledgeExtractor()
    >>> 
    >>> # Process video and extract knowledge
    >>> compressed_video = compressor.compress("tutorial.mp4", acceleration_factor=8.0)
    >>> knowledge_graph = extractor.extract_from_compressed_video(
    ...     compressed_video,
    ...     include_importance_markers=True
    ... )
    >>> 
    >>> # The knowledge graph will be enriched with importance data from compression
"""

__version__ = "0.1.0"
__author__ = "HyperbolicLearner Team"
__copyright__ = "Copyright 2023 HyperbolicLearner"
__license__ = "MIT"

# Core component imports - make high-level classes and functions directly accessible
from .importance_models import (
    # Base classes
    VisualImportanceModel,
    AudioImportanceModel,
    TranscriptImportanceModel,
    ImportanceModelBase,
    ImportanceModelFactory,
    
    # Visual importance models
    CNNVisualImportanceModel,
    TransformerVisualImportanceModel,
    AttentionMapGenerator,
    FeatureExtractor,
    ObjectDetectionImportance,
    MotionImportanceDetector,
    
    # Audio importance models
    SpectrogramImportanceModel,
    SpeechRateAnalyzer,
    EmphasisDetectionModel,
    AudioSegmentClassifier,
    KeywordEmphasisDetector,
    
    # Transcript importance models
    KeyphraseExtractionModel,
    TechnicalTermClassifier,
    DiscourseMarkerDetector,
    TopicSegmentationModel,
    SentimentImportanceModel,
    
    # Utilities
    ImportanceVisualizer,
    ModelRegistry,
    ImportanceScoreNormalizer
)

from .multimodal_fusion import (
    # Main fusion components
    MultimodalFusion,
    AttentionFusionStrategy,
    ContextAwareFusion,
    TemporalAlignmentProcessor,
    ModalityWeights,
    
    # Fusion strategies
    WeightedAverageFusion,
    DynamicAttentionFusion,
    CrossModalAttention,
    HierarchicalFusion,
    AdaptiveFusion,
    
    # Context modeling
    TemporalContextModel,
    ContentTypeClassifier,
    ContextWindow,
    ContextualImportanceAdjuster,
    
    # Alignment components
    ModalityAligner,
    SequenceAlignment,
    TemporalEmbedding,
    
    # Utilities
    FusionVisualizer,
    ModalityWeightOptimizer,
    FusionStrategySelector
)

from .semantic_compressor import (
    # Main compressor classes
    SemanticCompressor,
    CompressionConfig,
    AccelerationProfile,
    CompressedVideo,
    CompressionReport,
    
    # Processing components
    FrameSampler,
    ContentAwareAccelerator,
    AudioTimeStretcher,
    SilenceRemover,
    TransitionPreserver,
    
    # Frame processing
    FrameImportanceMap,
    KeyFrameDetector,
    FrameGrouper,
    
    # Acceleration strategies
    AdaptiveAcceleration,
    SegmentBasedAcceleration,
    GradualAccelerationController,
    AccelerationCurve,
    
    # Domain-specific handlers
    TutorialCompressor,
    LectureCompressor,
    DemonstrationCompressor,
    
    # Utilities
    AccelerationVisualizer,
    SegmentationAnalyzer,
    PerformanceOptimizer,
    StreamingCompressor
)

# Define public API
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__copyright__",
    "__license__",
    
    # Importance Models
    "VisualImportanceModel",
    "AudioImportanceModel",
    "TranscriptImportanceModel",
    "ImportanceModelBase",
    "ImportanceModelFactory",
    "CNNVisualImportanceModel",
    "TransformerVisualImportanceModel",
    "AttentionMapGenerator",
    "FeatureExtractor",
    "ObjectDetectionImportance",
    "MotionImportanceDetector",
    "SpectrogramImportanceModel",
    "SpeechRateAnalyzer",
    "EmphasisDetectionModel",
    "AudioSegmentClassifier",
    "KeywordEmphasisDetector",
    "KeyphraseExtractionModel",
    "TechnicalTermClassifier",
    "DiscourseMarkerDetector",
    "TopicSegmentationModel",
    "SentimentImportanceModel",
    "ImportanceVisualizer",
    "ModelRegistry",
    "ImportanceScoreNormalizer",
    
    # Multimodal Fusion
    "MultimodalFusion",
    "AttentionFusionStrategy",
    "ContextAwareFusion",
    "TemporalAlignmentProcessor",
    "ModalityWeights",
    "WeightedAverageFusion",
    "DynamicAttentionFusion",
    "CrossModalAttention",
    "HierarchicalFusion",
    "AdaptiveFusion",
    "TemporalContextModel",
    "ContentTypeClassifier",
    "ContextWindow",
    "ContextualImportanceAdjuster",
    "ModalityAligner",
    "SequenceAlignment",
    "TemporalEmbedding",
    "FusionVisualizer",
    "ModalityWeightOptimizer",
    "FusionStrategySelector",
    
    # Semantic Compressor
    "SemanticCompressor",
    "CompressionConfig",
    "AccelerationProfile",
    "CompressedVideo",
    "CompressionReport",
    "FrameSampler",
    "ContentAwareAccelerator",
    "AudioTimeStretcher",
    "SilenceRemover",
    "TransitionPreserver",
    "FrameImportanceMap",
    "KeyFrameDetector",
    "FrameGrouper",
    "AdaptiveAcceleration",
    "SegmentBasedAcceleration",
    "GradualAccelerationController",
    "AccelerationCurve",
    "TutorialCompressor",
    "LectureCompressor",
    "DemonstrationCompressor",
    "AccelerationVisualizer",
    "SegmentationAnalyzer",
    "PerformanceOptimizer",
    "StreamingCompressor",
]

# Convenience functions for common use cases
def create_compressor(config=None, use_gpu=True, models_dir=None, domain=None, optimization_level="balanced"):
    """
    Create a fully configured SemanticCompressor with optimized settings.
    
    Args:
        config (dict, optional): Configuration parameters to override defaults.
        use_gpu (bool): Whether to use GPU acceleration if available.
        models_dir (str, optional): Directory containing pre-trained models.
        domain (str, optional): Specify content domain for specialized processing:
            - "programming" - Code-focused tutorials and demonstrations
            - "science" - Scientific lectures and explanations
            - "business" - Business presentations and courses
            - "arts" - Creative software tutorials
            - "general" - General educational content
        optimization_level (str): Processing optimization strategy:
            - "speed" - Optimize for processing speed over accuracy
            - "quality" - Optimize for highest quality compression
            - "balanced" - Balance speed and quality (default)
            - "memory" - Optimize for lower memory usage
        
    Returns:
        SemanticCompressor: A ready-to-use semantic compressor instance.
    """
    from .semantic_compressor import SemanticCompressor, CompressionConfig
    from .importance_models import ImportanceModelFactory
    
    # Create base configuration
    default_config = CompressionConfig()
    if config:
        for key, value in config.items():
            setattr(default_config, key, value)
    
    # Create domain-specific models if domain specified
    importance_models = None
    if domain:
        importance_models = ImportanceModelFactory.create_models(
            domain=domain,
            models_dir=models_dir
        )
    
    # Configure performance optimization
    performance_settings = {
        "speed": {"batch_size": 16, "precision": "mixed", "cache_level": "minimal"},
        "quality": {"batch_size": 4, "precision": "full", "cache_level": "extensive"},
        "balanced": {"batch_size": 8, "precision": "mixed", "cache_level": "standard"},
        "memory": {"batch_size": 2, "precision": "mixed", "cache_level": "minimal"}
    }
    
    optimization = performance_settings.get(optimization_level, performance_settings["balanced"])
    
    # Create and return the configured compressor
    return SemanticCompressor(
        config=default_config,
        importance_models=importance_models,
        use_gpu=use_gpu,
        models_dir=models_dir,
        **optimization
    )

def batch_process_videos(video_paths, output_dir, acceleration_factor=5.0, use_gpu=True, 
                         num_workers=4, domain=None, callback=None):
    """
    Process multiple videos in batch mode with the semantic compressor.
    
    Args:
        video_paths (list): List of paths to videos for processing
        output_dir (str): Directory to save compressed videos
        acceleration_factor (float): Target acceleration factor
        use_gpu (bool): Whether to use GPU acceleration
        num_workers (int): Number of parallel workers (if 0, sequential processing)
        domain (str, optional): Specify content domain for specialized processing
        callback (callable, optional): Progress callback function
        
    Returns:
        list: List of CompressedVideo objects
    """
    import os
    import concurrent.futures
    from .semantic_compressor import SemanticCompressor
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create compressor instance
    compressor = create_compressor(use_gpu=use_gpu, domain=domain)
    
    # Define processing function
    def process_video(video_path):
        try:
            base_name = os.path.basename(video_path)
            output_name = f"compressed_{base_name}"
            output_path = os.path.join(output_dir, output_name)
            
            # Compress video
            compressed = compressor.compress(
                video_path=video_path,
                acceleration_factor=acceleration_factor
            )
            
            # Save compressed video
            compressed.save(output_path)
            
            return compressed
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return None
    
    # Process videos
    results = []
    if num_workers <= 0:
        # Sequential processing
        total = len(video_paths)
        for i, path in enumerate(video_paths):
            result = process_video(path)
            results.append(result)
            if callback:
                callback((i + 1) / total)
    else:
        # Parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_path = {executor.submit

