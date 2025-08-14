# --- AI-Powered Creative Suggestions API ---
import random

# --- Blueprint Setup (must come before any @api.route usage) ---
from flask import Blueprint, request, jsonify
api = Blueprint('api', __name__)

# --- API: AI-Powered Creative Suggestions ---
@api.route('/api/meta/creative-suggestions', methods=['GET'])
def api_meta_creative_suggestions():
    """Return AI-powered creative scenario and goal suggestions."""
    import random
    creative_scenarios = [
        "Hyperrealistic time-loop scenario with evolving avatar memories",
        "Scenario blending ASMR, VR, and deepfake for multi-sensory immersion",
        "Interactive group scenario with real-time audience co-creation",
        "AI-generated taboo scenario with explainable compliance",
        "Scenario with dynamic emotional arcs and adaptive chemistry"
    ]
    creative_goals = [
        {"goal": "Invent a new micro-detail realism feature", "domain": "realism", "priority": 9},
        {"goal": "Blend two unrelated genres for a unique scenario", "domain": "creative", "priority": 8},
        {"goal": "Maximize user engagement with adaptive feedback loops", "domain": "engagement", "priority": 10},
        {"goal": "Integrate external AI art or music models", "domain": "external", "priority": 7},
        {"goal": "Create a scenario that evolves based on user dreams", "domain": "dreams", "priority": 10}
    ]
    return jsonify({"scenarios": random.sample(creative_scenarios, 3), "goals": random.sample(creative_goals, 3)})


# --- Main Dashboard UI Route ---

# --- Flask App Setup (must come before any @app.route usage) ---
import os
from flask import Flask, render_template
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

@app.route('/', methods=['GET'])
def main_dashboard():
    """Main dashboard with clickable buttons for all major features."""
    return render_template('main_dashboard.html')

# --- Creative Suggestions UI Route ---
@app.route('/creative-ai', methods=['GET'])
def creative_ai_page():
    """Render a page for AI-powered creative scenario and goal suggestions."""
    return render_template('creative_ai.html')
# --- MetaOrchestrator Goal Editing API ---
@api.route('/api/meta/goals', methods=['POST'])
def api_meta_add_goal():
    """Add a new goal to the MetaOrchestratorAgent."""
    if hasattr(orchestrator, 'meta_orchestrator') and orchestrator.meta_orchestrator:
        data = request.json or {}
        goal = data.get('goal')
        domain = data.get('domain', 'core')
        priority = int(data.get('priority', 5))
        orchestrator.meta_orchestrator.set_goal(goal, priority, domain)
        return jsonify({"status": "Goal added"})
    return jsonify({"error": "MetaOrchestrator not active"}), 400

@api.route('/api/meta/goals', methods=['DELETE'])
def api_meta_delete_goal():
    """Delete a goal by index from the MetaOrchestratorAgent."""
    if hasattr(orchestrator, 'meta_orchestrator') and orchestrator.meta_orchestrator:
        idx = int(request.args.get('idx', -1))
        if 0 <= idx < len(orchestrator.meta_orchestrator.goals):
            orchestrator.meta_orchestrator.goals.pop(idx)
            return jsonify({"status": "Goal deleted"})
        return jsonify({"error": "Invalid index"}), 400
    return jsonify({"error": "MetaOrchestrator not active"}), 400
# --- Manual Evolution Trigger & Goals API ---
@api.route('/api/meta/evolve', methods=['POST'])
def api_meta_evolve():
    """Manually trigger a MetaOrchestratorAgent evolution cycle."""
    if hasattr(orchestrator, 'meta_orchestrator') and orchestrator.meta_orchestrator:
        orchestrator.meta_orchestrator.orchestrate()
        return jsonify({"status": "Evolution cycle triggered"})
    return jsonify({"error": "MetaOrchestrator not active"}), 400

@api.route('/api/meta/goals', methods=['GET'])
def api_meta_goals():
    """Expose current goals of the MetaOrchestratorAgent."""
    if hasattr(orchestrator, 'meta_orchestrator') and orchestrator.meta_orchestrator:
        return jsonify({"goals": orchestrator.meta_orchestrator.goals})
    return jsonify({"error": "MetaOrchestrator not active"}), 400
# --- MetaOrchestrator Dashboard UI Route ---
@app.route('/meta-orchestrator', methods=['GET'])
def meta_orchestrator_dashboard():
    """Render the MetaOrchestrator Dashboard UI."""
    return render_template('meta_orchestrator_dashboard.html')
# --- MetaOrchestratorAgent Analytics & Activation API ---
@api.route('/api/meta/analytics', methods=['GET'])
def api_meta_analytics():
    """Expose real-time analytics from the MetaOrchestratorAgent."""
    return jsonify(orchestrator.get_meta_analytics())

@api.route('/api/meta/activate', methods=['POST'])
def api_meta_activate():
    """Activate the MetaOrchestratorAgent for autonomous evolution."""
    interval = int(request.json.get('interval', 1800))
    orchestrator.activate_meta_orchestrator(interval=interval)
    return jsonify({"status": "MetaOrchestrator activated", "interval": interval})
# --- Orchestrated Scenario UI Route ---
@app.route('/orchestrated-scenario', methods=['GET'])
def orchestrated_scenario_page():
    """Render the Orchestrated Scenario Generator UI."""
    return render_template('orchestrated_scenario.html')
# --- Orchestrator and Agent Initialization ---
from agents.chemistry_emotion_agent import ChemistryEmotionAgent
from agents.audio_sync_agent import AudioSyncAgent
from agents.web_crawler_agent import WebCrawlerAgent
from agents.ethics_agent import EthicsAgent
from agents.hyperspeed_agent import HyperspeedAgent
from agents.scenario_learning_agent import ScenarioLearningAgent

scenario_agent = ScenarioLearningAgent()
chemistry_agent = ChemistryEmotionAgent()
audio_sync_agent = AudioSyncAgent()
web_crawler_agent = WebCrawlerAgent()
ethics_agent = EthicsAgent()
hyperspeed_agent = HyperspeedAgent()

from agents.realism_orchestrator_agent import RealismOrchestratorAgent
realism_orchestrator_agent = RealismOrchestratorAgent(
    agents=[scenario_agent, chemistry_agent, audio_sync_agent],
    crawler_agent=web_crawler_agent,
    hyperspeed_agent=hyperspeed_agent
)
scenario_agent.integrate_with_orchestrator(realism_orchestrator_agent)
# --- Orchestrated Realism API ---
@api.route('/api/scenario/orchestrated', methods=['POST'])
def api_generate_orchestrated_scenario():
    """
    Generate a scenario using the RealismOrchestratorAgent for full automation and micro-detail realism.
    Expects JSON: {"content_sources": ["url1", ...], "user_profile": {...}, "toggles": {...}}
    """
    data = request.json or {}
    content_sources = data.get('content_sources', [])
    user_profile = data.get('user_profile', {})
    toggles = data.get('toggles', {})
    scenario = scenario_agent.orchestrated_scenario(content_sources, user_profile, toggles)
    return jsonify({"scenario": scenario})
# --- Scenario Feedback & Analytics API ---
@api.route('/api/scenario/feedback', methods=['POST'])
def api_submit_scenario_feedback():
    data = request.json or {}
    scenario_id = data.get('scenario_id')
    feedback = data.get('feedback', {})
    if not scenario_id:
        return jsonify({'error': 'Missing scenario_id'}), 400
    orchestrator.scenario_agent.store_user_feedback(scenario_id, feedback)
    return jsonify({'status': 'success'})

@api.route('/api/scenario/analytics', methods=['GET'])
def api_get_scenario_analytics():
    scenario_id = request.args.get('scenario_id')
    if not scenario_id:
        return jsonify({'error': 'Missing scenario_id'}), 400
    analytics = orchestrator.scenario_agent.get_engagement_analytics(scenario_id)
    feedback = orchestrator.scenario_agent.get_feedback_for_scenario(scenario_id)
    return jsonify({'analytics': analytics, 'feedback': feedback})
# --- Advanced Scenario Generation API ---
from core.orchestrator import Orchestrator
from agents.scenario_learning_agent import ScenarioLearningAgent


# --- Orchestrator and Agent Initialization ---
from agents.chemistry_emotion_agent import ChemistryEmotionAgent
from agents.audio_sync_agent import AudioSyncAgent
from agents.web_crawler_agent import WebCrawlerAgent
from agents.ethics_agent import EthicsAgent
from agents.hyperspeed_agent import HyperspeedAgent
from agents.scenario_learning_agent import ScenarioLearningAgent

scenario_agent = ScenarioLearningAgent()
chemistry_agent = ChemistryEmotionAgent()
audio_sync_agent = AudioSyncAgent()
web_crawler_agent = WebCrawlerAgent()
ethics_agent = EthicsAgent()
hyperspeed_agent = HyperspeedAgent()

def preprocessor(data, metadata, model=None):
    return data
def synthesizer(data, metadata, model=None):
    return data
def validation(data, metadata, model=None):
    return {'valid': True}

agents = {
    'scenario_learning': scenario_agent,
    'chemistry_emotion': chemistry_agent,
    'audio_sync': audio_sync_agent,
    'web_crawler': web_crawler_agent,
    'ethics': ethics_agent,
    'hyperspeed': hyperspeed_agent,
    'preprocessor': preprocessor,
    'synthesizer': synthesizer,
    'validation': validation
}
orchestrator = Orchestrator(agents)

@api.route('/api/scenario/advanced', methods=['POST'])
def api_generate_advanced_scenario():
    """
    Generate a scenario with any combination of advanced features.
    Expects JSON: {"user_prefs": {...}, "features": ["ai_world_building", ...], "feature_kwargs": {"ai_world_building": {"world_type": "sci-fi"}, ...}}
    """
    if orchestrator is None:
        return jsonify({"error": "Orchestrator not initialized"}), 500
    data = request.json or {}
    user_prefs = data.get('user_prefs', {})
    features = data.get('features', [])
    feature_kwargs = data.get('feature_kwargs', {})
    scenario = orchestrator.generate_advanced_scenario(user_prefs, features, feature_kwargs)
    return jsonify({"scenario": scenario})
# --- API Endpoints for Real-Time Scene Direction, Batch Processing, and Randomization ---
from flask import Blueprint, request, jsonify

api = Blueprint('api', __name__)

# Endpoint: Real-time scene direction (adjust chemistry, emotion, positions, activities)
@api.route('/api/scene/direction', methods=['POST'])
def api_scene_direction():
    data = request.json
    # Placeholder: Integrate with orchestrator and agents for live adjustment
    # Example: {"job_id": ..., "chemistry": ..., "emotion": ..., "positions": ..., "activities": ...}
    # Return updated preview or status
    return jsonify({"status": "success", "message": "Scene direction updated", "data": data})

# Endpoint: Batch job submission
@api.route('/api/batch/submit', methods=['POST'])
def api_batch_submit():
    jobs = request.json.get('jobs', [])
    # Placeholder: Submit jobs to orchestrator for batch processing
    # Return job IDs and initial status
    return jsonify({"status": "submitted", "job_ids": [str(uuid.uuid4()) for _ in jobs]})

# Endpoint: Batch job status
@api.route('/api/batch/status', methods=['GET'])
def api_batch_status():
    job_ids = request.args.getlist('job_id')
    # Placeholder: Query orchestrator or job manager for status
    return jsonify({"statuses": {jid: "processing" for jid in job_ids}})

# Endpoint: Surprise Me / Randomization
@api.route('/api/scenario/surprise', methods=['GET'])
def api_scenario_surprise():
    # Placeholder: Use scenario agent to generate a random, unique scenario
    scenario = {"scenario": "randomized", "details": "A unique, surprising scenario was generated."}
    return jsonify(scenario)

## Register blueprint with Flask app (after app is defined)
# (Moved to after app = Flask(...) definition)
#!/usr/bin/env python3
"""
HyperbolicLearner Web Interface Module

This module provides a comprehensive web-based UI for interacting with all capabilities
of the HyperbolicLearner system, including video learning, knowledge visualization,
agent monitoring, and system configuration.

Features:
- Interactive dashboards with real-time system monitoring and analytics
- Advanced 3D knowledge visualization with exploration tools
- AI-powered recommendation system for video content and learning paths
- Autonomous agent management with communication style training
- One-click automation for complex workflows and task sequences
- Smart context-aware help system with visual tutorials
- User profiles with personalized learning analytics
- Team collaboration features for shared knowledge bases
- Progressive web app capabilities for mobile and offline use
- Integration with common productivity platforms and LLMs
- Voice command interface for hands-free operation
- Smart workflow suggestions based on past user behavior
- Adaptive UI that simplifies or expands based on user expertise
- Real-time collaborative learning with multi-user sessions
- Advanced knowledge export and integration capabilities
"""

import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, TypeVar, Generic, Protocol, Iterator, Generator, cast
from pathlib import Path
import uuid
import asyncio
import queue
import re
import shutil
import subprocess
import tempfile
import hashlib
import functools
import signal
import platform
import socket
import psutil
import pytz
import zipfile
import io
import csv
import pickle
import base64
import mimetypes
import secrets
import string
import dataclasses
import inspect
import traceback
import contextlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, lru_cache, partial
from itertools import chain, islice, groupby
from collections import defaultdict, Counter, deque, OrderedDict, namedtuple
from urllib.parse import urlparse, parse_qs, urlencode, quote, unquote

# Web framework and related imports
import flask
from flask import Flask, Blueprint, request, jsonify, render_template, redirect, url_for, session
from flask import flash, send_from_directory, send_file, Response, stream_with_context, g, current_app, make_response
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect, rooms, Namespace
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user, AnonymousUserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.local import LocalProxy
from werkzeug.exceptions import HTTPException, NotFound, Forbidden, BadRequest, Unauthorized, InternalServerError
from flask_wtf import FlaskForm, CSRFProtect
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField, TextAreaField, FloatField, IntegerField, HiddenField, RadioField, DateTimeField
from wtforms.validators import DataRequired, Email, EqualTo, ValidationError, URL, NumberRange, Length, Optional as OptionalValidator, Regexp, AnyOf, NoneOf, IPAddress, MacAddress, UUID, input_required
import flask_admin as admin
from flask_admin.contrib.sqla import ModelView
from flask_admin.form import rules, fields
from flask_admin.actions import action
from flask_admin.model.template import macro
from flask_admin.contrib.fileadmin import FileAdmin
from flask_cors import CORS
from flask_compress import Compress
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from flask_babel import Babel, _, lazy_gettext as _l, format_datetime, format_date, format_time, format_currency, format_number, format_percent
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, create_refresh_token, get_jwt_identity, verify_jwt_in_request
from flask_mail import Mail, Message
from flask_talisman import Talisman
from flask_seasurf import SeaSurf
from flask_sslify import SSLify
from flask_session import Session
from flask_debugtoolbar import DebugToolbarExtension
from flask_sitemap import Sitemap
from flask_uploads import UploadSet, configure_uploads, IMAGES, DOCUMENTS, DATA, ALL

# Database
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON, JSONB, ARRAY, HSTORE, UUID as PGUUID, INET, CIDR, MACADDR, ENUM, BYTEA, TSRANGE, INTERVAL
from sqlalchemy.ext.mutable import MutableDict, MutableList
import sqlalchemy as sa
from sqlalchemy import func, desc, asc, and_, or_, not_, text, case, cast, literal, between, distinct, type_coerce, tuple_, bindparam, union, union_all, intersect, except_, exists, inspect, event, select, insert, update, delete, alias, join, outerjoin, lateral, null, true, false
from sqlalchemy.sql import expression
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from sqlalchemy.orm import joinedload, selectinload, relationship, backref, deferred, column_property, aliased, lazyload, immediateload, noload, load_only, validates, foreign, remote, object_session, make_transient, make_transient_to_detached, scoped_session, sessionmaker, defer, undefer
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.sql.expression import cast, extract
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.pool import QueuePool, NullPool, StaticPool

# For knowledge graph visualization
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats, optimize, interpolate, signal, spatial, linalg
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis, NMF, KernelPCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, PowerTransformer, QuantileTransformer, OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, Birch, MeanShift, AffinityPropagation
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import umap
import hdbscan
from pydantic import BaseModel, Field, validator, root_validator

# For WebRTC and real-time communication
try:
    import aiortc
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
    from aiortc.contrib.media import MediaRelay, MediaBlackhole, MediaPlayer, MediaRecorder
    from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack, AudioStreamTrack
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# For ML model serving
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# For background tasks
from celery import Celery, Task, chain, group, chord, signature
from celery.schedules import crontab
from celery.signals import task_prerun, task_postrun, task_failure, task_success, worker_ready
from celery.utils.log import get_task_logger
from celery.result import AsyncResult
from celery.exceptions import TaskError, Retry, Ignore

# For API documentation
from flask_restx import Api, Resource, fields, Namespace, reqparse, abort, inputs, marshal_with, marshal
from flask_swagger_ui import get_swaggerui_blueprint

# For full-text search
try:
    import whoosh
    from whoosh.index import create_in, open_dir
    from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED, DATETIME
    from whoosh.qparser import QueryParser, MultifieldParser
    from whoosh.query import Term, And, Or, Not, Every
    from whoosh.sorting import FieldFacet, MultiFacet, FunctionFacet
    WHOOSH_AVAILABLE = True
except ImportError:
    WHOOSH_AVAILABLE = False

# For advanced caching
try:
    import diskcache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False

# Import HyperbolicLearner components
# Absolute imports for Flask compatibility
from core.config import Config
from video_processor.youtube_learner import YouTubeLearner
from video_processor.accelerator import VideoAccelerator
from ml_engine.content_analyzer import ContentAnalyzer
from knowledge_base.graph_db import KnowledgeGraph
from ui_automation.ui_analyzer import UIAnalyzer
from action_executor.executor import ActionExecutor
from agents.realtime_agent import RealtimeAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hyperbolic_ui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'hyperbolic_secret_key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI', 'sqlite:///hyperbolic_ui.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max upload
app.config['SESSION_TYPE'] = 'filesystem'
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['CACHE_TYPE'] = 'filesystem'
app.config['CACHE_DIR'] = os.path.join(os.path.dirname(__file__), 'cache')
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_DEFAULT_TIMEZONE'] = 'UTC'
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'hyperbolic_jwt_secret')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'True').lower() in ('true', 'yes', '1')
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', 'hyperbolic@example.com')
app.config['SECURITY_PASSWORD_SALT'] = os.environ.get('SECURITY_PASSWORD_SALT', 'hyperbolic_security_salt')
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=31)
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # 1 hour
app.config['CELERY_TIMEZONE'] = 'UTC'
app.config['APP_VERSION'] = '1.0.0'
app.config['APP_NAME'] = 'HyperbolicLearner'

# Apply middleware
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)
Compress(app)
# Enable HTTPS if not in development
if not app.debug:
    SSLify(app)

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent', message_queue=os.environ.get('SOCKETIO_MESSAGE_QUEUE'), logger=True, engineio_logger=True)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'
csrf = CSRFProtect(app)
CORS(app)
limiter = Limiter(app, key_func=get_remote_address, default_limits=["200 per minute", "10 per second"])
cache = Cache(app)
babel = Babel(app)
jwt = JWTManager(app)
mail = Mail(app)
Session(app)
sitemap = Sitemap(app)
talisman = Talisman(app, content_security_policy=None)

# Initialize upload sets
images = UploadSet('images', IMAGES)
documents = UploadSet('documents', DOCUMENTS)
data_files = UploadSet('data', DATA)
configure_uploads(app, (images, documents, data_files))

# Set up Celery for background tasks
celery = Celery(app.name, broker=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
               backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'))
celery.conf.update(app.config)
celery.conf.task_serializer = 'json'
celery.conf.result_serializer = 'json'
celery.conf.accept_content = ['json']
celery.conf.task_routes = {
    'tasks.*': {'queue': 'default'},
    'high_priority_tasks.*': {'queue': 'high_priority'},
    'low_priority_tasks.*': {'queue': 'low_priority'},
    'email.*': {'queue': 'email'},
    'video_processing.*': {'queue': 'video_processing'},
    'ml_model_serving.*': {'queue': 'ml_model_serving'},
    'webrtc.*': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.worker_concurrency = 4
celery.conf.broker_pool_limit = 10
celery.conf.result_backend_transport_options = {'visibility_timeout': 3600}
celery.conf.task_time_limit = 300
celery.conf.worker_max_tasks_per_child = 100
celery.conf.task_annotations = {'tasks.add': {'rate_limit': '10/s'}, 'tasks.mul': {'rate_limit': '10/s'}}
celery.conf.beat_schedule = {
    'every-30-minutes': {
        'task': 'tasks.my_scheduled_task',
        'schedule': crontab(minute=0, hour='*/1'),  # Every hour
        'args': ()
    },
    'every-5-minutes': {
        'task': 'tasks.another_scheduled_task',
        'schedule': crontab(minute='*/5'),  # Every 5 minutes
        'args': ()
    }
}
celery.conf.task_create_missing_queues = True
celery.conf.worker_prefetch_multiplier = 1
celery.conf.task_acks_late = True
celery.conf.worker_send_task_events = True
celery.conf.event_serializer = 'json'
celery.conf.task_serializer = 'json'
celery.conf.result_serializer = 'json'
celery.conf.accept_content = ['json']
celery.conf.broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
celery.conf.result_backend = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
celery.conf.timezone = 'UTC'
celery.conf.enable_utc = True
celery.conf.worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
celery.conf.task_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
celery.conf.worker_redirect_stdouts = False
celery.conf.worker_stdout_logfile = os.path.join(os.path.dirname(__file__), 'logs', 'celery_worker.log')
celery.conf.worker_stderr_logfile = os.path.join(os.path.dirname(__file__), 'logs', 'celery_worker_error.log')
celery.conf.beat_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
celery.conf.event_loop = 'asyncio'
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}
celery.conf.task_default_queue = 'default'
celery.conf.task_default_exchange = 'default'
celery.conf.task_default_routing_key = 'default'
celery.conf.task_routes = {
    'tasks.add': {'queue': 'default'},
    'tasks.mul': {'queue': 'default'},
    'high_priority_tasks.add': {'queue': 'high_priority'},
    'high_priority_tasks.mul': {'queue': 'high_priority'},
    'low_priority_tasks.add': {'queue': 'low_priority'},
    'low_priority_tasks.mul': {'queue': 'low_priority'},
    'email.send': {'queue': 'email'},
    'video_processing.start': {'queue': 'video_processing'},
    'ml_model_serving.predict': {'queue': 'ml_model_serving'},
    'webrtc.offer': {'queue': 'webrtc'},
    'webrtc.answer': {'queue': 'webrtc'},
    'webrtc.ice_candidate': {'queue': 'webrtc'},
    'default': {'queue': 'default'}
}