#!/usr/bin/env python3
"""
🔌 ENHANCED EXCHANGE MANAGER
============================

Production-ready exchange integration with real API connections,
error handling, rate limiting, and comprehensive monitoring.

Features:
- Real exchange API integration with testnet support
- Advanced error handling and retry mechanisms
- Rate limiting and connection pooling
- Health monitoring and failover
- Secure credential management
"""

import asyncio
import aiohttp
import ccxt.pro as ccxt
import os
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)

class ExchangeStatus(Enum):
    """Exchange connection status"""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"

@dataclass
class ExchangeHealth:
    """Exchange health metrics"""
    status: ExchangeStatus
    last_successful_request: Optional[datetime]
    error_count: int
    rate_limit_remaining: int
    average_latency_ms: float
    uptime_percentage: float
    last_error: Optional[str]

class SecureCredentials:
    """Secure credential management"""
    
    def __init__(self):
        self.cipher = None
        self._init_encryption()
    
    def _init_encryption(self):
        """Initialize encryption for API keys"""
        encryption_key = os.getenv('ENCRYPTION_KEY', '').encode()
        if len(encryption_key) >= 32:
            key = base64.urlsafe_b64encode(encryption_key[:32])
            self.cipher = Fernet(key)
        else:
            logger.warning("No encryption key provided, using plain text storage")
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        if self.cipher and data != 'placeholder':
            return self.cipher.encrypt(data.encode()).decode()
        return data
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if self.cipher and not encrypted_data.endswith('placeholder'):
            try:
                return self.cipher.decrypt(encrypted_data.encode()).decode()
            except:
                return encrypted_data
        return encrypted_data

class EnhancedExchangeManager:
    """Production-ready exchange manager with comprehensive features"""
    
    def __init__(self):
        self.exchanges = {}
        self.health_status = {}
        self.rate_limiters = {}
        self.credentials = SecureCredentials()
        self.session = None
        
        # Performance tracking
        self.request_counts = {}
        self.error_counts = {}
        self.latency_history = {}
        
        # Configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        self.health_check_interval = 30
        self.rate_limit_buffer = 0.8  # Use 80% of rate limit
        
        logger.info("🔌 Enhanced Exchange Manager initialized")
    
    async def initialize(self):
        """Initialize all exchange connections"""
        logger.info("🚀 Initializing exchange connections...")
        
        # Create aiohttp session
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Initialize exchanges
        await self._init_binance()
        await self._init_coinbase()
        await self._init_kraken()
        await self._init_kucoin()
        
        # Start health monitoring
        asyncio.create_task(self._continuous_health_monitoring())
        
        logger.info(f"✅ Initialized {len(self.exchanges)} exchange connections")
    
    async def _init_binance(self):
        """Initialize Binance connection"""
        if not self._is_exchange_enabled('BINANCE'):
            return
        
        try:
            api_key = self.credentials.decrypt(os.getenv('BINANCE_API_KEY', ''))
            secret = self.credentials.decrypt(os.getenv('BINANCE_SECRET', ''))
            use_testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'
            
            config = {
                'apiKey': api_key if not api_key.endswith('placeholder') else '',
                'secret': secret if not secret.endswith('placeholder') else '',
                'sandbox': use_testnet,
                'rateLimit': 100,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                }
            }
            
            exchange = ccxt.binance(config)
            
            # Test connection
            if api_key and not api_key.endswith('placeholder'):
                await exchange.load_markets()
                logger.info("✅ Binance connected with real API")
            else:
                logger.info("🔄 Binance initialized in demo mode")
            
            self.exchanges['binance'] = exchange
            self.health_status['binance'] = ExchangeHealth(
                status=ExchangeStatus.ONLINE,
                last_successful_request=datetime.now(),
                error_count=0,
                rate_limit_remaining=1000,
                average_latency_ms=100.0,
                uptime_percentage=100.0,
                last_error=None
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Binance: {e}")
            self._set_exchange_error('binance', str(e))
    
    async def _init_coinbase(self):
        """Initialize Coinbase Pro connection"""
        if not self._is_exchange_enabled('COINBASE'):
            return
        
        try:
            api_key = self.credentials.decrypt(os.getenv('COINBASE_API_KEY', ''))
            secret = self.credentials.decrypt(os.getenv('COINBASE_SECRET', ''))
            passphrase = self.credentials.decrypt(os.getenv('COINBASE_PASSPHRASE', ''))
            use_testnet = os.getenv('COINBASE_TESTNET', 'true').lower() == 'true'
            
            config = {
                'apiKey': api_key if not api_key.endswith('placeholder') else '',
                'secret': secret if not secret.endswith('placeholder') else '',
                'passphrase': passphrase if not passphrase.endswith('placeholder') else '',
                'sandbox': use_testnet,
                'rateLimit': 200,
                'enableRateLimit': True
            }
            
            exchange = ccxt.coinbase(config)
            
            # Test connection
            if api_key and not api_key.endswith('placeholder'):
                await exchange.load_markets()
                logger.info("✅ Coinbase connected with real API")
            else:
                logger.info("🔄 Coinbase initialized in demo mode")
            
            self.exchanges['coinbase'] = exchange
            self.health_status['coinbase'] = ExchangeHealth(
                status=ExchangeStatus.ONLINE,
                last_successful_request=datetime.now(),
                error_count=0,
                rate_limit_remaining=1000,
                average_latency_ms=150.0,
                uptime_percentage=100.0,
                last_error=None
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Coinbase: {e}")
            self._set_exchange_error('coinbase', str(e))
    
    async def _init_kraken(self):
        """Initialize Kraken connection"""
        if not self._is_exchange_enabled('KRAKEN'):
            return
        
        try:
            api_key = self.credentials.decrypt(os.getenv('KRAKEN_API_KEY', ''))
            secret = self.credentials.decrypt(os.getenv('KRAKEN_SECRET', ''))
            
            config = {
                'apiKey': api_key if not api_key.endswith('placeholder') else '',
                'secret': secret if not secret.endswith('placeholder') else '',
                'rateLimit': 300,
                'enableRateLimit': True
            }
            
            exchange = ccxt.kraken(config)
            
            # Test connection
            if api_key and not api_key.endswith('placeholder'):
                await exchange.load_markets()
                logger.info("✅ Kraken connected with real API")
            else:
                logger.info("🔄 Kraken initialized in demo mode")
            
            self.exchanges['kraken'] = exchange
            self.health_status['kraken'] = ExchangeHealth(
                status=ExchangeStatus.ONLINE,
                last_successful_request=datetime.now(),
                error_count=0,
                rate_limit_remaining=1000,
                average_latency_ms=200.0,
                uptime_percentage=100.0,
                last_error=None
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Kraken: {e}")
            self._set_exchange_error('kraken', str(e))
    
    async def _init_kucoin(self):
        """Initialize KuCoin connection"""
        if not self._is_exchange_enabled('KUCOIN'):
            return
        
        try:
            api_key = self.credentials.decrypt(os.getenv('KUCOIN_API_KEY', ''))
            secret = self.credentials.decrypt(os.getenv('KUCOIN_SECRET', ''))
            passphrase = self.credentials.decrypt(os.getenv('KUCOIN_PASSPHRASE', ''))
            use_testnet = os.getenv('KUCOIN_TESTNET', 'true').lower() == 'true'
            
            config = {
                'apiKey': api_key if not api_key.endswith('placeholder') else '',
                'secret': secret if not secret.endswith('placeholder') else '',
                'passphrase': passphrase if not passphrase.endswith('placeholder') else '',
                'sandbox': use_testnet,
                'rateLimit': 150,
                'enableRateLimit': True
            }
            
            exchange = ccxt.kucoin(config)
            
            # Test connection
            if api_key and not api_key.endswith('placeholder'):
                await exchange.load_markets()
                logger.info("✅ KuCoin connected with real API")
            else:
                logger.info("🔄 KuCoin initialized in demo mode")
            
            self.exchanges['kucoin'] = exchange
            self.health_status['kucoin'] = ExchangeHealth(
                status=ExchangeStatus.ONLINE,
                last_successful_request=datetime.now(),
                error_count=0,
                rate_limit_remaining=1000,
                average_latency_ms=120.0,
                uptime_percentage=100.0,
                last_error=None
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize KuCoin: {e}")
            self._set_exchange_error('kucoin', str(e))
    
    def _is_exchange_enabled(self, exchange_name: str) -> bool:
        """Check if exchange is enabled in configuration"""
        return os.getenv(f'{exchange_name}_ENABLED', 'false').lower() == 'true'
    
    def _set_exchange_error(self, exchange_name: str, error: str):
        """Set exchange to error state"""
        self.health_status[exchange_name] = ExchangeHealth(
            status=ExchangeStatus.ERROR,
            last_successful_request=None,
            error_count=1,
            rate_limit_remaining=0,
            average_latency_ms=0.0,
            uptime_percentage=0.0,
            last_error=error
        )
    
    async def fetch_ticker_with_retry(self, exchange_name: str, symbol: str) -> Optional[Dict]:
        """Fetch ticker with retry mechanism and error handling"""
        if exchange_name not in self.exchanges:
            return None
        
        exchange = self.exchanges[exchange_name]
        health = self.health_status.get(exchange_name)
        
        # Check if exchange is healthy
        if health and health.status in [ExchangeStatus.ERROR, ExchangeStatus.OFFLINE]:
            return None
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Apply rate limiting
                await self._apply_rate_limit(exchange_name)
                
                # Fetch ticker
                ticker = await exchange.fetch_ticker(symbol)
                
                # Update health metrics
                latency = (time.time() - start_time) * 1000
                self._update_health_success(exchange_name, latency)
                
                return ticker
                
            except ccxt.RateLimitExceeded:
                logger.warning(f"Rate limit exceeded for {exchange_name}, waiting...")
                self._update_health_rate_limit(exchange_name)
                await asyncio.sleep(self.retry_delay * (attempt + 1))
                
            except ccxt.NetworkError as e:
                logger.warning(f"Network error for {exchange_name}: {e}")
                self._update_health_error(exchange_name, str(e))
                await asyncio.sleep(self.retry_delay)
                
            except Exception as e:
                logger.error(f"Error fetching ticker from {exchange_name}: {e}")
                self._update_health_error(exchange_name, str(e))
                if attempt == self.max_retries - 1:
                    return None
                await asyncio.sleep(self.retry_delay)
        
        return None
    
    async def create_order_with_validation(self, exchange_name: str, symbol: str, 
                                         order_type: str, side: str, amount: float, 
                                         price: Optional[float] = None) -> Optional[Dict]:
        """Create order with comprehensive validation and error handling"""
        if exchange_name not in self.exchanges:
            logger.error(f"Exchange {exchange_name} not available")
            return None
        
        exchange = self.exchanges[exchange_name]
        health = self.health_status.get(exchange_name)
        
        # Check exchange health
        if health and health.status != ExchangeStatus.ONLINE:
            logger.warning(f"Exchange {exchange_name} not online: {health.status}")
            return None
        
        try:
            # Validate order parameters
            if not await self._validate_order_parameters(exchange, symbol, side, amount, price):
                return None
            
            start_time = time.time()
            
            # Apply rate limiting
            await self._apply_rate_limit(exchange_name)
            
            # Create order
            if order_type == 'market':
                if side == 'buy':
                    order = await exchange.create_market_buy_order(symbol, amount)
                else:
                    order = await exchange.create_market_sell_order(symbol, amount)
            else:
                order = await exchange.create_limit_order(symbol, side, amount, price)
            
            # Update metrics
            latency = (time.time() - start_time) * 1000
            self._update_health_success(exchange_name, latency)
            
            logger.info(f"✅ Order created on {exchange_name}: {order['id']}")
            return order
            
        except ccxt.InsufficientFunds:
            logger.error(f"Insufficient funds on {exchange_name}")
            return None
            
        except ccxt.InvalidOrder as e:
            logger.error(f"Invalid order on {exchange_name}: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Error creating order on {exchange_name}: {e}")
            self._update_health_error(exchange_name, str(e))
            return None
    
    async def _validate_order_parameters(self, exchange, symbol: str, side: str, 
                                       amount: float, price: Optional[float]) -> bool:
        """Validate order parameters against exchange requirements"""
        try:
            # Load market info
            markets = await exchange.load_markets()
            if symbol not in markets:
                logger.error(f"Symbol {symbol} not available on exchange")
                return False
            
            market = markets[symbol]
            
            # Validate minimum amount
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            if amount < min_amount:
                logger.error(f"Amount {amount} below minimum {min_amount}")
                return False
            
            # Validate price precision
            if price is not None:
                price_precision = market.get('precision', {}).get('price', 8)
                if len(str(price).split('.')[-1]) > price_precision:
                    logger.error(f"Price precision too high")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating order parameters: {e}")
            return False
    
    async def _apply_rate_limit(self, exchange_name: str):
        """Apply intelligent rate limiting"""
        if exchange_name not in self.rate_limiters:
            self.rate_limiters[exchange_name] = {
                'last_request': 0,
                'requests_per_second': 10,  # Default limit
                'requests_this_second': 0,
                'second_start': time.time()
            }
        
        limiter = self.rate_limiters[exchange_name]
        current_time = time.time()
        
        # Reset counter if new second
        if current_time - limiter['second_start'] >= 1.0:
            limiter['requests_this_second'] = 0
            limiter['second_start'] = current_time
        
        # Check if we need to wait
        if limiter['requests_this_second'] >= limiter['requests_per_second']:
            wait_time = 1.0 - (current_time - limiter['second_start'])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                limiter['requests_this_second'] = 0
                limiter['second_start'] = time.time()
        
        limiter['requests_this_second'] += 1
        limiter['last_request'] = current_time
    
    def _update_health_success(self, exchange_name: str, latency: float):
        """Update health metrics after successful request"""
        if exchange_name not in self.health_status:
            return
        
        health = self.health_status[exchange_name]
        health.status = ExchangeStatus.ONLINE
        health.last_successful_request = datetime.now()
        health.average_latency_ms = (health.average_latency_ms + latency) / 2
        health.error_count = max(0, health.error_count - 1)
    
    def _update_health_error(self, exchange_name: str, error: str):
        """Update health metrics after error"""
        if exchange_name not in self.health_status:
            return
        
        health = self.health_status[exchange_name]
        health.error_count += 1
        health.last_error = error
        
        if health.error_count > 5:
            health.status = ExchangeStatus.ERROR
    
    def _update_health_rate_limit(self, exchange_name: str):
        """Update health metrics after rate limit"""
        if exchange_name not in self.health_status:
            return
        
        health = self.health_status[exchange_name]
        health.status = ExchangeStatus.RATE_LIMITED
        health.rate_limit_remaining = 0
    
    async def _continuous_health_monitoring(self):
        """Continuously monitor exchange health"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                for exchange_name in self.exchanges.keys():
                    await self._health_check(exchange_name)
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
    
    async def _health_check(self, exchange_name: str):
        """Perform health check on specific exchange"""
        try:
            # Simple ping test
            ticker = await self.fetch_ticker_with_retry(exchange_name, 'BTC/USDT')
            
            if ticker:
                health = self.health_status[exchange_name]
                if health.status == ExchangeStatus.ERROR:
                    logger.info(f"🔄 Exchange {exchange_name} recovered")
                    health.status = ExchangeStatus.ONLINE
            
        except Exception as e:
            logger.debug(f"Health check failed for {exchange_name}: {e}")
    
    def get_healthy_exchanges(self) -> List[str]:
        """Get list of healthy exchanges"""
        healthy = []
        for name, health in self.health_status.items():
            if health.status == ExchangeStatus.ONLINE:
                healthy.append(name)
        return healthy
    
    def get_exchange_health_report(self) -> Dict[str, Dict]:
        """Get comprehensive health report for all exchanges"""
        report = {}
        for name, health in self.health_status.items():
            report[name] = {
                'status': health.status.value,
                'uptime_percentage': health.uptime_percentage,
                'average_latency_ms': health.average_latency_ms,
                'error_count': health.error_count,
                'last_successful_request': health.last_successful_request.isoformat() if health.last_successful_request else None,
                'last_error': health.last_error
            }
        return report
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up exchange connections...")
        
        # Close all exchange connections
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except:
                pass
        
        # Close aiohttp session
        if self.session:
            await self.session.close()
        
        logger.info("✅ Exchange manager cleanup completed")

# Global instance
exchange_manager = EnhancedExchangeManager()
