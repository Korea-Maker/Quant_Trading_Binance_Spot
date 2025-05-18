from dotenv import load_dotenv
import os

load_dotenv()

# APP Setting
APP_MODE = os.getenv('APP_MODE', 'development')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# API_KEY Setting
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Data DIR Setting
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
BACKTEST_DATA_DIR = os.path.join(DATA_DIR, 'backtest')

# Logger Setting
LOG_DIR = os.path.join(ROOT_DIR, 'logs')

# Trading Setting
DEFAULT_SYMBOL = os.getenv('DEFAULT_SYMBOL', 'BTCUSDT')
DEFAULT_INTERVALS = ['1m', '5m', '15m', '1h', '4h', '1d']

# 리스크 관리 설정
MAX_POSITION_SIZE_PCT = 0.1  # 총 자산의 10%
STOP_LOSS_PCT = 0.02  # 진입가 기준 2% 손실
TAKE_PROFIT_PCT = 0.05  # 진입가 기준 5% 이익

# 경로가 없으면 생성
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, BACKTEST_DATA_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)