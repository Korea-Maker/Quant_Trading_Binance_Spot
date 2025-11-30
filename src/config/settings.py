# src/config/settings.py

from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Tuple, List

# .env 파일 로드
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
env_path = Path(ROOT_DIR) / '.env'

# .env 파일이 없으면 경고
if not env_path.exists():
    print("⚠️  경고: .env 파일이 없습니다. .env.example을 참고하여 .env 파일을 생성하세요.")

load_dotenv(dotenv_path=env_path)

# APP Setting
APP_MODE = os.getenv('APP_MODE', 'development')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
# TEST_MODE: 'True' 또는 'False' 문자열을 boolean으로 변환
TEST_MODE_STR = os.getenv('TEST_MODE', 'True')
TEST_MODE = TEST_MODE_STR.lower() in ('true', '1', 'yes', 'on')

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
# TRADING_TYPE: 'spot' 또는 'futures' (기본값: 'spot')
TRADING_TYPE = os.getenv('TRADING_TYPE', 'spot').lower()
if TRADING_TYPE not in ['spot', 'futures']:
    TRADING_TYPE = 'spot'  # 기본값으로 설정

# SSL 검증 설정 (Windows 환경에서 SSL 오류 발생 시 False로 설정)
# 주의: 프로덕션 환경에서는 True로 유지하는 것이 안전합니다
SSL_VERIFY_STR = os.getenv('SSL_VERIFY', 'True')
SSL_VERIFY = SSL_VERIFY_STR.lower() not in ('false', '0', 'no', 'off')

# 리스크 관리 설정
MAX_POSITION_SIZE_PCT = 0.1  # 총 자산의 10%
STOP_LOSS_PCT = 0.02  # 진입가 기준 2% 손실
TAKE_PROFIT_PCT = 0.05  # 진입가 기준 5% 이익

# 경로가 없으면 생성
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, BACKTEST_DATA_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)


def validate_settings() -> Tuple[bool, List[str]]:
    """
    설정 검증
    
    Returns:
        (is_valid, errors): (유효성 여부, 오류 목록)
    """
    errors = []
    
    # API 키 검증
    if not BINANCE_API_KEY or BINANCE_API_KEY == 'your_api_key_here':
        errors.append("BINANCE_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    
    if not BINANCE_API_SECRET or BINANCE_API_SECRET == 'your_api_secret_here':
        errors.append("BINANCE_API_SECRET이 설정되지 않았습니다. .env 파일을 확인하세요.")
    
    # 모드 검증
    if APP_MODE not in ['development', 'production']:
        errors.append(f"APP_MODE는 'development' 또는 'production'이어야 합니다. 현재: {APP_MODE}")
    
    # 로그 레벨 검증
    valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if LOG_LEVEL not in valid_log_levels:
        errors.append(f"LOG_LEVEL은 {valid_log_levels} 중 하나여야 합니다. 현재: {LOG_LEVEL}")
    
    # 거래 타입 검증
    if TRADING_TYPE not in ['spot', 'futures']:
        errors.append(f"TRADING_TYPE은 'spot' 또는 'futures'여야 합니다. 현재: {TRADING_TYPE}")
    
    return len(errors) == 0, errors


if __name__ == "__main__":
    # 설정 검증 테스트
    is_valid, errors = validate_settings()
    
    if is_valid:
        print("✅ 설정 검증 통과")
        print(f"  - APP_MODE: {APP_MODE}")
        print(f"  - LOG_LEVEL: {LOG_LEVEL}")
        print(f"  - TEST_MODE: {TEST_MODE}")
        print(f"  - TRADING_TYPE: {TRADING_TYPE}")
        print(f"  - DEFAULT_SYMBOL: {DEFAULT_SYMBOL}")
        print(f"  - API_KEY 설정: {'✅' if BINANCE_API_KEY else '❌'}")
        print(f"  - API_SECRET 설정: {'✅' if BINANCE_API_SECRET else '❌'}")
    else:
        print("❌ 설정 검증 실패:")
        for error in errors:
            print(f"  - {error}")