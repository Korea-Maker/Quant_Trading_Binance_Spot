# 빠른 시작 가이드

## 1. 의존성 설치

### 자동 설치 (권장)

```bash
# 가상환경 활성화
.venv\Scripts\activate  # Windows
# 또는
source .venv/bin/activate  # Linux/Mac

# 자동 설치 스크립트 실행
python install_dependencies.py
```

### 수동 설치

```bash
# 가상환경 활성화
.venv\Scripts\activate  # Windows

# websocket-client 설치
pip install websocket-client>=1.6.0

# TA-Lib 설치 (wheel 파일 사용)
pip install ta_lib-0.6.3-cp313-cp313-win_amd64.whl

# 기타 의존성 설치
pip install -r requirements.txt
```

## 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 추가하세요:

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
DEFAULT_SYMBOL=BTCUSDT
TEST_MODE=True
```

**📖 자세한 API 키 설정 방법**: [API_SETUP_GUIDE.md](API_SETUP_GUIDE.md) 참조

## 3. 시스템 실행

```bash
python main.py
```

## 문제 해결

### API 키 오류 (거래 실행 실패)

거래가 실행되지 않거나 "Invalid API-key" 오류가 발생하는 경우:

1. [API_SETUP_GUIDE.md](API_SETUP_GUIDE.md) 참조
2. 바이낸스 테스트넷에서 API 키 생성 확인
3. `.env` 파일의 API 키 확인
4. API 키 권한 확인 (Enable Spot & Margin Trading)

### TA-Lib 오류

```bash
# Python 버전 확인
python --version

# Wheel 파일 재설치
pip uninstall TA-Lib
pip install ta_lib-0.6.3-cp313-cp313-win_amd64.whl
```

### websocket-client 오류

```bash
# 방법 1: install_dependencies.py 사용 (권장)
python install_dependencies.py --reinstall-websocket

# 방법 2: 수동 재설치
pip uninstall -y websocket-client websockets
pip install --upgrade --force-reinstall websocket-client>=1.6.0
```

### 전체 재설치

```bash
# 가상환경 재생성
deactivate
rm -rf .venv  # 또는 rmdir /s .venv (Windows)
python -m venv .venv
.venv\Scripts\activate
python install_dependencies.py
```

## 테스트

```bash
# TA-Lib 테스트
python test_talib.py

# 전체 시스템 테스트
python test_system.py
```
