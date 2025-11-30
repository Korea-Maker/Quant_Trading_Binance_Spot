# 바이낸스 퀀트 트레이딩 시스템

바이낸스 스팟 거래를 위한 자동화된 퀀트 트레이딩 시스템입니다.

## 기능

- 실시간 및 과거 시장 데이터 수집
- 기술적 지표 기반 분석
- 다양한 트레이딩 전략 구현
- 리스크 관리 및 포지션 사이징
- 성과 모니터링 및 분석

## 설치 방법

### 빠른 설치 (권장)

```bash
# 1. 가상 환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate  # Windows
# 또는
source .venv/bin/activate  # Linux/Mac

# 2. 의존성 자동 설치
python install_dependencies.py
```

### 수동 설치

```bash
# 1. 가상 환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate  # Windows

# 2. 필수 패키지 설치
pip install websocket-client>=1.6.0
pip install ta_lib-0.6.3-cp313-cp313-win_amd64.whl  # Windows wheel 파일
pip install -r requirements.txt

# 3. 환경 변수 설정
# .env 파일을 생성하고 필요한 API 키와 설정을 추가하세요.
```

자세한 설치 가이드는 [INSTALL.md](INSTALL.md) 또는 [QUICK_START.md](QUICK_START.md)를 참조하세요.

## API 키 설정

**⚠️ 중요**: 실제 거래를 실행하려면 바이낸스 API 키 설정이 필요합니다.

자세한 설정 방법은 [API_SETUP_GUIDE.md](API_SETUP_GUIDE.md)를 참조하세요.

## 사용 방법

### 기본 실행

```bash
# 메인 시스템 실행
python main.py
```

### 테스트

```bash
# TA-Lib 설치 확인
python test_talib.py

# 전체 시스템 테스트
python test_system.py
```

### 개별 모듈 실행

```bash
# 데이터 수집
python -m src.data_collection.collectors

# 전략 백테스트
python -m src.backtesting.backtest_engine
```

ini

## 프로젝트 구조

[프로젝트 구조 설명]

## 기여 방법

[기여 방법 설명]

## 라이선스

[라이선스 정보]