# 바이낸스 퀀트 트레이딩 시스템

바이낸스 스팟 거래를 위한 자동화된 퀀트 트레이딩 시스템입니다.

## 기능

- 실시간 및 과거 시장 데이터 수집
- 기술적 지표 기반 분석
- 다양한 트레이딩 전략 구현
- 리스크 관리 및 포지션 사이징
- 성과 모니터링 및 분석

## 설치 방법

1. 저장소 클론
git clone https://github.com/yourusername/Quant_Trading_Binance_Spot.git
cd Quant_Trading_Binance_Spot

markdown

2. 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows

markdown

3. 의존성 설치
pip install -e .

markdown

4. 환경 변수 설정
`.env` 파일을 생성하고 필요한 API 키와 설정을 추가하세요.

## 사용 방법

1. 데이터 수집
python -m src.data_collection.collectors

markdown

2. 전략 백테스트
python -m src.strategy.backtest

markdown

3. 실시간 트레이딩
python -m src.main

ini

## 프로젝트 구조

[프로젝트 구조 설명]

## 기여 방법

[기여 방법 설명]

## 라이선스

[라이선스 정보]