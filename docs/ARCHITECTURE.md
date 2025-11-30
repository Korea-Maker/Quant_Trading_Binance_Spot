# 아키텍처 결정사항 (Architecture Decisions)

이 문서는 시스템의 아키텍처 구조와 설계 원칙을 기록합니다.

## 시스템 개요

### 거래 타입 지원
- **Spot 거래**: 실제 자산 보유 기반 거래
- **Futures 거래**: 레버리지 기반 계약 거래
- **선택 방식**: `TRADING_TYPE` 환경 변수로 동적 선택

### 핵심 원칙

1. **공통 인터페이스 + 구현체 분리**
   - 모든 거래 타입이 공통 인터페이스를 구현
   - 거래 타입별 특화 로직은 구현체에서 처리

2. **단일 책임 원칙**
   - 각 모듈은 하나의 명확한 책임만 가짐
   - 예: `DataPreprocessor`는 전처리만, `RiskChecker`는 리스크 체크만

3. **확장성 우선**
   - 새로운 거래 타입이나 기능 추가 시 기존 코드 수정 최소화
   - Factory 패턴으로 동적 객체 생성

## 모듈 구조

### 리스크 관리 모듈 (`src/risk_management/`)

```
risk_management/
├── __init__.py              # 모듈 exports
├── stop_loss.py             # 손절/익절 관리 (Spot/Futures 공통)
├── position_sizing.py        # 포지션 사이징 (Spot/Futures 공통)
├── exposure_manager.py      # 최대 노출 제한 통합 관리
└── risk_checker.py          # 단계별 리스크 체크
```

**설계 원칙:**
- Base 클래스로 공통 인터페이스 정의
- 거래 타입별 구현체 분리
- Factory 함수로 동적 생성

### 데이터 처리 모듈 (`src/data_processing/`)

```
data_processing/
├── preprocessor.py          # 데이터 전처리 (독립 모듈)
├── indicators.py            # 기술적 지표 계산
├── pattern_recognition.py   # 패턴 인식
├── features.py             # 특징 생성 및 신호 처리
└── unified_processor.py     # 통합 데이터 프로세서
```

**데이터 흐름:**
1. 원본 데이터 → `DataPreprocessor` → 전처리된 데이터
2. 전처리된 데이터 → `TechnicalIndicators` → 지표 추가
3. 지표 데이터 → `PatternRecognition` → 패턴 인식
4. 패턴 데이터 → `TradingSignalProcessor` → 신호 생성
5. 각 단계마다 `RiskChecker`로 리스크 체크

### 실행 모듈 (`src/execution/`)

```
execution/
├── order_manager.py         # Futures 주문 관리
└── spot_order_manager.py   # Spot 주문 관리
```

**참고:** 리스크 관리 로직은 `src/risk_management/`로 분리됨

## 프로세스 플로우

### 메인 프로세스 플로우 (이미지 2 참조)

```
실시간 데이터
  ↓ [리스크체크1: 데이터 수집]
데이터 전처리
  ↓ [리스크체크2: 전처리]
기술지표 계산
  ↓ [리스크체크3: 지표 계산]
패턴 인식
  ↓ [리스크체크4: 패턴 인식]
신호 생성
  ↓ [리스크체크5: 신호 생성]
전략 실행
  ↓
리스크 관리
  ↓
성과 모니터링
```

### 피드백 루프

```
성과 기록
  ↓
리스크 재평가
  ↓
데이터 전처리 (파라미터 조정)
  ↓
포지션 업데이트 / 체결 확인 / 주문 실행
```

## 인터페이스 설계

### 리스크 관리 인터페이스

```python
# 공통 인터페이스
class BaseStopLossManager(ABC):
    @abstractmethod
    def calculate_stop_loss(...) -> float: pass
    
    @abstractmethod
    def calculate_take_profit(...) -> float: pass

# Factory 함수
def create_stop_loss_manager(trading_type: str, ...) -> BaseStopLossManager:
    if trading_type == 'futures':
        return FuturesStopLossManager(...)
    else:
        return SpotStopLossManager(...)
```

### 포지션 사이징 인터페이스

```python
class BasePositionSizer(ABC):
    @abstractmethod
    def calculate_position_size(...) -> float: pass
    
    @abstractmethod
    def check_max_exposure(...) -> bool: pass
```

## 데이터 구조

### 포지션 노출 정보

```python
@dataclass
class PositionExposure:
    symbol: str
    trading_type: str  # 'spot' or 'futures'
    position_size: float
    entry_price: float
    current_price: float
    notional_value: float      # 명목 가치
    actual_exposure: float      # 실제 노출 (Futures는 마진 기준)
    leverage: float = 1.0
```

### 리스크 체크 결과

```python
class RiskCheckResult:
    passed: bool
    risk_level: RiskLevel  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    details: Dict[str, Any]
```

## 통합 지점

### UnifiedDataProcessor 통합

- `DataPreprocessor` 인스턴스 사용
- 각 처리 단계에서 `IntegratedRiskChecker` 호출
- 리스크 체크 실패 시 처리 중단

### RealtimeBacktestIntegration 통합

- `TRADING_TYPE`에 따라 적절한 OrderManager 선택
- 리스크 관리 모듈과 OrderManager 연동 (Phase 2)
- ExposureManager로 통합 노출 관리

## 확장성 고려사항

### 새로운 거래 타입 추가

1. `BaseStopLossManager`를 상속한 새 구현체 생성
2. `BasePositionSizer`를 상속한 새 구현체 생성
3. Factory 함수에 새 거래 타입 분기 추가

### 새로운 리스크 체크 단계 추가

1. `BaseRiskChecker`를 상속한 새 체커 생성
2. `IntegratedRiskChecker.checkers`에 추가
3. 데이터 처리 파이프라인에 통합

## 성능 고려사항

- 리스크 체크는 비동기로 수행 가능 (Phase 3)
- 전처리는 배치 처리로 최적화
- 노출 관리는 메모리 기반 (대용량 시 DB 고려)

## 보안 고려사항

- API 키는 환경 변수로 관리 (`.env`)
- 로그에서 민감 정보 제거
- 리스크 관리 설정은 검증 로직 포함

