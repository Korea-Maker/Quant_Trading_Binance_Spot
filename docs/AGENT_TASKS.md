# Agent 작업 상세 명세서

각 Agent의 구체적인 작업 내용과 구현 가이드를 제공합니다.

---

## Agent 1: Risk Integration Agent

### 작업 목표
기존 OrderManager의 리스크 관리 로직을 새로 구현된 리스크 관리 모듈로 교체하고, ExposureManager와 연동합니다.

### 상세 작업 내용

#### 1. FuturesOrderManager 통합

**작업 파일**: `src/execution/order_manager.py`

**작업 내용:**
1. `FuturesOrderManager.__init__()`에서 리스크 관리 모듈 초기화
   ```python
   from src.risk_management import (
       create_stop_loss_manager,
       create_position_sizer,
       ExposureManager
   )
   
   # 초기화 시
   self.stop_loss_manager = create_stop_loss_manager(
       trading_type='futures',
       symbol=self.symbol,
       leverage=self.leverage
   )
   self.position_sizer = create_position_sizer(
       trading_type='futures',
       symbol=self.symbol,
       leverage=self.leverage
   )
   ```

2. `check_risk_management()` 메서드 수정
   - 기존 로직을 `self.stop_loss_manager` 사용하도록 변경
   - `FuturesStopLossManager.check_stop_loss()` 사용
   - `FuturesStopLossManager.check_take_profit()` 사용
   - `FuturesStopLossManager.check_liquidation_risk()` 추가

3. `calculate_position_size()` 메서드 수정
   - 기존 로직을 `self.position_sizer.calculate_position_size()` 사용하도록 변경
   - `ExposureManager.can_open_new_position()` 체크 추가

4. `place_order()` 메서드 수정
   - 주문 전 `ExposureManager.add_position()` 호출
   - 주문 성공 후 노출 업데이트
   - 주문 실패 시 노출 롤백

5. `close_position()` 메서드 수정
   - 포지션 청산 시 `ExposureManager.remove_position()` 호출

**주의사항:**
- 기존 `check_risk_management()` 메서드의 동작을 그대로 유지
- 레버리지, 마진 정보는 FuturesOrderManager에서 관리
- ExposureManager는 통합 노출 관리만 담당

#### 2. SpotOrderManager 통합

**작업 파일**: `src/execution/spot_order_manager.py`

**작업 내용:**
1. `SpotOrderManager.__init__()`에서 리스크 관리 모듈 초기화
   ```python
   from src.risk_management import (
       create_stop_loss_manager,
       create_position_sizer,
       ExposureManager
   )
   
   # 초기화 시
   self.stop_loss_manager = create_stop_loss_manager(
       trading_type='spot',
       symbol=self.symbol
   )
   self.position_sizer = create_position_sizer(
       trading_type='spot',
       symbol=self.symbol
   )
   ```

2. `check_risk_management()` 메서드 수정
   - 기존 로직을 `self.stop_loss_manager` 사용하도록 변경

3. 포지션 사이징 로직 수정
   - `self.position_sizer.calculate_position_size()` 사용
   - `ExposureManager.can_open_new_position()` 체크

4. 주문 실행 시 ExposureManager 연동
   - 주문 전 노출 체크
   - 주문 후 노출 업데이트

#### 3. RealtimeBacktestIntegration 통합

**작업 파일**: `src/integration/realtime_backtest_integration.py`

**작업 내용:**
1. `ExposureManager` 인스턴스 생성 및 초기화
   ```python
   from src.risk_management import ExposureManager
   
   self.exposure_manager = ExposureManager(
       max_total_exposure_pct=0.3,
       max_per_symbol_exposure_pct=0.1,
       max_concurrent_positions=5
   )
   self.exposure_manager.set_total_capital(total_capital)
   ```

2. OrderManager 초기화 시 ExposureManager 전달
   - OrderManager가 ExposureManager를 사용하도록 수정

3. 주문 실행 전 노출 체크
   - `exposure_manager.can_open_new_position()` 호출

4. 주문 실행 후 노출 업데이트
   - `exposure_manager.add_position()` 또는 `remove_position()` 호출

#### 4. 레거시 코드 제거 (Should-have)

**작업 파일**: 
- `src/execution/order_manager.py`
- `src/execution/spot_order_manager.py`

**작업 내용:**
1. 기존 리스크 관리 로직 제거
   - `check_risk_management()` 내부의 레거시 로직 제거
   - `calculate_position_size()` 내부의 레거시 로직 제거

2. 삭제 전 확인:
   - [ ] 새 모듈이 모든 기능을 대체하는지 확인
   - [ ] 모든 테스트 통과 확인
   - [ ] DECISIONS.md에 삭제 이유 기록

### 완료 기준
- [ ] FuturesOrderManager가 새 리스크 관리 모듈 사용
- [ ] SpotOrderManager가 새 리스크 관리 모듈 사용
- [ ] ExposureManager와 OrderManager 연동 완료
- [ ] 기존 동작과 호환성 유지 (회귀 테스트 통과)
- [ ] 레거시 코드 제거 (Should-have)

---

## Agent 2: Feedback Loop Agent

### 작업 목표
피드백 루프를 강화하여 실시간 성과 기반 자동 조정과 Spot/Futures별 성과 추적을 구현합니다.

### 상세 작업 내용

#### 1. 성과 추적 모듈 확장

**작업 파일**: `src/monitoring/performance.py`

**작업 내용:**
1. Spot/Futures별 성과 추적 추가
   ```python
   # 성과 추적을 거래 타입별로 분리
   self.spot_performance: Dict[str, Any] = {}
   self.futures_performance: Dict[str, Any] = {}
   ```

2. `record_trade()` 메서드 확장
   - 거래 타입(Spot/Futures) 정보 추가
   - 거래 타입별 통계 업데이트

3. `get_statistics()` 메서드 확장
   - Spot/Futures별 통계 반환 옵션 추가

#### 2. 피드백 루프 구현

**작업 파일**: `src/integration/realtime_backtest_integration.py`

**작업 내용:**
1. 성과 기반 리스크 재평가 로직 구현
   ```python
   async def _performance_based_risk_reassessment(self):
       """성과 기반 리스크 재평가"""
       # 최근 성과 분석
       recent_perf = self.performance_monitor.get_recent_performance(days=7)
       
       # 성과가 나쁘면 리스크 관리 강화
       if recent_perf['win_rate'] < 50:
           # 리스크 관리 파라미터 조정
           # 예: 손절 비율 감소, 포지션 크기 감소
       
       # 성과가 좋으면 리스크 관리 완화 (선택적)
   ```

2. 리스크 관리 → 데이터 수집 피드백 구현
   ```python
   def _adjust_data_collection_params(self, risk_assessment: Dict):
       """리스크 평가 결과에 따라 데이터 수집 파라미터 조정"""
       # 리스크가 높으면 더 자주 데이터 수집
       # 리스크가 낮으면 데이터 수집 빈도 감소
   ```

3. 실시간 성과 기반 자동 조정
   ```python
   async def _adaptive_adjustment_loop(self):
       """실시간 성과 기반 자동 조정 루프"""
       while self.is_running:
           # 성과 분석
           # 리스크 재평가
           # 전략 파라미터 조정
           # 데이터 수집 파라미터 조정
           await asyncio.sleep(3600)  # 1시간마다
   ```

#### 3. 전략 파라미터 동적 조정

**작업 파일**: `src/strategy/signals.py` (확장)

**작업 내용:**
1. 동적 파라미터 조정 메서드 추가
   ```python
   def adjust_parameters(self, performance_feedback: Dict):
       """성과 피드백에 따라 전략 파라미터 조정"""
       # 승률이 낮으면 min_confidence 증가
       # 승률이 높으면 min_confidence 감소 (선택적)
   ```

### 완료 기준
- [ ] 리스크 관리 → 데이터 수집 피드백 구현
- [ ] 실시간 성과 기반 자동 조정 로직 구현
- [ ] Spot/Futures별 성과 추적 기능 추가
- [ ] 성과 기록 → 리스크 재평가 → 전략 조정 파이프라인 완성

---

## Agent 3: Risk Check Integration Agent

### 작업 목표
단계별 리스크 체크를 데이터 처리 파이프라인에 통합합니다.

### 상세 작업 내용

#### 1. UnifiedDataProcessor에 RiskChecker 통합

**작업 파일**: `src/data_processing/unified_processor.py`

**작업 내용:**
1. `IntegratedRiskChecker` 인스턴스 생성
   ```python
   from src.risk_management import IntegratedRiskChecker
   from src.config.settings import TRADING_TYPE
   
   self.risk_checker = IntegratedRiskChecker(trading_type=TRADING_TYPE)
   ```

2. `_process_batch_data()` 메서드 수정
   - 각 처리 단계에서 리스크 체크 수행
   - 리스크 체크 실패 시 처리 중단 또는 경고

3. `_process_streaming_data()` 메서드 수정
   - 실시간 데이터 처리 시에도 리스크 체크 수행

4. 리스크 체크 데이터 준비
   ```python
   def _prepare_risk_check_data(self, df: pd.DataFrame, stage: str) -> Dict:
       """각 단계별 리스크 체크 데이터 준비"""
       if stage == 'data_collection':
           return {
               'price': df['close'].iloc[-1] if 'close' in df.columns else None,
               'connection_status': 'connected',
               'data_delay_ms': 0,  # 실제로는 계산 필요
               'orderbook': {}  # 실제로는 수집 필요
           }
       elif stage == 'preprocessing':
           # 전처리 통계 사용
           stats = self.preprocessor.get_preprocessing_stats()
           return {
               'missing_data_count': stats.get('missing_data', {}).get('missing_before', 0),
               'total_data_count': len(df),
               'outlier_count': stats.get('outliers', {}).get('removed_count', 0),
               'data_consistency': stats.get('data_consistency', {}).get('is_consistent', True)
           }
       # ... 각 단계별 데이터 준비
   ```

5. 리스크 체크 결과 처리
   ```python
   def _handle_risk_check_result(self, results: Dict[str, RiskCheckResult]) -> bool:
       """리스크 체크 결과 처리 및 다음 단계 진행 여부 결정"""
       can_proceed = self.risk_checker.should_proceed(results)
       
       if not can_proceed:
           self.logger.warning("리스크 체크 실패로 인해 처리 중단")
           # CRITICAL 리스크인 경우 중단
           # HIGH 리스크인 경우 경고 후 계속 (선택적)
       
       return can_proceed
   ```

#### 2. 각 처리 단계에 리스크 체크 통합

**작업 내용:**
1. 데이터 수집 단계 (리스크체크1)
   - `_process_streaming_data()` 시작 시 체크
   - 데이터 품질, 연결 상태 확인

2. 전처리 단계 (리스크체크2)
   - `_preprocess_data()` 후 체크
   - 전처리 통계를 리스크 체크 데이터로 사용

3. 기술지표 계산 단계 (리스크체크3)
   - 지표 계산 후 체크
   - 지표 신뢰도 확인

4. 패턴 인식 단계 (리스크체크4)
   - 패턴 인식 후 체크
   - 패턴 신뢰도 확인

5. 신호 생성 단계 (리스크체크5)
   - 신호 생성 후 체크
   - 신호 강도, 시장 상태 확인

### 완료 기준
- [ ] UnifiedDataProcessor에 IntegratedRiskChecker 통합
- [ ] 각 처리 단계에서 리스크 체크 수행
- [ ] 리스크 체크 실패 시 처리 중단 로직 구현
- [ ] 리스크 체크 결과 로깅 및 모니터링

---

## Agent 4: Dashboard Agent (Should-have)

### 작업 목표
웹 기반 실시간 대시보드를 구현합니다.

### 상세 작업 내용

#### 1. 백엔드 API 구현

**작업 파일**: `src/monitoring/web_dashboard/api.py` (새로 생성)

**작업 내용:**
1. FastAPI 또는 Flask 기반 REST API 구현
2. 엔드포인트:
   - `GET /api/status`: 시스템 상태
   - `GET /api/performance`: 성능 통계
   - `GET /api/positions`: 현재 포지션
   - `GET /api/signals`: 최근 신호
   - `GET /api/risk`: 리스크 정보
   - `WebSocket /ws/realtime`: 실시간 데이터 스트리밍

#### 2. 프론트엔드 구현

**작업 파일**: `src/monitoring/web_dashboard/frontend/` (새로 생성)

**작업 내용:**
1. React 기반 대시보드 구현
2. 컴포넌트:
   - 실시간 차트 (Candlestick)
   - 지표 표시
   - 포지션 모니터링
   - 성과 분석 차트
   - 리스크 정보 표시

### 완료 기준
- [ ] 웹 기반 대시보드 백엔드 구현
- [ ] 웹 기반 대시보드 프론트엔드 구현
- [ ] 실시간 차트 구현
- [ ] 포지션 모니터링 구현
- [ ] Spot/Futures 모드 표시

---

## 공통 작업 가이드

### 코드 스타일
- PEP 8 준수
- 타입 힌팅 사용
- 로깅 포함
- 예외 처리 포함

### 테스트
- 기존 기능 회귀 테스트
- 새 기능 단위 테스트 (가능한 경우)

### 문서화
- 함수/클래스 docstring 작성
- 중요한 의사결정은 DECISIONS.md에 기록

