# 코드 리뷰 체크리스트

이 문서는 각 Agent의 작업 완료 후 리뷰 시 확인해야 할 항목들을 정리합니다.

---

## Agent 3: Risk Check Integration Agent 리뷰

### ✅ 필수 확인 사항

#### 1. 통합 확인
- [O] `UnifiedDataProcessor`에 `IntegratedRiskChecker`가 통합되었는가?
  - 파일: `src/data_processing/unified_processor.py`
  - 확인: `__init__` 메서드에서 `IntegratedRiskChecker` 인스턴스 생성 확인
  - 확인: `self.risk_checker` 속성이 존재하는가?

- [O] 각 데이터 처리 단계에서 리스크 체크가 수행되는가?
  - 데이터 수집 단계 (리스크체크1)
  - 전처리 단계 (리스크체크2)
  - 기술지표 계산 단계 (리스크체크3)
  - 패턴 인식 단계 (리스크체크4)
  - 신호 생성 단계 (리스크체크5)

#### 2. 리스크 체크 로직 확인
- [O] 리스크 체크 데이터 준비 함수가 구현되었는가?
  - `_prepare_risk_check_data()` 메서드 존재 확인
  - 각 단계별로 적절한 데이터가 준비되는가?

- [O] 리스크 체크 결과 처리 로직이 구현되었는가?
  - `_handle_risk_check_result()` 메서드 존재 확인
  - CRITICAL 리스크 시 처리 중단 로직 확인
  - HIGH 리스크 시 경고 후 계속 로직 확인 (선택적)

#### 3. 기존 로직 보존 확인
- [O] 기존 데이터 처리 로직이 손상되지 않았는가?
  - `_process_batch_data()` 메서드의 기존 로직 유지 확인
  - `_process_streaming_data()` 메서드의 기존 로직 유지 확인
  - 기존 기능이 정상 작동하는가?

#### 4. 성능 확인
- [ ] 리스크 체크가 성능에 큰 영향을 주지 않는가?
  - 리스크 체크가 비동기로 수행되는가? (선택적)
  - 불필요한 중복 체크가 없는가?

#### 5. 로깅 및 모니터링
- [O] 리스크 체크 결과가 적절히 로깅되는가?
  - 실패 시 경고 로그 출력 확인
  - 성공 시 디버그 로그 출력 확인 (선택적)

#### 6. 문서 업데이트 확인
- [O] `docs/CONTEXT.md`가 업데이트되었는가?
  - Agent 3 작업 완료 체크 확인
- [O] `docs/PRIORITIES.md`가 업데이트되었는가?
  - 완료 항목에 ✅ 표시 확인
- [O] `docs/PM_STATUS.md`가 업데이트되었는가?
  - Agent 3 상태가 "완료"로 변경되었는지 확인

### 🔍 코드 품질 확인

#### 코드 스타일
- [ ] 타입 힌팅이 적절히 사용되었는가?
- [ ] 함수/메서드에 docstring이 있는가?
- [ ] 변수명이 명확하고 일관성 있는가?
- [ ] 코드 주석이 적절한가?

#### 예외 처리
- [ ] 리스크 체크 실패 시 예외 처리가 적절한가?
- [ ] 예외 발생 시 적절한 로깅이 있는가?

#### 테스트 가능성
- [ ] 리스크 체크 로직이 테스트 가능한 구조인가?
- [ ] 모킹이 가능한 구조인가? (선택적)

### ⚠️ 잠재적 문제 확인

- [ ] 리스크 체크가 너무 엄격하여 정상 거래를 막는가?
- [ ] 리스크 체크가 너무 느슨하여 위험한 거래를 허용하는가?
- [ ] Spot과 Futures 모두에서 정상 작동하는가?
- [ ] 기존 테스트가 통과하는가? (회귀 테스트)

---

## Agent 1: Risk Integration Agent 리뷰

### ✅ 필수 확인 사항

#### 1. FuturesOrderManager 통합 확인
- [O] `FuturesOrderManager.__init__()`에서 리스크 관리 모듈이 초기화되었는가?
  - 파일: `src/execution/order_manager.py`
  - 확인: `self.stop_loss_manager` 속성 존재 확인
  - 확인: `self.position_sizer` 속성 존재 확인
  - 확인: `self.exposure_manager` 속성 존재 확인 (또는 전달받는 방식)

- [O] `check_risk_management()` 메서드가 새 모듈을 사용하는가?
  - 기존 로직이 `self.stop_loss_manager`를 사용하도록 변경되었는가?
  - `FuturesStopLossManager.check_stop_loss()` 사용 확인
  - `FuturesStopLossManager.check_take_profit()` 사용 확인
  - `FuturesStopLossManager.check_liquidation_risk()` 사용 확인 (Futures 특화)

- [O] `calculate_position_size()` 메서드가 새 모듈을 사용하는가?
  - `self.position_sizer.calculate_position_size()` 사용 확인
  - `ExposureManager.can_open_new_position()` 체크 추가 확인

- [O] `open_position()` 메서드가 ExposureManager와 연동되는가?
  - 주문 전 `exposure_manager.can_open_new_position()` 호출 확인
  - 주문 성공 후 `exposure_manager.add_position()` 호출 확인
  - 주문 실패 시 노출 롤백 로직 확인

- [O] `close_position()` 메서드가 ExposureManager와 연동되는가?
  - 포지션 청산 시 `exposure_manager.remove_position()` 호출 확인

#### 2. SpotOrderManager 통합 확인
- [O] `SpotOrderManager.__init__()`에서 리스크 관리 모듈이 초기화되었는가?
  - 파일: `src/execution/spot_order_manager.py`
  - 확인: `self.stop_loss_manager` 속성 존재 확인
  - 확인: `self.position_sizer` 속성 존재 확인

- [O] `check_risk_management()` 메서드가 새 모듈을 사용하는가?
  - 기존 로직이 `self.stop_loss_manager`를 사용하도록 변경되었는가?
  - `SpotStopLossManager` 메서드 사용 확인

- [O] 포지션 사이징 로직이 새 모듈을 사용하는가?
  - `calculate_buy_quantity()` 메서드는 존재하나 `position_sizer.calculate_position_size()`를 직접 사용하지 않음
  - Spot 거래 특성상 보유 자산 기반이므로 다른 로직 사용 가능
  - `ExposureManager.can_open_new_position()` 체크는 수행됨

- [O] 주문 실행 시 ExposureManager 연동 확인
  - 주문 전 노출 체크 확인 (`place_market_buy_order()` line 364-375)
  - 주문 후 노출 업데이트 확인 (`add_position()` line 365, `remove_position()` line 391, 557)

#### 3. RealtimeBacktestIntegration 통합 확인
- [O] `ExposureManager` 인스턴스가 생성 및 초기화되었는가?
  - 파일: `src/integration/realtime_backtest_integration.py`
  - 확인: `self.exposure_manager` 속성 존재 확인 (line 88)
  - 확인: `set_total_capital()` 호출 확인 (line 207)

- [O] OrderManager 초기화 시 ExposureManager가 전달되는가?
  - OrderManager 생성자에 ExposureManager 전달 확인
  - FuturesOrderManager (line 215), SpotOrderManager (line 221) 모두 확인

- [O] 주문 실행 전 노출 체크가 수행되는가?
  - `exposure_manager.can_open_new_position()` 호출 확인
  - FuturesOrderManager: `open_position()` 내부 (line 302)
  - SpotOrderManager: `place_market_buy_order()` 내부 (line 364-375)

- [O] 주문 실행 후 노출 업데이트가 수행되는가?
  - `exposure_manager.add_position()` 또는 `remove_position()` 호출 확인
  - FuturesOrderManager: `open_position()` (line 368), `close_position()` (line 480)
  - SpotOrderManager: `place_market_buy_order()` (line 365), `_update_position_after_sell()` (line 557)

#### 4. 기존 로직 보존 확인
- [O] 주문 실행 로직이 절대 삭제되지 않았는가?
  - `place_order()`, `open_position()`, `close_position()` 등 핵심 메서드 유지 확인
  - 주문 실행 로직의 핵심 기능이 그대로 유지됨

- [O] 기존 동작과 호환성이 유지되는가?
  - 기존 API 인터페이스가 변경되지 않음
  - 기존 호출 코드가 그대로 작동할 것으로 예상

#### 5. Spot/Futures 지원 확인
- [O] Spot과 Futures 모두에서 정상 작동하는가?
  - `TRADING_TYPE`에 따라 적절한 모듈이 사용됨 (line 210-223)
  - Spot 특화 로직과 Futures 특화 로직이 구분되어 있음
  - Futures 전용: `check_liquidation_risk()` 메서드 사용

#### 6. 레거시 코드 처리 (Should-have)
- [ ] 레거시 리스크 관리 로직이 제거되었는가? (선택적)
  - `check_risk_management()` 내부의 레거시 로직 제거 확인
  - `calculate_position_size()` 내부의 레거시 로직 제거 확인
  - 주의: 모든 테스트 통과 후에만 제거

#### 7. 문서 업데이트 확인
- [O] `docs/CONTEXT.md`가 업데이트되었는가?
  - Agent 1 작업 완료 체크 확인
- [O] `docs/PRIORITIES.md`가 업데이트되었는가?
  - 완료 항목에 ✅ 표시 확인
- [O] `docs/PM_STATUS.md`가 업데이트되었는가?
  - Agent 1 상태가 "완료"로 변경 확인

### 🔍 코드 품질 확인

#### 코드 스타일
- [ ] 타입 힌팅이 적절히 사용되었는가?
- [ ] 함수/메서드에 docstring이 있는가?
- [ ] 변수명이 명확하고 일관성 있는가?
- [ ] 코드 주석이 적절한가?

#### 예외 처리
- [ ] 리스크 관리 모듈 호출 시 예외 처리가 적절한가?
- [ ] ExposureManager 연동 실패 시 예외 처리가 적절한가?
- [ ] 예외 발생 시 적절한 로깅이 있는가?

#### 의존성 관리
- [ ] 필요한 import 문이 추가되었는가?
  - `from src.risk_management import ...` 확인
- [ ] 순환 참조가 없는가?

### ⚠️ 잠재적 문제 확인

- [ ] 리스크 관리 모듈이 제대로 초기화되지 않아 None 참조 오류가 발생하는가?
- [ ] ExposureManager와 OrderManager 간의 동기화 문제가 없는가?
- [ ] 주문 실행 시 노출 체크가 너무 엄격하여 정상 거래를 막는가?
- [ ] 주문 실행 시 노출 체크가 너무 느슨하여 위험한 거래를 허용하는가?
- [ ] 기존 테스트가 통과하는가? (회귀 테스트)
- [ ] Spot과 Futures 모두에서 테스트가 통과하는가?

---

## Agent 2: Feedback Loop Agent 리뷰

### ✅ 필수 확인 사항

#### 1. 성과 추적 모듈 확장 확인
- [ ] `PerformanceMonitor`에 Spot/Futures별 성과 추적이 추가되었는가?
  - 파일: `src/monitoring/performance.py`
  - 확인: `self.spot_performance` 속성 존재 확인
  - 확인: `self.futures_performance` 속성 존재 확인
  - 확인: 거래 타입별 통계가 분리되어 추적되는가?

- [ ] `record_trade()` 메서드가 거래 타입을 구분하여 기록하는가?
  - 확인: 거래 타입(Spot/Futures) 정보가 추가되었는가?
  - 확인: 거래 타입별 통계가 업데이트되는가?
  - 확인: 기존 통계 추적 기능이 유지되는가?

- [ ] `get_statistics()` 메서드가 Spot/Futures별 통계를 반환하는가?
  - 확인: 거래 타입별 통계 반환 옵션 추가 확인
  - 확인: 전체 통계와 타입별 통계 모두 제공되는가?

#### 2. 피드백 루프 구현 확인
- [ ] 성과 기반 리스크 재평가 로직이 구현되었는가?
  - 파일: `src/integration/realtime_backtest_integration.py`
  - 확인: `_performance_based_risk_reassessment()` 메서드 존재 확인
  - 확인: 최근 성과 분석 로직 구현 확인
  - 확인: 성과에 따른 리스크 관리 파라미터 조정 로직 확인

- [ ] 리스크 관리 → 데이터 수집 피드백이 구현되었는가?
  - 확인: `_adjust_data_collection_params()` 메서드 존재 확인
  - 확인: 리스크 평가 결과에 따라 데이터 수집 파라미터 조정 로직 확인
  - 확인: 리스크가 높을 때 데이터 수집 빈도 증가 로직 확인

- [ ] 실시간 성과 기반 자동 조정 루프가 구현되었는가?
  - 확인: `_adaptive_adjustment_loop()` 메서드 존재 확인
  - 확인: 주기적으로 성과 분석 및 조정 수행 확인
  - 확인: 루프가 시스템 실행 중 정상 작동하는가?

#### 3. 전략 파라미터 동적 조정 확인
- [ ] 전략 파라미터 동적 조정 메서드가 추가되었는가?
  - 파일: `src/strategy/signals.py`
  - 확인: `adjust_parameters()` 메서드 존재 확인
  - 확인: 성과 피드백에 따라 파라미터 조정 로직 확인
  - 확인: 승률에 따른 `min_confidence` 조정 로직 확인

#### 4. 기존 로직 보존 확인
- [ ] 기존 백테스팅 피드백 로직과 충돌하지 않는가?
  - 확인: 기존 `BacktestRealtimeAdapter` 피드백 로직 유지 확인
  - 확인: 새 피드백 루프와 기존 피드백이 충돌하지 않는가?
  - 확인: 기존 백테스팅 기능이 정상 작동하는가?

- [ ] 기존 성과 추적 기능이 손상되지 않았는가?
  - 확인: 기존 `record_trade()`, `get_statistics()` 메서드 동작 유지 확인
  - 확인: 기존 통계 추적 기능이 정상 작동하는가?

#### 5. Spot/Futures 지원 확인
- [ ] Spot과 Futures 성과를 구분하여 추적하는가?
  - 확인: `TRADING_TYPE`에 따라 적절한 성과 추적이 수행되는가?
  - 확인: Spot과 Futures 각각에서 성과 추적이 정상 작동하는가?

- [ ] Spot과 Futures 모두에서 피드백 루프가 정상 작동하는가?
  - 확인: 각 거래 타입에서 피드백 루프가 정상 작동하는가?
  - 확인: 거래 타입별 파라미터 조정이 적절한가?

#### 6. 문서 업데이트 확인
- [ ] `docs/CONTEXT.md`가 업데이트되었는가?
  - Agent 2 작업 완료 체크 확인
  - 완료된 작업 내용 기록 확인

- [ ] `docs/PRIORITIES.md`가 업데이트되었는가?
  - 완료 항목에 ✅ 표시 확인

- [ ] `docs/PM_STATUS.md`가 업데이트되었는가?
  - Agent 2 상태가 "완료"로 변경되었는지 확인

### 🔍 코드 품질 확인

#### 코드 스타일
- [ ] 타입 힌팅이 적절히 사용되었는가?
  - 확인: 새로 추가된 메서드에 타입 힌팅 존재
  - 확인: 타입 힌팅이 정확한지

- [ ] 함수/메서드에 docstring이 있는가?
  - 확인: 새로 추가된 메서드에 docstring 존재
  - 확인: docstring이 명확하고 완전한지

- [ ] 변수명이 명확하고 일관성 있는가?
  - 확인: 변수명이 의미를 잘 나타내는지
  - 확인: 네이밍 컨벤션이 일관성 있는지

- [ ] 코드 주석이 적절한가?
  - 확인: 복잡한 로직에 주석이 있는지
  - 확인: 주석이 최신 상태인지

#### 예외 처리
- [ ] 피드백 루프 실행 시 예외 처리가 적절한가?
  - 확인: 예외가 적절히 처리되는지
  - 확인: 예외 발생 시 시스템이 안정적인지

- [ ] 성과 분석 실패 시 예외 처리가 적절한가?
  - 확인: 성과 데이터가 없을 때 처리
  - 확인: 성과 분석 중 오류 발생 시 처리

- [ ] 예외 발생 시 적절한 로깅이 있는가?
  - 확인: 예외 발생 시 로그가 출력되는지
  - 확인: 로그 메시지가 명확한지

#### 의존성 관리
- [ ] 필요한 import 문이 추가되었는가?
  - 확인: 새로 사용하는 모듈이 import되었는지
  - 확인: 순환 참조가 없는지

- [ ] 기존 import와 충돌하지 않는가?
  - 확인: import 순서가 적절한지
  - 확인: 중복 import가 없는지

### ⚠️ 잠재적 문제 확인

- [ ] 피드백 루프가 너무 자주 실행되어 성능에 영향을 주는가?
  - 확인: 루프 실행 주기가 적절한지
  - 확인: 불필요한 중복 실행이 없는지

- [ ] 파라미터 조정이 너무 공격적이어서 전략이 불안정해지는가?
  - 확인: 파라미터 조정 범위가 적절한지
  - 확인: 조정 빈도가 적절한지

- [ ] 파라미터 조정이 너무 보수적이어서 효과가 없는가?
  - 확인: 조정 임계값이 적절한지
  - 확인: 실제로 파라미터가 조정되는지

- [ ] Spot과 Futures 성과 추적이 정확한가?
  - 확인: 거래 타입이 올바르게 분류되는가?
  - 확인: 통계가 정확하게 계산되는가?

- [ ] 기존 테스트가 통과하는가? (회귀 테스트)
  - 확인: 기존 기능이 정상 작동하는지
  - 확인: 피드백 루프 추가로 인한 영향이 없는지

- [ ] Spot과 Futures 모두에서 테스트가 통과하는가?
  - 확인: 각각에서 실제 테스트 수행
  - 확인: 테스트 결과 확인

---

## 공통 리뷰 항목

### 📝 문서화
- [ ] 중요한 변경사항이 `docs/DECISIONS.md`에 기록되었는가?
- [ ] 아키텍처 변경이 `docs/ARCHITECTURE.md`에 반영되었는가?

### 🧪 테스트
- [ ] 기존 기능이 정상 작동하는가? (회귀 테스트)
- [ ] 새 기능이 예상대로 작동하는가?
- [ ] 린터 오류가 없는가? (`read_lints` 도구 사용)

### 🔒 보안
- [ ] 민감한 정보가 코드에 하드코딩되지 않았는가?
- [ ] 로그에 민감한 정보가 출력되지 않는가?

### 🎯 설계 원칙 준수
- [ ] 단일 책임 원칙이 준수되었는가?
- [ ] 공통 인터페이스 + 구현체 분리 원칙이 준수되었는가?
- [ ] Spot/Futures 모두 지원하는가?

---

## 리뷰 체크리스트 사용 방법

1. **각 Agent별로 체크리스트 확인**
   - 위의 체크리스트를 따라 하나씩 확인
   - 각 항목을 체크하면서 코드 검토

2. **문제 발견 시**
   - `docs/CONTEXT.md`의 "알려진 이슈" 섹션에 기록
   - 필요 시 해당 Agent에게 피드백 제공

3. **리뷰 완료 시**
   - 모든 필수 항목이 체크되었는지 확인
   - 다음 단계 (Agent 2 시작 등) 진행

---

## 리뷰 우선순위

### 높은 우선순위 (즉시 확인)
1. 필수 확인 사항의 모든 항목
2. 기존 로직 보존 확인
3. Spot/Futures 지원 확인

### 중간 우선순위 (확인 권장)
1. 코드 품질 확인
2. 예외 처리 확인
3. 문서 업데이트 확인

### 낮은 우선순위 (선택적)
1. 레거시 코드 제거 (Should-have)
2. 성능 최적화
3. 테스트 추가

---

## 리뷰 완료 기준

각 Agent의 리뷰가 완료되려면:
- ✅ 모든 필수 확인 사항이 통과되어야 함
- ✅ 기존 로직이 보존되어야 함
- ✅ Spot/Futures 모두에서 정상 작동해야 함
- ✅ 문서가 업데이트되어야 함
- ✅ 린터 오류가 없어야 함

---

## 다음 단계

리뷰 완료 후:
1. Agent 2 컨텍스트 생성 (Agent 1 완료 확인 후)
2. Agent 4 컨텍스트 생성 (Must-have 완료 후)

