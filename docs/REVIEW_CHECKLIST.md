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

- [ ] `close_position()` 메서드가 ExposureManager와 연동되는가?
  - 포지션 청산 시 `exposure_manager.remove_position()` 호출 확인

#### 2. SpotOrderManager 통합 확인
- [ ] `SpotOrderManager.__init__()`에서 리스크 관리 모듈이 초기화되었는가?
  - 파일: `src/execution/spot_order_manager.py`
  - 확인: `self.stop_loss_manager` 속성 존재 확인
  - 확인: `self.position_sizer` 속성 존재 확인

- [ ] `check_risk_management()` 메서드가 새 모듈을 사용하는가?
  - 기존 로직이 `self.stop_loss_manager`를 사용하도록 변경되었는가?

- [ ] 포지션 사이징 로직이 새 모듈을 사용하는가?
  - `self.position_sizer.calculate_position_size()` 사용 확인
  - `ExposureManager.can_open_new_position()` 체크 확인

- [ ] 주문 실행 시 ExposureManager 연동 확인
  - 주문 전 노출 체크 확인
  - 주문 후 노출 업데이트 확인

#### 3. RealtimeBacktestIntegration 통합 확인
- [ ] `ExposureManager` 인스턴스가 생성 및 초기화되었는가?
  - 파일: `src/integration/realtime_backtest_integration.py`
  - 확인: `self.exposure_manager` 속성 존재 확인
  - 확인: `set_total_capital()` 호출 확인

- [ ] OrderManager 초기화 시 ExposureManager가 전달되는가?
  - OrderManager 생성자에 ExposureManager 전달 확인
  - 또는 OrderManager가 ExposureManager에 접근할 수 있는 방법 확인

- [ ] 주문 실행 전 노출 체크가 수행되는가?
  - `exposure_manager.can_open_new_position()` 호출 확인

- [ ] 주문 실행 후 노출 업데이트가 수행되는가?
  - `exposure_manager.add_position()` 또는 `remove_position()` 호출 확인

#### 4. 기존 로직 보존 확인
- [ ] 주문 실행 로직이 절대 삭제되지 않았는가?
  - `place_order()`, `cancel_order()`, `close_position()` 등 핵심 메서드 유지 확인
  - 주문 실행 로직의 핵심 기능이 그대로 유지되는가?

- [ ] 기존 동작과 호환성이 유지되는가?
  - 기존 API 인터페이스가 변경되지 않았는가?
  - 기존 호출 코드가 그대로 작동하는가?

#### 5. Spot/Futures 지원 확인
- [ ] Spot과 Futures 모두에서 정상 작동하는가?
  - `TRADING_TYPE`에 따라 적절한 모듈이 사용되는가?
  - Spot 특화 로직과 Futures 특화 로직이 구분되어 있는가?

#### 6. 레거시 코드 처리 (Should-have)
- [ ] 레거시 리스크 관리 로직이 제거되었는가? (선택적)
  - `check_risk_management()` 내부의 레거시 로직 제거 확인
  - `calculate_position_size()` 내부의 레거시 로직 제거 확인
  - 주의: 모든 테스트 통과 후에만 제거

#### 7. 문서 업데이트 확인
- [ ] `docs/CONTEXT.md`가 업데이트되었는가?
  - Agent 1 작업 완료 체크 확인
- [ ] `docs/PRIORITIES.md`가 업데이트되었는가?
  - 완료 항목에 ✅ 표시 확인
- [ ] `docs/PM_STATUS.md`가 업데이트되었는가?
  - Agent 1 상태가 "완료"로 변경되었는지 확인

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

