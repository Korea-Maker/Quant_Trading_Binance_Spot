# To Do List

**최종 업데이트**: 2025-12-02  
**현재 Phase**: Phase 2 (Must-have 완료, Should-have 진행 중)

---

## 📋 즉시 시작 가능한 작업 (Phase 2 Should-have)

### 높은 우선순위

#### 1. Data Collection 리스크 체크 데이터 수집 로직 구현
- [ ] **`data_delay_ms` 실제 계산 로직 추가**
  - 파일: `src/data_processing/unified_processor.py`
  - 메서드: `_prepare_risk_check_data()` (data_collection 단계)
  - 작업: DataFrame의 마지막 타임스탬프와 현재 시간 비교하여 계산
  - 예상 작업량: 중간
  - 참고: `docs/SHOULD_HAVE_REVIEW.md`

- [ ] **`connection_status` 실제 확인 로직 추가**
  - 파일: `src/data_processing/unified_processor.py`
  - 메서드: `_prepare_risk_check_data()` (data_collection 단계)
  - 작업: 웹소켓 클라이언트의 `connected` 속성 확인
  - 예상 작업량: 낮음
  - 참고: `docs/SHOULD_HAVE_REVIEW.md`

#### 2. 주문 성공 여부 확인 로직 추가
- [ ] **주문 응답 확인 후 노출 업데이트**
  - 파일: `src/execution/order_manager.py`, `src/execution/spot_order_manager.py`
  - 메서드: `open_position()`, `place_market_buy_order()`
  - 작업: 주문 응답을 받은 후 `order.status == 'FILLED'` 확인, 성공 시에만 `add_position()` 호출
  - 예상 작업량: 낮음
  - 참고: `docs/SHOULD_HAVE_REVIEW.md`

- [ ] **주문 성공 여부(`order.status == 'FILLED'`) 확인**
  - 파일: `src/execution/order_manager.py`, `src/execution/spot_order_manager.py`
  - 작업: 주문 응답에서 상태 확인, FILLED가 아닌 경우 노출 추가하지 않음
  - 예상 작업량: 낮음
  - 참고: `docs/SHOULD_HAVE_REVIEW.md`

### 중간 우선순위

- [ ] **하이브리드 방식 구현 (주문 성공 여부 확인)**
  - 파일: `src/execution/order_manager.py`, `src/execution/spot_order_manager.py`
  - 작업: 주문 전 `can_open_new_position()` 체크 + 주문 성공 후 `add_position()` 호출
  - 예상 작업량: 중간
  - 참고: `docs/SHOULD_HAVE_REVIEW.md`

### 낮은 우선순위 (선택적)

- [ ] **`orderbook` 데이터 수집 기능 추가**
  - 파일: `src/data_collection/websocket_client.py`, `src/data_processing/unified_processor.py`
  - 작업: 웹소켓 클라이언트에 오더북 구독 기능 추가, 유동성 리스크 체크에 활용
  - 예상 작업량: 높음
  - 참고: `docs/SHOULD_HAVE_REVIEW.md`

- [ ] **부분 체결 처리 로직 추가**
  - 파일: `src/execution/order_manager.py`, `src/execution/spot_order_manager.py`
  - 작업: 부분 체결 시 부분 노출만 추가, 나머지 체결 대기
  - 예상 작업량: 높음
  - 참고: `docs/SHOULD_HAVE_REVIEW.md`

---

## 🎨 Agent 4: Dashboard Agent (Should-have)

- [ ] **웹 기반 실시간 대시보드 구현**
  - 파일: `src/monitoring/web_dashboard/` (새 모듈 생성)
  - 작업:
    - Flask/FastAPI + React 기반 웹 대시보드
    - 실시간 차트 (Candlestick, 지표, 신호)
    - 포지션 모니터링 (Spot: 보유량, Futures: 레버리지, 마진)
    - 성과 분석 차트 (Equity Curve, Drawdown)
    - Spot/Futures 모드 표시 및 전환 기능
  - 예상 작업량: 높음
  - 참고: `docs/AGENT_ROLES.md`, `docs/AGENT_TASKS.md`, `docs/PM_STATUS.md`
  - 주의사항:
    - 기존 콘솔 대시보드와 병행 운영 가능하도록
    - 보안 고려 (API 키 등 민감 정보 노출 방지)

---

## 🧹 레거시 코드 정리

- [ ] **`unified_processor.py`의 레거시 `_remove_outliers()` 제거**
  - 파일: `src/data_processing/unified_processor.py`
  - 작업: `DataPreprocessor` 사용 확인 후 레거시 메서드 제거
  - 예상 작업량: 낮음
  - 참고: `docs/PRIORITIES.md` (레거시 코드 관리 섹션)
  - 삭제 전 확인:
    - [ ] 모든 호출 지점이 새 모듈 사용 중
    - [ ] 테스트 통과 확인

---

## 🚀 Phase 3: Medium Priority (Must-have)

### 이벤트 기반 아키텍처

- [ ] **Event Bus 구현**
  - 파일: `src/core/event_bus.py` (새 모듈)
  - 작업:
    - 이벤트 발행/구독 시스템 구현
    - 모듈 간 느슨한 결합
    - 비동기 이벤트 처리
  - 예상 작업량: 높음
  - 참고: `docs/PRIORITIES.md`

- [ ] **모듈 간 이벤트 기반 통신으로 전환**
  - 파일: 전체 시스템
  - 작업: 직접 호출을 이벤트 기반으로 전환
  - 예상 작업량: 매우 높음
  - 참고: `docs/ARCHITECTURE.md`

---

## 📊 Phase 3: Medium Priority (Should-have)

### 데이터 품질 관리 강화

- [ ] **데이터 검증 파이프라인 구현**
  - 파일: `src/data_processing/data_validator.py` (새 모듈)
  - 작업: 데이터 검증 규칙 정의 및 실행
  - 예상 작업량: 중간

- [ ] **데이터 품질 메트릭 구현**
  - 파일: `src/data_processing/data_quality.py` (새 모듈)
  - 작업: 데이터 품질 지표 계산 및 모니터링
  - 예상 작업량: 중간

### 백테스팅 기능 확장

- [ ] **Spot/Futures 구분 백테스팅**
  - 파일: `src/backtesting/backtest_engine.py`
  - 작업: 거래 타입별 백테스팅 결과 분리
  - 예상 작업량: 중간

- [ ] **Walk-Forward Analysis 구현**
  - 파일: `src/backtesting/walk_forward.py` (새 모듈)
  - 작업: Walk-Forward Analysis 알고리즘 구현
  - 예상 작업량: 높음

---

## 🧪 테스트 및 품질 관리

### 단위 테스트

- [ ] **리스크 관리 모듈 단위 테스트**
  - 파일: `tests/test_risk_management.py` (새 파일)
  - 작업: stop_loss, position_sizing, exposure_manager 테스트
  - 예상 작업량: 중간
  - 참고: `docs/PRIORITIES.md`

- [ ] **리스크 체크 통합 테스트**
  - 파일: `tests/test_risk_checker.py` (새 파일)
  - 작업: IntegratedRiskChecker 테스트
  - 예상 작업량: 중간
  - 참고: `docs/PRIORITIES.md`

### 통합 테스트

- [ ] **실제 거래 환경 테스트 (선택적)**
  - 작업: 테스트넷에서 실제 거래 시나리오 테스트
  - 예상 작업량: 중간
  - 주의: 테스트넷 사용, 실제 자금 사용 금지

---

## 📝 문서화

- [ ] **DECISIONS.md 업데이트**
  - 작업: 새로운 의사결정 기록 (필요 시)
  - 예상 작업량: 낮음

- [ ] **ARCHITECTURE.md 업데이트**
  - 작업: 새로운 모듈이나 인터페이스 추가 시 업데이트
  - 예상 작업량: 낮음

- [ ] **코드 주석 및 docstring 보완**
  - 작업: 복잡한 로직에 주석 추가
  - 예상 작업량: 중간

---

## 🔍 코드 리뷰 및 개선

- [ ] **전체 코드 리뷰 (선택적)**
  - 작업: 코드 품질, 성능, 보안 검토
  - 예상 작업량: 높음

- [ ] **성능 최적화 (선택적)**
  - 작업: 병목 지점 식별 및 최적화
  - 예상 작업량: 높음
  - 참고: `docs/PRIORITIES.md` (Nice-to-have)

---

## 📌 우선순위 가이드

### 즉시 시작 권장 (Phase 2 Should-have)
1. Data Collection 리스크 체크 - `data_delay_ms` 계산
2. Data Collection 리스크 체크 - `connection_status` 확인
3. 주문 성공 여부 확인 - 주문 응답 확인 후 노출 업데이트
4. 주문 성공 여부 확인 - `order.status == 'FILLED'` 확인

### Must-have 완료 후 (Phase 3)
5. 이벤트 기반 아키텍처 도입
6. 데이터 품질 관리 강화
7. 백테스팅 기능 확장

### 독립적 작업 (병렬 가능)
- Agent 4 (Dashboard Agent)
- 레거시 코드 정리
- 단위 테스트 작성

---

## 📚 참고 문서

- **현재 컨텍스트**: `docs/CONTEXT.md`
- **우선순위 관리**: `docs/PRIORITIES.md`
- **PM 상태**: `docs/PM_STATUS.md`
- **Should-have 검토**: `docs/SHOULD_HAVE_REVIEW.md`
- **Agent 역할**: `docs/AGENT_ROLES.md`
- **Agent 작업 명세**: `docs/AGENT_TASKS.md`
- **통합 테스트 결과**: `docs/INTEGRATION_TEST_RESULTS.md`
- **버그 수정**: `docs/BUGFIX_PATTERN_DESCRIPTIONS.md`

---

## ✅ 완료된 작업 (참고용)

### Phase 1: Critical (완료)
- ✅ 리스크 관리 모듈 구현
- ✅ 단계별 리스크 체크 시스템
- ✅ 데이터 전처리 모듈 명시화

### Phase 2: High Priority - Must-have (완료)
- ✅ Agent 1: 리스크 관리 모듈 통합
- ✅ Agent 2: 피드백 루프 강화
- ✅ Agent 3: 리스크 체크 통합
- ✅ 통합 테스트 완료
- ✅ 버그 수정 (PATTERN_DESCRIPTIONS float 변환 오류)

---

**마지막 업데이트**: 2025-12-02

