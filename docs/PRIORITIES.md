# 우선순위 관리 (Priority Management)

이 문서는 기능과 작업의 우선순위를 관리합니다.
각 항목은 Must-have, Should-have, Nice-to-have로 분류됩니다.

## 우선순위 정의

### Must-have (필수)
- 시스템의 핵심 기능
- 없으면 시스템이 제대로 작동하지 않음
- 보안 및 안정성 관련 기능
- **삭제 금지**: 레거시로 표시되어도 삭제 전 마이그레이션 필수

### Should-have (권장)
- 시스템의 주요 기능
- 없어도 작동하지만 있으면 훨씬 좋음
- 사용자 경험 개선
- **조건부 삭제**: 대체 기능이 완성되면 삭제 가능

### Nice-to-have (선택)
- 추가 기능
- 없어도 시스템 작동에 문제 없음
- 성능 최적화, 편의 기능
- **자유 삭제**: 필요 없으면 삭제 가능

## Phase별 우선순위

### Phase 1: Critical (완료 ✅)

#### Must-have
- ✅ 리스크 관리 모듈 구현
  - `stop_loss.py` (Spot/Futures 공통 인터페이스)
  - `position_sizing.py` (Spot/Futures 차별화)
  - `exposure_manager.py` (최대 노출 제한)
- ✅ 단계별 리스크 체크 시스템
  - `risk_checker.py` (리스크체크1-5)
- ✅ 데이터 전처리 모듈 명시화
  - `preprocessor.py` (독립 모듈)

#### Should-have
- ⏳ 리스크 관리 모듈 단위 테스트
- ⏳ 리스크 체크 통합 테스트

#### Nice-to-have
- ⏳ 리스크 관리 모듈 성능 최적화

---

### Phase 2: High Priority (진행 중)

#### Must-have
- ✅ 피드백 루프 강화
  - 리스크 관리 → 데이터 수집 피드백
  - 실시간 성과 기반 자동 조정
  - Spot/Futures별 성과 추적
- ✅ 리스크 관리 모듈과 OrderManager 통합
  - 기존 OrderManager의 리스크 관리 로직을 새 모듈로 교체
  - ExposureManager와 OrderManager 연동
- ✅ 리스크 체크 통합
  - UnifiedDataProcessor에 IntegratedRiskChecker 통합
  - 각 데이터 처리 단계에서 리스크 체크 수행
  - 리스크 체크 실패 시 처리 중단 로직 구현

#### Should-have
- ⏳ 실시간 대시보드 개선
  - 웹 기반 대시보드 (Flask/FastAPI + React)
  - 실시간 차트
  - Spot/Futures 모드 표시
- ✅ 레거시 코드 제거
  - `order_manager.py`의 레거시 리스크 관리 로직 제거
  - `spot_order_manager.py`의 레거시 리스크 관리 로직 제거
  - `unified_processor.py`의 레거시 `_remove_outliers()` 제거 (아직 미완료)
- ⏳ Data Collection 리스크 체크 데이터 수집 로직 구현
  - `data_delay_ms` 실제 계산 로직 추가 (높은 우선순위)
  - `connection_status` 실제 확인 로직 추가 (높은 우선순위)
  - `orderbook` 데이터 수집 기능 추가 (낮은 우선순위, 선택적)
- ⏳ 주문 성공 여부 확인 로직 추가
  - 주문 응답 확인 후 노출 업데이트 (높은 우선순위)
  - 주문 성공 여부(`order.status == 'FILLED'`) 확인 (높은 우선순위)
  - 하이브리드 방식 구현 (주문 전 체크 + 주문 성공 후 추가) (중간 우선순위)
  - 부분 체결 처리 로직 추가 (낮은 우선순위, 선택적)

#### Nice-to-have
- ⏳ 대시보드 고급 기능
  - 실시간 알림
  - 성과 분석 차트

---

### Phase 3: Medium Priority

#### Must-have
- ⏳ 이벤트 기반 아키텍처
  - Event Bus 구현
  - 모듈 간 느슨한 결합

#### Should-have
- ⏳ 데이터 품질 관리 강화
  - 데이터 검증 파이프라인
  - 데이터 품질 메트릭
- ⏳ 백테스팅 기능 확장
  - Spot/Futures 구분
  - Walk-Forward Analysis

#### Nice-to-have
- ⏳ 고급 백테스팅 기능
  - Monte Carlo 시뮬레이션
  - 전략 최적화

---

### Phase 4: Low Priority

#### Must-have
- 없음

#### Should-have
- ⏳ 머신러닝 통합
  - 신호 예측 모델
  - 강화학습 기반 포지션 사이징

#### Nice-to-have
- ⏳ 고급 모니터링 시스템
  - ELK Stack
  - Prometheus
- ⏳ 자동화된 리포트 생성
- ⏳ 멀티 거래 타입 동시 지원

---

## 레거시 코드 관리

### 현재 레거시 코드

#### Must-have (유지 필요)
다음 코드는 핵심 기능이므로 **절대 삭제하지 마세요**:

1. **`src/execution/order_manager.py`의 주문 실행 로직**
   - 상태: 핵심 기능
   - 리스크 관리 로직만 새 모듈로 교체 예정

2. **`src/execution/spot_order_manager.py`의 주문 실행 로직**
   - 상태: 핵심 기능
   - 리스크 관리 로직만 새 모듈로 교체 예정

#### Should-have (조건부 삭제)
다음 코드는 대체 기능이 완성되면 삭제 가능:

1. **`src/execution/order_manager.py`의 리스크 관리 로직**
   - 대체: `src/risk_management/stop_loss.py`, `position_sizing.py`
   - 삭제 조건: Phase 2에서 통합 완료 후
   - 삭제 전 확인: 모든 테스트 통과, 마이그레이션 완료

2. **`src/execution/spot_order_manager.py`의 리스크 관리 로직**
   - 대체: `src/risk_management/stop_loss.py`, `position_sizing.py`
   - 삭제 조건: Phase 2에서 통합 완료 후
   - 삭제 전 확인: 모든 테스트 통과, 마이그레이션 완료

3. **`src/data_processing/unified_processor.py`의 `_remove_outliers()` 메서드**
   - 대체: `src/data_processing/preprocessor.py`의 `_remove_outliers()`
   - 삭제 조건: `DataPreprocessor` 사용 확인 후
   - 삭제 전 확인: 모든 호출 지점이 새 모듈 사용 중

#### Nice-to-have (자유 삭제)
다음 코드는 필요 없으면 삭제 가능:

- 없음 (현재)

---

## 다음 작업자를 위한 가이드

### 작업 시작 전 확인사항

1. **PRIORITIES.md 확인**
   - 현재 Phase 확인
   - Must-have 항목 우선 처리
   - 레거시 코드 삭제 전 DECISIONS.md 확인

2. **DECISIONS.md 확인**
   - 관련 의사결정 확인
   - 설계 원칙 준수
   - 레거시 코드 삭제 시 마이그레이션 계획 확인

3. **ARCHITECTURE.md 확인**
   - 모듈 구조 이해
   - 인터페이스 설계 원칙 준수
   - 통합 지점 확인

### 작업 완료 후 업데이트

1. **완료된 항목 체크**
   - PRIORITIES.md에서 ✅ 표시

2. **새로운 의사결정 기록**
   - DECISIONS.md에 추가
   - Why, 결정 과정, 우선순위 명시

3. **아키텍처 변경 기록**
   - ARCHITECTURE.md 업데이트
   - 새로운 모듈이나 인터페이스 추가 시

### 레거시 코드 삭제 시 체크리스트

- [ ] 대체 기능이 완전히 구현되었는가?
- [ ] 모든 테스트가 통과하는가?
- [ ] 마이그레이션이 완료되었는가?
- [ ] DECISIONS.md에 삭제 이유가 기록되었는가?
- [ ] 다른 모듈에서 참조하지 않는가?

---

## 우선순위 변경 이력

### [2025-11-30] Phase 1 완료
- 리스크 관리 모듈: Must-have → ✅ 완료
- 단계별 리스크 체크: Must-have → ✅ 완료
- 데이터 전처리 모듈: Must-have → ✅ 완료

### [2025-11-30] Phase 2 시작
- 피드백 루프 강화: Must-have (진행 중)
- 리스크 관리 통합: Must-have (진행 중)

