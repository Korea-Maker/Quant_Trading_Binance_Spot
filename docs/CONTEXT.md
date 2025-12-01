# 현재 컨텍스트 (Current Context)

이 문서는 현재 작업 중인 컨텍스트와 진행 상황을 기록합니다.
각 AI Agent가 작업을 시작할 때 이 문서를 먼저 확인하세요.

## 현재 Phase

**Phase 2: High Priority (Must-have 완료, Should-have 진행 중)**

## 최근 완료 사항

### Phase 1: Critical (완료 ✅)

1. **리스크 관리 모듈 구현**
   - `src/risk_management/stop_loss.py`: Spot/Futures 공통 손절/익절 관리
   - `src/risk_management/position_sizing.py`: Spot/Futures 포지션 사이징
   - `src/risk_management/exposure_manager.py`: 최대 노출 제한 통합 관리
   - `src/risk_management/risk_checker.py`: 단계별 리스크 체크 시스템

2. **데이터 전처리 모듈 명시화**
   - `src/data_processing/preprocessor.py`: 독립 전처리 모듈
   - `src/data_processing/unified_processor.py`: 새 전처리 모듈 사용

## 현재 작업 중인 항목

### 1. 피드백 루프 강화 (Must-have) ✅

**목표:**
- 리스크 관리 → 데이터 수집 피드백 구현
- 실시간 성과 기반 자동 조정
- Spot/Futures별 성과 추적

**진행 상황:**
- ✅ 완료 (2025-12-02)

**완료된 작업:**
- `src/monitoring/performance.py`: Spot/Futures별 성과 추적 이미 구현됨 (확인 완료)
- `src/integration/realtime_backtest_integration.py`: 
  - 성과 기반 리스크 재평가 로직 구현 (`_performance_based_risk_reassessment()`)
  - 리스크 관리 → 데이터 수집 피드백 구현 (`_adjust_data_collection_params()`)
  - 실시간 성과 기반 자동 조정 루프 구현 (`_adaptive_adjustment_loop()`)
  - 리스크 관리 모듈 파라미터 조정 (`_adjust_risk_management_params()`)
  - 거래 기록에 trading_type 추가 및 성능 모니터 연동
- `src/strategy/signals.py`: 
  - `SignalBasedStrategy`에 `adjust_parameters()` 메서드 추가
  - 성과 피드백에 따라 전략 파라미터 동적 조정 구현

### 2. 리스크 관리 모듈 통합 (Must-have) ✅

**목표:**
- 기존 OrderManager의 리스크 관리 로직을 새 모듈로 교체
- ExposureManager와 OrderManager 연동

**진행 상황:**
- ✅ 완료 (2025-11-30)
- FuturesOrderManager가 새 리스크 관리 모듈 사용
- SpotOrderManager가 새 리스크 관리 모듈 사용
- ExposureManager와 OrderManager 연동 완료
- 레거시 리스크 관리 코드 제거 완료

**완료된 작업:**
- `src/execution/order_manager.py`: FuturesOrderManager 리스크 관리 모듈 통합 완료
- `src/execution/spot_order_manager.py`: SpotOrderManager 리스크 관리 모듈 통합 완료
- `src/integration/realtime_backtest_integration.py`: ExposureManager 생성 및 OrderManager에 전달
- 레거시 리스크 관리 코드 제거 (stop_loss_percentage, take_profit_percentage 등)

### 3. 리스크 체크 통합 (Must-have) ✅

**목표:**
- UnifiedDataProcessor에 IntegratedRiskChecker 통합
- 각 데이터 처리 단계에서 리스크 체크 수행
- 리스크 체크 실패 시 처리 중단 로직 구현

**진행 상황:**
- ✅ 완료 (2025-11-30)

**완료 사항:**
- `UnifiedDataProcessor`에 `IntegratedRiskChecker` 통합 완료
- 각 처리 단계(데이터 수집, 전처리, 지표 계산, 패턴 인식, 신호 생성)에서 리스크 체크 수행
- CRITICAL 리스크 발생 시 처리 중단 로직 구현
- 리스크 체크 결과 모니터링 및 로깅 구현
- 배치 처리 및 실시간 스트리밍 처리 모두에 리스크 체크 통합

## 다음 작업자를 위한 정보

### 시작하기 전에

1. **문서 읽기 순서:**
   ```
   CONTEXT.md (이 파일) → PRIORITIES.md → DECISIONS.md → ARCHITECTURE.md
   ```

2. **현재 Phase 확인:**
   - Phase 2의 Must-have 항목 우선 처리
   - Should-have는 Must-have 완료 후

3. **레거시 코드 주의:**
   - `src/execution/order_manager.py`의 리스크 관리 로직은 레거시
   - 삭제 전 마이그레이션 필수
   - DECISIONS.md에서 삭제 조건 확인

### 작업 시 주의사항

1. **설계 원칙 준수:**
   - 공통 인터페이스 + 구현체 분리
   - 단일 책임 원칙
   - Spot/Futures 모두 지원

2. **의사결정 기록:**
   - 중요한 결정은 DECISIONS.md에 기록
   - Why, 결정 과정, 우선순위 명시

3. **코드 일관성:**
   - 기존 코드 스타일 유지
   - 타입 힌팅 사용
   - 로깅 포함

### 작업 완료 후

1. **문서 업데이트:**
   - CONTEXT.md: 완료 항목 체크, 다음 작업 명시
   - PRIORITIES.md: 완료 항목 ✅ 표시
   - DECISIONS.md: 새로운 의사결정 기록 (필요 시)

2. **테스트 확인:**
   - 기존 테스트 통과 확인
   - 새 기능 테스트 추가 (가능한 경우)

## 알려진 이슈

### Agent 3: Data Collection 리스크 체크 데이터 수집 로직 미구현

**발견 일자**: 2025-11-30  
**우선순위**: Should-have

**문제점**:
- `_prepare_risk_check_data()` 메서드의 data_collection 단계에서:
  - `connection_status`: 항상 'connected'로 하드코딩 (실제 연결 상태 미확인)
  - `data_delay_ms`: 항상 0으로 하드코딩 (실제 지연 시간 미계산)
  - `orderbook`: 빈 딕셔너리 (오더북 데이터 미수집)

**영향**:
- 데이터 수집 단계의 리스크 체크가 제한적으로 작동
- 연결 끊김, 데이터 지연, 유동성 리스크를 정확히 감지하지 못함

**해결 방안**:
1. `data_delay_ms`: DataFrame의 마지막 타임스탬프와 현재 시간 비교하여 계산
2. `connection_status`: 웹소켓 클라이언트의 `connected` 속성 확인
3. `orderbook`: 웹소켓 클라이언트에 오더북 구독 기능 추가 (선택적)

**상태**: 향후 개선 사항으로 기록됨 (Phase 2 Must-have 완료 후 처리 예정)

### Agent 1: 주문 성공 여부 확인 로직 미구현

**발견 일자**: 2025-11-30  
**우선순위**: Should-have

**문제점**:
- `open_position()` 메서드에서 `ExposureManager.add_position()`이 주문 전에 호출됨
- 주문 성공 여부를 명확히 확인하지 않음
- 주문 응답 상태를 확인하지 않음
- 원래 계획(AGENT_TASKS.md)에는 "주문 성공 후 노출 업데이트"가 포함되어 있었으나 완전히 구현되지 않음

**영향**:
- 주문이 실제로 성공하지 않았는데 노출이 추가될 수 있음 (롤백 로직은 존재)
- 부분 체결이나 다른 주문 상태를 처리하지 않음
- 노출 관리의 정확도가 떨어질 수 있음

**해결 방안**:
1. 주문 응답 확인 후 노출 업데이트 로직 추가
2. 주문 성공 여부(`order.status == 'FILLED'`) 확인
3. 하이브리드 방식 고려 (주문 전 `can_open_new_position()` 체크 + 주문 성공 후 `add_position()` 호출)
4. 부분 체결 처리 로직 추가 (선택적)

**상태**: 향후 개선 사항으로 기록됨 (Phase 2 Must-have 완료 후 처리 예정)

## 향후 계획

### Phase 2 Should-have (진행 예정)

1. Data Collection 리스크 체크 데이터 수집 로직 구현
2. 주문 성공 여부 확인 로직 추가
3. Agent 4: 실시간 대시보드 개선
4. 레거시 코드 정리

### Phase 2 완료 후 (Phase 3)

1. 이벤트 기반 아키텍처 도입
2. 데이터 품질 관리 강화
3. 백테스팅 기능 확장

### Phase 3 완료 후 (Phase 4)

1. 머신러닝 통합
2. 고급 모니터링 시스템
3. 자동화된 리포트 생성

## To Do List

**상세한 작업 목록**: `docs/TODO_LIST.md` 참고

## 참고 자료

- **의사결정 기록**: `docs/DECISIONS.md`
- **아키텍처 문서**: `docs/ARCHITECTURE.md`
- **우선순위 관리**: `docs/PRIORITIES.md`
- **Agent 역할 분담**: `docs/AGENT_ROLES.md`
- **Agent 작업 명세**: `docs/AGENT_TASKS.md`
- **Agent 시작 가이드**: `docs/AGENT_START_GUIDE.md` ⭐ **새 컨텍스트 시작 시 필수**
- **프로젝트 README**: `README.md`

## 작업 히스토리

### [2025-11-30] Agent 2 작업 완료
- 피드백 루프 강화 완료
- 성과 기반 리스크 재평가 로직 구현
- 리스크 관리 → 데이터 수집 피드백 구현
- 실시간 성과 기반 자동 조정 루프 구현
- 전략 파라미터 동적 조정 기능 추가

### [2025-11-30] Agent 3 작업 완료
- UnifiedDataProcessor에 IntegratedRiskChecker 통합 완료
- 각 데이터 처리 단계에서 리스크 체크 수행 구현
- 리스크 체크 실패 시 처리 중단 로직 구현
- 배치 및 실시간 처리 모두에 리스크 체크 통합

### [2025-11-30] Phase 1 완료
- 리스크 관리 모듈 구현 완료
- 단계별 리스크 체크 시스템 구현 완료
- 데이터 전처리 모듈 명시화 완료
- Phase 2 시작

### [2025-11-30] 문서화 프레임워크 구축
- DECISIONS.md 생성
- ARCHITECTURE.md 생성
- PRIORITIES.md 생성
- CONTEXT.md 생성

