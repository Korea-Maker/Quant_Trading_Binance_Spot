# PM 상태 보고서

이 문서는 프로젝트의 전체 상태와 각 Agent의 작업 진행 상황을 관리합니다.

## 현재 Phase

**Phase 2: High Priority (진행 중)**

## 전체 진행 상황

### Phase 1: Critical (완료 ✅)

- ✅ 리스크 관리 모듈 구현
- ✅ 단계별 리스크 체크 시스템
- ✅ 데이터 전처리 모듈 명시화

### Phase 2: High Priority (진행 중)

#### Must-have
- ⏳ **Agent 1**: 리스크 관리 모듈 통합 (대기 중)
- ⏳ **Agent 2**: 피드백 루프 강화 (대기 중)
- ⏳ **Agent 3**: 리스크 체크 통합 (대기 중)

#### Should-have
- ⏳ **Agent 4**: 실시간 대시보드 개선 (대기 중)

---

## Agent별 작업 상태

### Agent 1: Risk Integration Agent

**상태**: 대기 중  
**우선순위**: Must-have  
**작업 파일**: 
- `src/execution/order_manager.py`
- `src/execution/spot_order_manager.py`
- `src/integration/realtime_backtest_integration.py`

**의존성**: 없음 (즉시 시작 가능)

**시작 지시문**:
```
docs/AGENT_ROLES.md와 docs/AGENT_TASKS.md를 참고하여 Agent 1 (Risk Integration Agent) 작업을 수행하세요.

작업 목표:
- 기존 OrderManager의 리스크 관리 로직을 새로 구현된 리스크 관리 모듈로 교체
- ExposureManager와 OrderManager 연동

주의사항:
- 주문 실행 로직은 절대 삭제하지 않음
- Spot과 Futures 모두 지원해야 함
- 기존 동작과 호환성 유지 필수
```

---

### Agent 2: Feedback Loop Agent

**상태**: 대기 중  
**우선순위**: Must-have  
**작업 파일**:
- `src/integration/realtime_backtest_integration.py`
- `src/monitoring/performance.py`
- `src/strategy/signals.py`

**의존성**: Agent 1 완료 권장 (필수는 아님)

**시작 지시문**:
```
docs/AGENT_ROLES.md와 docs/AGENT_TASKS.md를 참고하여 Agent 2 (Feedback Loop Agent) 작업을 수행하세요.

작업 목표:
- 리스크 관리 → 데이터 수집 피드백 구현
- 실시간 성과 기반 자동 조정
- Spot/Futures별 성과 추적

주의사항:
- 기존 백테스팅 피드백 로직과 충돌하지 않도록
- Spot과 Futures 성과를 구분하여 추적
```

---

### Agent 3: Risk Check Integration Agent

**상태**: 대기 중  
**우선순위**: Must-have (최우선)  
**작업 파일**:
- `src/data_processing/unified_processor.py`
- `src/data_processing/preprocessor.py`

**의존성**: 없음 (즉시 시작 가능, 최우선 권장)

**시작 지시문**:
```
docs/AGENT_ROLES.md와 docs/AGENT_TASKS.md를 참고하여 Agent 3 (Risk Check Integration Agent) 작업을 수행하세요.

작업 목표:
- UnifiedDataProcessor에 IntegratedRiskChecker 통합
- 각 데이터 처리 단계에서 리스크 체크 수행
- 리스크 체크 실패 시 처리 중단 로직 구현

주의사항:
- 기존 데이터 처리 로직과 충돌하지 않도록
- 성능 영향 최소화
```

---

### Agent 4: Dashboard Agent

**상태**: 대기 중  
**우선순위**: Should-have  
**작업 파일**: 새 모듈 생성
- `src/monitoring/web_dashboard/` (생성)

**의존성**: 없음 (독립적, Must-have 완료 후 권장)

**시작 지시문**:
```
docs/AGENT_ROLES.md와 docs/AGENT_TASKS.md를 참고하여 Agent 4 (Dashboard Agent) 작업을 수행하세요.

작업 목표:
- 웹 기반 실시간 대시보드 구현 (Flask/FastAPI + React)
- 실시간 차트 및 포지션 모니터링
- Spot/Futures 모드 표시

주의사항:
- 기존 콘솔 대시보드와 병행 운영 가능하도록
- 보안 고려 (API 키 등 민감 정보 노출 방지)
```

---

## 권장 작업 순서

### 즉시 시작 가능 (병렬 작업 권장)

1. **Agent 3** (Risk Check Integration) - 최우선
   - 이유: 데이터 처리 파이프라인의 기반
   - 독립성: 높음

2. **Agent 1** (Risk Integration) - 병렬 가능
   - 이유: OrderManager 통합
   - 독립성: 높음

### 순차 작업 권장

3. **Agent 2** (Feedback Loop) - Agent 1 완료 후 권장
   - 이유: 리스크 관리 통합 후 피드백 구현이 더 안정적
   - 독립성: 중간

### Must-have 완료 후

4. **Agent 4** (Dashboard) - 독립적
   - 이유: Should-have이므로 Must-have 완료 후
   - 독립성: 매우 높음

---

## 작업 완료 시 업데이트

각 Agent가 작업을 완료하면 다음을 업데이트하세요:

1. **이 문서 (PM_STATUS.md)**
   - Agent 상태를 "진행 중" → "완료"로 변경
   - 완료 날짜 기록

2. **CONTEXT.md**
   - 완료 항목 체크
   - 다음 작업 명시

3. **PRIORITIES.md**
   - 완료 항목 ✅ 표시

---

## 알려진 이슈

### 현재 없음

---

## 다음 단계

Phase 2 Must-have 완료 후:
- Phase 3 작업 계획 수립
- 새로운 Agent 역할 정의

