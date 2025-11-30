# AI Agent 역할 분담 (Agent Role Assignment)

이 문서는 각 AI Agent의 역할과 책임을 정의합니다.
각 Agent는 특정 모듈이나 기능에 집중하여 독립적으로 작업할 수 있습니다.

## Agent 생성 원칙

1. **단일 책임**: 각 Agent는 하나의 명확한 목표에 집중
2. **독립성**: 다른 Agent의 작업에 최소한의 의존성
3. **명확한 범위**: 작업 범위와 제약사항이 명확히 정의됨
4. **문서 기반 협업**: CONTEXT.md, DECISIONS.md, ARCHITECTURE.md를 통해 협업

---

## Phase 2: High Priority - Agent 역할 분담

### Agent 1: Risk Integration Agent (리스크 관리 통합 전문가)

**역할**: 리스크 관리 모듈과 OrderManager 통합

**책임 범위:**
- `src/execution/order_manager.py`의 리스크 관리 로직을 새 모듈로 교체
- `src/execution/spot_order_manager.py`의 리스크 관리 로직을 새 모듈로 교체
- `ExposureManager`와 `FuturesOrderManager`/`SpotOrderManager` 연동
- 레거시 리스크 관리 코드 제거 (통합 완료 후)

**작업 파일:**
- `src/execution/order_manager.py` (수정)
- `src/execution/spot_order_manager.py` (수정)
- `src/integration/realtime_backtest_integration.py` (통합)
- `src/risk_management/` 모듈 (사용)

**주의사항:**
- 주문 실행 로직은 절대 삭제하지 않음 (핵심 기능)
- 리스크 관리 로직만 새 모듈로 교체
- Spot과 Futures 모두 지원해야 함
- 기존 동작과 호환성 유지

**완료 조건:**
- [ ] FuturesOrderManager가 새 리스크 관리 모듈 사용
- [ ] SpotOrderManager가 새 리스크 관리 모듈 사용
- [ ] ExposureManager와 OrderManager 연동 완료
- [ ] 레거시 리스크 관리 코드 제거 (Should-have)
- [ ] 모든 테스트 통과

**참고 문서:**
- `docs/ARCHITECTURE.md`: 통합 지점 확인
- `docs/DECISIONS.md`: 레거시 코드 삭제 조건
- `src/risk_management/`: 사용할 모듈

---

### Agent 2: Feedback Loop Agent (피드백 루프 전문가)

**역할**: 피드백 루프 강화 및 실시간 성과 기반 자동 조정

**책임 범위:**
- 리스크 관리 → 데이터 수집 피드백 구현
- 실시간 성과 기반 자동 조정 로직 구현
- Spot/Futures별 성과 추적 기능 추가
- 성과 기록 → 리스크 재평가 → 전략 조정 파이프라인 구현

**작업 파일:**
- `src/integration/realtime_backtest_integration.py` (수정)
- `src/monitoring/performance.py` (확장)
- `src/risk_management/` (피드백 연동)
- `src/data_processing/unified_processor.py` (피드백 적용)

**주의사항:**
- 기존 백테스팅 피드백 로직과 충돌하지 않도록
- Spot과 Futures 성과를 구분하여 추적
- 실시간 성과 기반 조정은 점진적으로 적용

**완료 조건:**
- [ ] 리스크 관리 → 데이터 수집 피드백 구현
- [ ] 실시간 성과 기반 자동 조정 로직 구현
- [ ] Spot/Futures별 성과 추적 기능 추가
- [ ] 성과 기록 → 리스크 재평가 → 전략 조정 파이프라인 완성

**참고 문서:**
- `docs/ARCHITECTURE.md`: 피드백 루프 구조
- `src/integration/realtime_backtest_integration.py`: 기존 피드백 로직

---

### Agent 3: Risk Check Integration Agent (리스크 체크 통합 전문가)

**역할**: 단계별 리스크 체크를 데이터 처리 파이프라인에 통합

**책임 범위:**
- `UnifiedDataProcessor`에 `IntegratedRiskChecker` 통합
- 각 데이터 처리 단계에서 리스크 체크 수행
- 리스크 체크 실패 시 처리 중단 로직 구현
- 리스크 체크 결과를 신호 생성에 반영

**작업 파일:**
- `src/data_processing/unified_processor.py` (수정)
- `src/risk_management/risk_checker.py` (사용)
- `src/data_processing/preprocessor.py` (리스크 체크 데이터 제공)

**주의사항:**
- 기존 데이터 처리 로직과 충돌하지 않도록
- 리스크 체크 실패 시 적절한 처리 (중단 또는 경고)
- 성능 영향 최소화

**완료 조건:**
- [ ] UnifiedDataProcessor에 IntegratedRiskChecker 통합
- [ ] 각 처리 단계에서 리스크 체크 수행
- [ ] 리스크 체크 실패 시 처리 중단 로직 구현
- [ ] 리스크 체크 결과 로깅 및 모니터링

**참고 문서:**
- `docs/ARCHITECTURE.md`: 데이터 흐름 및 리스크 체크 위치
- `src/risk_management/risk_checker.py`: 사용할 체커

---

### Agent 4: Dashboard Agent (대시보드 개발 전문가) - Should-have

**역할**: 웹 기반 실시간 대시보드 개발

**책임 범위:**
- Flask/FastAPI 기반 백엔드 API 구현
- React 기반 프론트엔드 대시보드 구현
- 실시간 차트 (Candlestick, 지표, 신호)
- 포지션 모니터링 (Spot: 보유량, Futures: 레버리지, 마진)
- Spot/Futures 모드 표시

**작업 파일:**
- `src/monitoring/dashboard.py` (확장 또는 새 모듈)
- 새 디렉토리: `src/monitoring/web_dashboard/` (생성)
- 새 디렉토리: `src/monitoring/api/` (생성, 선택적)

**주의사항:**
- 기존 콘솔 대시보드와 병행 운영 가능하도록
- 실시간 데이터 스트리밍 (WebSocket 또는 SSE)
- 보안 고려 (API 키 등 민감 정보 노출 방지)

**완료 조건:**
- [ ] 웹 기반 대시보드 백엔드 구현
- [ ] 웹 기반 대시보드 프론트엔드 구현
- [ ] 실시간 차트 구현
- [ ] 포지션 모니터링 구현
- [ ] Spot/Futures 모드 표시

**참고 문서:**
- `src/monitoring/dashboard.py`: 기존 대시보드 구조
- `src/monitoring/performance.py`: 성능 데이터 구조

---

## 작업 순서 및 의존성

### Phase 2 Must-have 작업 순서

1. **Agent 3 (Risk Check Integration)** - 최우선
   - 이유: 데이터 처리 파이프라인의 기반이 되는 리스크 체크 통합
   - 의존성: 없음 (이미 구현된 RiskChecker 사용)

2. **Agent 1 (Risk Integration)** - 병렬 가능
   - 이유: OrderManager와 리스크 관리 모듈 통합
   - 의존성: 없음 (이미 구현된 리스크 관리 모듈 사용)

3. **Agent 2 (Feedback Loop)** - Agent 1 완료 후 권장
   - 이유: 리스크 관리 모듈 통합 후 피드백 루프 구현이 더 안정적
   - 의존성: Agent 1 완료 권장 (필수는 아님)

### Phase 2 Should-have 작업

4. **Agent 4 (Dashboard)** - 독립적
   - 이유: Must-have 완료 후 진행 가능
   - 의존성: 없음 (독립적 작업)

---

## Agent 작업 시작 체크리스트

각 Agent는 작업 시작 전 다음을 확인해야 합니다:

- [ ] `docs/CONTEXT.md` 읽기 완료
- [ ] `docs/PRIORITIES.md`에서 자신의 작업 우선순위 확인
- [ ] `docs/ARCHITECTURE.md`에서 관련 모듈 구조 이해
- [ ] `docs/DECISIONS.md`에서 관련 의사결정 확인
- [ ] 자신의 작업 범위와 제약사항 명확히 이해
- [ ] 레거시 코드 삭제 시 DECISIONS.md의 체크리스트 확인

---

## Agent 작업 완료 체크리스트

각 Agent는 작업 완료 시 다음을 수행해야 합니다:

- [ ] `docs/CONTEXT.md` 업데이트 (완료 항목 체크)
- [ ] `docs/PRIORITIES.md` 업데이트 (✅ 표시)
- [ ] `docs/DECISIONS.md` 업데이트 (새로운 의사결정 기록, 필요 시)
- [ ] `docs/ARCHITECTURE.md` 업데이트 (아키텍처 변경 시)
- [ ] 코드 린터 오류 확인 및 수정
- [ ] 기존 기능 동작 확인 (회귀 테스트)

---

## Agent 간 협업 가이드

### 공유 파일 수정 시

1. **충돌 방지**: 작업 전 해당 파일의 최신 상태 확인
2. **명확한 범위**: 자신의 작업 범위를 명확히 정의
3. **문서화**: 수정 사항과 이유를 명확히 기록

### 의존성 있는 작업 시

1. **선행 작업 확인**: 의존하는 Agent의 작업 완료 여부 확인
2. **인터페이스 확인**: 공유 인터페이스 변경 시 DECISIONS.md에 기록
3. **통합 테스트**: 여러 모듈이 연동되는 경우 통합 테스트 수행

---

## 다음 Phase 예상 Agent

### Phase 3: Medium Priority

- **Event Architecture Agent**: 이벤트 기반 아키텍처 구현
- **Data Quality Agent**: 데이터 품질 관리 강화
- **Backtesting Enhancement Agent**: 백테스팅 기능 확장

### Phase 4: Low Priority

- **ML Integration Agent**: 머신러닝 통합
- **Monitoring System Agent**: 고급 모니터링 시스템
- **Reporting Agent**: 자동화된 리포트 생성

---

## 현재 권장 작업 순서

1. **Agent 3 (Risk Check Integration)** - 즉시 시작 가능
2. **Agent 1 (Risk Integration)** - 즉시 시작 가능 (Agent 3과 병렬)
3. **Agent 2 (Feedback Loop)** - Agent 1 완료 후 시작 권장
4. **Agent 4 (Dashboard)** - Must-have 완료 후 시작

