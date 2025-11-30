# Agent 시작 가이드

이 문서는 각 Agent가 새로운 컨텍스트에서 작업을 시작할 때 따라야 할 가이드를 제공합니다.

## 작업 시작 전 필수 확인사항

### 1. 문서 읽기 (순서대로)

```
1. docs/CONTEXT.md          → 현재 Phase와 진행 상황
2. docs/AGENT_ROLES.md      → 자신의 역할과 책임
3. docs/AGENT_TASKS.md      → 구체적인 작업 내용
4. docs/PRIORITIES.md       → 우선순위 확인
5. docs/DECISIONS.md        → 관련 의사결정 확인
6. docs/ARCHITECTURE.md     → 모듈 구조 이해
```

### 2. 자신의 역할 확인

`docs/AGENT_ROLES.md`에서 자신이 담당하는 Agent 번호와 역할을 확인하세요.

- **Agent 1**: Risk Integration Agent
- **Agent 2**: Feedback Loop Agent
- **Agent 3**: Risk Check Integration Agent
- **Agent 4**: Dashboard Agent

### 3. 작업 범위 확인

`docs/AGENT_TASKS.md`에서 자신의 상세 작업 내용을 확인하세요.

---

## 각 Agent별 시작 지시문

### Agent 1: Risk Integration Agent

**시작 지시문:**
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

### Agent 2: Feedback Loop Agent

**시작 지시문:**
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

### Agent 3: Risk Check Integration Agent

**시작 지시문:**
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

### Agent 4: Dashboard Agent

**시작 지시문:**
```
ㄴ
```

---

## 작업 완료 체크리스트

각 Agent는 작업 완료 시 다음을 수행해야 합니다:

### 필수 작업
- [ ] `docs/CONTEXT.md` 업데이트
  - 완료 항목 체크
  - 다음 작업 명시
  
- [ ] `docs/PRIORITIES.md` 업데이트
  - 완료 항목 ✅ 표시

- [ ] 코드 린터 오류 확인 및 수정
  - `read_lints` 도구 사용

- [ ] 기존 기능 동작 확인
  - 회귀 테스트 (가능한 경우)

### 선택 작업
- [ ] `docs/DECISIONS.md` 업데이트
  - 새로운 의사결정이 있는 경우만

- [ ] `docs/ARCHITECTURE.md` 업데이트
  - 아키텍처 변경이 있는 경우만

---

## 충돌 방지 가이드

### 공유 파일 수정 시

1. **작업 전 확인**
   - 해당 파일의 최신 상태 확인
   - 다른 Agent가 동시에 작업 중인지 확인 (CONTEXT.md 확인)

2. **작업 범위 명확화**
   - 자신의 작업 범위를 명확히 정의
   - 다른 부분은 건드리지 않음

3. **문서화**
   - 수정 사항과 이유를 명확히 기록
   - DECISIONS.md에 중요한 변경사항 기록

### 의존성 있는 작업 시

1. **선행 작업 확인**
   - CONTEXT.md에서 의존하는 Agent의 작업 완료 여부 확인
   - 완료되지 않았으면 대기 또는 부분 작업 가능 여부 확인

2. **인터페이스 변경 시**
   - 공유 인터페이스 변경은 DECISIONS.md에 반드시 기록
   - 다른 Agent에게 영향이 있는지 확인

---

## 문제 발생 시

### 알려진 이슈 기록
- `docs/CONTEXT.md`의 "알려진 이슈" 섹션에 추가

### 의사결정 필요 시
- `docs/DECISIONS.md`에 기록
- 상황, 고려사항, 결정, 이유 명시

### 다른 Agent와의 협의 필요 시
- CONTEXT.md에 이슈 기록
- DECISIONS.md에 협의 사항 기록

---

## 작업 우선순위

### Must-have (필수)
- 반드시 완료해야 하는 작업
- 다른 작업보다 우선
- 완료 전 테스트 통과 필수

### Should-have (권장)
- Must-have 완료 후 진행
- 가능한 경우 테스트 추가

### Nice-to-have (선택)
- Must-have, Should-have 완료 후
- 여유가 있을 때만

---

## 참고

- **문서 위치**: 모든 문서는 `docs/` 디렉토리에 있음
- **코드 위치**: 모든 소스 코드는 `src/` 디렉토리에 있음
- **설정 파일**: `src/config/settings.py`에서 환경 변수 확인

