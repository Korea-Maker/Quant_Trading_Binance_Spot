# 사용자 가이드: Agent 컨텍스트 생성 및 관리

이 문서는 사용자가 각 Agent를 위한 컨텍스트를 생성하고 관리하는 방법을 안내합니다.

## 📋 작업 순서 개요

### 즉시 시작 가능 (병렬 작업 권장)

1. **Agent 3** (Risk Check Integration) - 최우선 ⭐
2. **Agent 1** (Risk Integration) - 병렬 가능

### 순차 작업 권장

3. **Agent 2** (Feedback Loop) - Agent 1 완료 후 권장

### Must-have 완료 후

4. **Agent 4** (Dashboard) - 독립적, Should-have

---

## 🚀 각 Agent 컨텍스트 생성 방법

### Agent 3: Risk Check Integration Agent (최우선)

**1. 새 컨텍스트 생성**
- Cursor에서 새 채팅/컨텍스트 시작

**2. 시작 지시문 복사하여 붙여넣기:**
```
docs/AGENT_ROLES.md와 docs/AGENT_TASKS.md를 참고하여 Agent 3 (Risk Check Integration Agent) 작업을 수행하세요.

작업 목표:
- UnifiedDataProcessor에 IntegratedRiskChecker 통합
- 각 데이터 처리 단계에서 리스크 체크 수행
- 리스크 체크 실패 시 처리 중단 로직 구현

주의사항:
- 기존 데이터 처리 로직과 충돌하지 않도록
- 성능 영향 최소화

작업 완료 후:
- docs/CONTEXT.md 업데이트
- docs/PRIORITIES.md 업데이트
- docs/PM_STATUS.md 업데이트 (상태를 "완료"로 변경)
```

**3. Agent가 작업 시작**

---

### Agent 1: Risk Integration Agent

**1. 새 컨텍스트 생성**
- Cursor에서 새 채팅/컨텍스트 시작

**2. 시작 지시문 복사하여 붙여넣기:**
```
docs/AGENT_ROLES.md와 docs/AGENT_TASKS.md를 참고하여 Agent 1 (Risk Integration Agent) 작업을 수행하세요.

작업 목표:
- 기존 OrderManager의 리스크 관리 로직을 새로 구현된 리스크 관리 모듈로 교체
- ExposureManager와 OrderManager 연동

주의사항:
- 주문 실행 로직은 절대 삭제하지 않음
- Spot과 Futures 모두 지원해야 함
- 기존 동작과 호환성 유지 필수

작업 완료 후:
- docs/CONTEXT.md 업데이트
- docs/PRIORITIES.md 업데이트
- docs/PM_STATUS.md 업데이트 (상태를 "완료"로 변경)
```

**3. Agent가 작업 시작**

---

### Agent 2: Feedback Loop Agent

**1. 새 컨텍스트 생성**
- Cursor에서 새 채팅/컨텍스트 시작

**2. 시작 지시문 복사하여 붙여넣기:**
```
docs/AGENT_ROLES.md와 docs/AGENT_TASKS.md를 참고하여 Agent 2 (Feedback Loop Agent) 작업을 수행하세요.

작업 목표:
- 리스크 관리 → 데이터 수집 피드백 구현
- 실시간 성과 기반 자동 조정
- Spot/Futures별 성과 추적

주의사항:
- 기존 백테스팅 피드백 로직과 충돌하지 않도록
- Spot과 Futures 성과를 구분하여 추적

작업 완료 후:
- docs/CONTEXT.md 업데이트
- docs/PRIORITIES.md 업데이트
- docs/PM_STATUS.md 업데이트 (상태를 "완료"로 변경)
```

**3. Agent가 작업 시작**

---

### Agent 4: Dashboard Agent (Should-have)

**1. 새 컨텍스트 생성**
- Cursor에서 새 채팅/컨텍스트 시작

**2. 시작 지시문 복사하여 붙여넣기:**
```
docs/AGENT_ROLES.md와 docs/AGENT_TASKS.md를 참고하여 Agent 4 (Dashboard Agent) 작업을 수행하세요.

작업 목표:
- 웹 기반 실시간 대시보드 구현 (Flask/FastAPI + React)
- 실시간 차트 및 포지션 모니터링
- Spot/Futures 모드 표시

주의사항:
- 기존 콘솔 대시보드와 병행 운영 가능하도록
- 보안 고려 (API 키 등 민감 정보 노출 방지)

작업 완료 후:
- docs/CONTEXT.md 업데이트
- docs/PRIORITIES.md 업데이트
- docs/PM_STATUS.md 업데이트 (상태를 "완료"로 변경)
```

**3. Agent가 작업 시작**

---

## ✅ 작업 완료 확인 방법

각 Agent가 작업을 완료하면 다음을 확인하세요:

### 1. 문서 업데이트 확인
- `docs/CONTEXT.md`: 완료 항목이 체크되었는지
- `docs/PRIORITIES.md`: 완료 항목에 ✅ 표시되었는지
- `docs/PM_STATUS.md`: Agent 상태가 "완료"로 변경되었는지

### 2. 코드 변경 확인
- 해당 Agent의 작업 파일들이 수정되었는지
- 린터 오류가 없는지

### 3. 다음 Agent 시작
- 의존성이 있는 경우 선행 Agent 완료 확인
- 다음 Agent 컨텍스트 생성

---

## 📊 진행 상황 추적

### PM_STATUS.md 확인
각 Agent의 상태를 `docs/PM_STATUS.md`에서 확인할 수 있습니다:
- **대기 중**: 아직 시작하지 않음
- **진행 중**: 작업 중
- **완료**: 작업 완료

### CONTEXT.md 확인
전체 프로젝트 진행 상황은 `docs/CONTEXT.md`에서 확인할 수 있습니다.

---

## ⚠️ 주의사항

### 병렬 작업 시
- Agent 1과 Agent 3은 병렬로 작업 가능
- 같은 파일을 수정하지 않도록 주의
- 충돌 발생 시 CONTEXT.md에 이슈 기록

### 순차 작업 시
- Agent 2는 Agent 1 완료 후 시작 권장
- Agent 1의 작업 결과를 확인한 후 시작

### 파일 충돌 시
- CONTEXT.md에서 다른 Agent의 작업 상태 확인
- DECISIONS.md에 충돌 해결 방법 기록

---

## 🔄 작업 흐름도

```
시작
  ↓
Agent 3 컨텍스트 생성 (최우선)
  ↓
Agent 1 컨텍스트 생성 (병렬)
  ↓
Agent 3 완료 확인
  ↓
Agent 1 완료 확인
  ↓
Agent 2 컨텍스트 생성
  ↓
Agent 2 완료 확인
  ↓
Phase 2 Must-have 완료 ✅
  ↓
Agent 4 컨텍스트 생성 (Should-have)
  ↓
Agent 4 완료 확인
  ↓
Phase 2 완료 ✅
```

---

## 📝 체크리스트

### 각 Agent 시작 전
- [ ] 해당 Agent의 문서 확인 (AGENT_ROLES.md, AGENT_TASKS.md)
- [ ] 의존성 확인 (선행 Agent 완료 여부)
- [ ] 새 컨텍스트 생성
- [ ] 시작 지시문 붙여넣기

### 각 Agent 완료 후
- [ ] 문서 업데이트 확인
- [ ] 코드 변경 확인
- [ ] PM_STATUS.md 업데이트 확인
- [ ] 다음 Agent 시작 준비

---

## 🆘 문제 발생 시

### Agent가 작업을 완료하지 못한 경우
1. CONTEXT.md의 "알려진 이슈" 섹션 확인
2. DECISIONS.md에서 관련 의사결정 확인
3. 필요 시 PM에게 문의 (새 컨텍스트에서)

### 파일 충돌 발생 시
1. git status로 충돌 파일 확인
2. CONTEXT.md에서 다른 Agent의 작업 상태 확인
3. DECISIONS.md에 충돌 해결 방법 기록
4. 필요 시 수동으로 병합

---

## 📚 참고 문서

- **Agent 역할**: `docs/AGENT_ROLES.md`
- **Agent 작업 명세**: `docs/AGENT_TASKS.md`
- **Agent 시작 가이드**: `docs/AGENT_START_GUIDE.md`
- **프로젝트 상태**: `docs/PM_STATUS.md`
- **현재 컨텍스트**: `docs/CONTEXT.md`
- **우선순위**: `docs/PRIORITIES.md`

---

## 💡 팁

1. **병렬 작업**: Agent 1과 Agent 3은 동시에 진행 가능하므로 시간 절약
2. **문서 확인**: 각 Agent는 자동으로 문서를 확인하지만, 문제 발생 시 수동 확인
3. **상태 추적**: PM_STATUS.md를 주기적으로 확인하여 진행 상황 파악
4. **커밋**: 각 Agent 작업 완료 후 Git 커밋 권장 (선택사항)

