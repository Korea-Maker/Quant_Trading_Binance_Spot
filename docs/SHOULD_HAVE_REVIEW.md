# Should-have 개선 사항 검토

**검토 일자**: 2025-12-02  
**Phase**: Phase 2 Should-have

---

## 📋 Should-have 개선 사항 목록

### 1. Data Collection 리스크 체크 데이터 수집 로직 구현

**우선순위**: 높은 우선순위  
**상태**: ⏳ 대기 중  
**발견 일자**: 2025-11-30

#### 문제점
- `_prepare_risk_check_data()` 메서드의 data_collection 단계에서:
  - `connection_status`: 항상 'connected'로 하드코딩 (실제 연결 상태 미확인)
  - `data_delay_ms`: 항상 0으로 하드코딩 (실제 지연 시간 미계산)
  - `orderbook`: 빈 딕셔너리 (오더북 데이터 미수집)

#### 영향
- 데이터 수집 단계의 리스크 체크가 제한적으로 작동
- 연결 끊김, 데이터 지연, 유동성 리스크를 정확히 감지하지 못함

#### 해결 방안
1. **`data_delay_ms` 실제 계산 로직 추가** (높은 우선순위)
   - DataFrame의 마지막 타임스탬프와 현재 시간 비교하여 계산
   - 파일: `src/data_processing/unified_processor.py`
   - 메서드: `_prepare_risk_check_data()`

2. **`connection_status` 실제 확인 로직 추가** (높은 우선순위)
   - 웹소켓 클라이언트의 `connected` 속성 확인
   - 파일: `src/data_processing/unified_processor.py`
   - 메서드: `_prepare_risk_check_data()`

3. **`orderbook` 데이터 수집 기능 추가** (낮은 우선순위, 선택적)
   - 웹소켓 클라이언트에 오더북 구독 기능 추가
   - 유동성 리스크 체크에 활용

#### 구현 위치
- 파일: `src/data_processing/unified_processor.py`
- 메서드: `_prepare_risk_check_data()` (data_collection 단계)

#### 예상 작업량
- `data_delay_ms`: 중간 (타임스탬프 비교 로직)
- `connection_status`: 낮음 (속성 확인)
- `orderbook`: 높음 (웹소켓 구독 추가)

---

### 2. 주문 성공 여부 확인 로직 추가

**우선순위**: 높은 우선순위  
**상태**: ⏳ 대기 중  
**발견 일자**: 2025-11-30

#### 문제점
- `open_position()` 메서드에서 `ExposureManager.add_position()`이 주문 전에 호출됨
- 주문 성공 여부를 명확히 확인하지 않음
- 주문 응답 상태를 확인하지 않음
- 원래 계획(AGENT_TASKS.md)에는 "주문 성공 후 노출 업데이트"가 포함되어 있었으나 완전히 구현되지 않음

#### 영향
- 주문이 실제로 성공하지 않았는데 노출이 추가될 수 있음 (롤백 로직은 존재)
- 부분 체결이나 다른 주문 상태를 처리하지 않음
- 노출 관리의 정확도가 떨어질 수 있음

#### 해결 방안
1. **주문 응답 확인 후 노출 업데이트** (높은 우선순위)
   - 주문 응답을 받은 후 `order.status == 'FILLED'` 확인
   - 성공 시에만 `add_position()` 호출
   - 파일: `src/execution/order_manager.py`, `src/execution/spot_order_manager.py`

2. **주문 성공 여부(`order.status == 'FILLED'`) 확인** (높은 우선순위)
   - 주문 응답에서 상태 확인
   - FILLED가 아닌 경우 노출 추가하지 않음

3. **하이브리드 방식 구현** (중간 우선순위)
   - 주문 전 `can_open_new_position()` 체크 (현재 구현됨)
   - 주문 성공 후 `add_position()` 호출 (추가 필요)

4. **부분 체결 처리 로직 추가** (낮은 우선순위, 선택적)
   - 부분 체결 시 부분 노출만 추가
   - 나머지 체결 대기

#### 구현 위치
- 파일: `src/execution/order_manager.py` (FuturesOrderManager)
- 파일: `src/execution/spot_order_manager.py` (SpotOrderManager)
- 메서드: `open_position()`, `place_market_buy_order()`

#### 예상 작업량
- 주문 응답 확인: 낮음 (응답 상태 확인 로직)
- 하이브리드 방식: 중간 (로직 재구성)
- 부분 체결 처리: 높음 (복잡한 로직)

---

## 📊 우선순위별 정리

### 높은 우선순위 (즉시 구현 권장)
1. **Data Collection 리스크 체크 - `data_delay_ms` 계산**
   - 작업량: 중간
   - 영향: 데이터 지연 리스크 감지 개선
   - 구현 난이도: 중간

2. **Data Collection 리스크 체크 - `connection_status` 확인**
   - 작업량: 낮음
   - 영향: 연결 상태 리스크 감지 개선
   - 구현 난이도: 낮음

3. **주문 성공 여부 확인 - 주문 응답 확인 후 노출 업데이트**
   - 작업량: 낮음
   - 영향: 노출 관리 정확도 개선
   - 구현 난이도: 낮음

4. **주문 성공 여부 확인 - `order.status == 'FILLED'` 확인**
   - 작업량: 낮음
   - 영향: 노출 관리 정확도 개선
   - 구현 난이도: 낮음

### 중간 우선순위 (구현 권장)
5. **주문 성공 여부 확인 - 하이브리드 방식 구현**
   - 작업량: 중간
   - 영향: 노출 관리 정확도 및 안정성 개선
   - 구현 난이도: 중간

### 낮은 우선순위 (선택적)
6. **Data Collection 리스크 체크 - `orderbook` 데이터 수집**
   - 작업량: 높음
   - 영향: 유동성 리스크 감지 개선
   - 구현 난이도: 높음

7. **주문 성공 여부 확인 - 부분 체결 처리**
   - 작업량: 높음
   - 영향: 부분 체결 시 노출 관리 정확도 개선
   - 구현 난이도: 높음

---

## 🎯 권장 구현 순서

### 1단계: 높은 우선순위 항목 (즉시 구현)
1. 주문 성공 여부 확인 - 주문 응답 확인 후 노출 업데이트
2. 주문 성공 여부 확인 - `order.status == 'FILLED'` 확인
3. Data Collection 리스크 체크 - `connection_status` 확인
4. Data Collection 리스크 체크 - `data_delay_ms` 계산

### 2단계: 중간 우선순위 항목
5. 주문 성공 여부 확인 - 하이브리드 방식 구현

### 3단계: 낮은 우선순위 항목 (선택적)
6. Data Collection 리스크 체크 - `orderbook` 데이터 수집
7. 주문 성공 여부 확인 - 부분 체결 처리

---

## 📝 구현 시 주의사항

### Data Collection 리스크 체크
- 기존 리스크 체크 로직과 충돌하지 않도록
- 성능 영향 최소화
- 웹소켓 클라이언트의 상태 확인 방법 확인 필요

### 주문 성공 여부 확인
- 기존 롤백 로직과 충돌하지 않도록
- 주문 실패 시 적절한 에러 처리
- Spot과 Futures 모두에서 동일하게 작동하도록

---

## ✅ 검토 완료

- 모든 Should-have 개선 사항 확인 완료
- 우선순위 분류 완료
- 구현 방안 제시 완료
- 구현 순서 제안 완료

