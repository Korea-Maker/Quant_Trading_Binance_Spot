# 버그 수정: PATTERN_DESCRIPTIONS float 변환 오류

**수정 일자**: 2025-12-02  
**오류 유형**: ValueError (could not convert string to float)

---

## 문제점

실시간 데이터 처리 중 다음과 같은 오류가 발생했습니다:

```
could not convert string to float: '🔼 장대양봉/음봉 - 강한 추세 지속 신호 | 🔼 띠 보유 - 강한 추세 신호 | 🔼 종가 장대봉 - 강한 추세 지속 신호 | 🔼 긴 선 - 강한 추세  신호'
```

### 원인

1. **`PATTERN_DESCRIPTIONS` 컬럼**: 패턴 설명이 문자열로 저장되는 컬럼
2. **`_prepare_risk_check_data` 메서드**: 리스크 체크 데이터 준비 시 모든 `PATTERN_`으로 시작하는 컬럼을 처리하려고 시도
3. **float 변환 시도**: `PATTERN_DESCRIPTIONS`가 `PATTERN_`으로 시작하지 않지만, 다른 곳에서 float 변환을 시도할 수 있음

---

## 수정 내용

### 1. `pattern_recognition` 단계 수정

**파일**: `src/data_processing/unified_processor.py`  
**위치**: `_prepare_risk_check_data` 메서드 (line 305-313)

**변경 전**:
```python
pattern_cols = [col for col in df.columns if col.startswith('PATTERN_')]
patterns = {}
for col in pattern_cols:
    if col in df.columns and len(df) > 0:
        val = df[col].iloc[-1]
        if pd.notna(val) and val != 0:
            patterns[col] = float(val)
```

**변경 후**:
```python
# PATTERN_DESCRIPTIONS는 문자열이므로 제외
pattern_cols = [col for col in df.columns 
              if col.startswith('PATTERN_') and col != 'PATTERN_DESCRIPTIONS']
patterns = {}
for col in pattern_cols:
    if col in df.columns and len(df) > 0:
        val = df[col].iloc[-1]
        if pd.notna(val) and val != 0:
            try:
                patterns[col] = float(val)
            except (ValueError, TypeError):
                # 문자열이나 변환 불가능한 값은 건너뛰기
                self.logger.debug(f"패턴 컬럼 {col}의 값을 float로 변환할 수 없음: {val}")
                continue
```

### 2. `indicators` 단계 안전장치 추가

**파일**: `src/data_processing/unified_processor.py`  
**위치**: `_prepare_risk_check_data` 메서드 (line 281-292)

**변경 전**:
```python
indicators = {}
for col in indicator_cols:
    if col in df.columns and len(df) > 0:
        val = df[col].iloc[-1]
        if pd.notna(val):
            if 'RSI' in col:
                indicators['RSI'] = float(val)
            elif 'MACD' in col:
                indicators['MACD'] = float(val)
            elif 'MA' in col:
                indicators['MA'] = float(val)
```

**변경 후**:
```python
indicators = {}
for col in indicator_cols:
    if col in df.columns and len(df) > 0:
        val = df[col].iloc[-1]
        if pd.notna(val):
            try:
                float_val = float(val)
                if 'RSI' in col:
                    indicators['RSI'] = float_val
                elif 'MACD' in col:
                    indicators['MACD'] = float_val
                elif 'MA' in col:
                    indicators['MA'] = float_val
            except (ValueError, TypeError):
                # 문자열이나 변환 불가능한 값은 건너뛰기
                self.logger.debug(f"지표 컬럼 {col}의 값을 float로 변환할 수 없음: {val}")
                continue
```

### 3. `signal_generation` 단계 안전장치 추가

**파일**: `src/data_processing/unified_processor.py`  
**위치**: `_prepare_risk_check_data` 메서드 (line 341-349)

**변경 전**:
```python
if 'SIGNAL_CONFIDENCE' in df.columns and len(df) > 0:
    signals['confidence'] = float(df['SIGNAL_CONFIDENCE'].iloc[-1])
else:
    signals['confidence'] = 50.0

if 'COMBINED_SIGNAL' in df.columns and len(df) > 0:
    signals['signal_strength'] = abs(float(df['COMBINED_SIGNAL'].iloc[-1]))
else:
    signals['signal_strength'] = 0.0
```

**변경 후**:
```python
if 'SIGNAL_CONFIDENCE' in df.columns and len(df) > 0:
    try:
        signals['confidence'] = float(df['SIGNAL_CONFIDENCE'].iloc[-1])
    except (ValueError, TypeError):
        signals['confidence'] = 50.0
else:
    signals['confidence'] = 50.0

if 'COMBINED_SIGNAL' in df.columns and len(df) > 0:
    try:
        signals['signal_strength'] = abs(float(df['COMBINED_SIGNAL'].iloc[-1]))
    except (ValueError, TypeError):
        signals['signal_strength'] = 0.0
else:
    signals['signal_strength'] = 0.0
```

### 4. `generate_signals` 메서드 안전장치 추가

**파일**: `src/data_processing/unified_processor.py`  
**위치**: `generate_signals` 메서드 (line 609-611)

**변경 전**:
```python
# 지표 정보
for col in processed_data.columns:
    if col in ['RSI_14', 'MACD', 'MA_20', 'MA_50', 'BB_upper_20', 'BB_lower_20', 'close']:
        signals['indicators'][col] = float(latest_data.get(col, 0))
```

**변경 후**:
```python
# 지표 정보
for col in processed_data.columns:
    if col in ['RSI_14', 'MACD', 'MA_20', 'MA_50', 'BB_upper_20', 'BB_lower_20', 'close']:
        try:
            val = latest_data.get(col, 0)
            signals['indicators'][col] = float(val)
        except (ValueError, TypeError):
            # 문자열이나 변환 불가능한 값은 기본값 사용
            self.logger.debug(f"지표 {col}의 값을 float로 변환할 수 없음: {val}")
            signals['indicators'][col] = 0.0
```

---

## 수정 효과

1. ✅ `PATTERN_DESCRIPTIONS` 컬럼이 float 변환 시도에서 제외됨
2. ✅ 모든 float 변환에 try-except 블록 추가하여 안전성 향상
3. ✅ 변환 실패 시 기본값 사용 또는 건너뛰기로 시스템 안정성 향상
4. ✅ 디버그 로그 추가로 문제 추적 용이

---

## 테스트

- ✅ 린터 오류 없음
- ⏳ 실제 실행 테스트 필요 (사용자 확인)

---

## 관련 파일

- `src/data_processing/unified_processor.py`

