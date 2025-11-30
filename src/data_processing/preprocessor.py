"""
데이터 전처리 모듈

이 모듈은 시장 데이터의 전처리를 담당합니다.
결측치 처리, 이상치 제거, 데이터 정규화 등을 수행합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime

from src.utils.logger import get_logger


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(
        self,
        outlier_std_threshold: float = 3.0,
        missing_data_threshold: float = 0.1,  # 10% 이상 결측 시 경고
        enable_normalization: bool = False
    ):
        """
        Args:
            outlier_std_threshold: 이상치 감지 표준편차 임계값
            missing_data_threshold: 결측치 경고 임계값
            enable_normalization: 정규화 활성화 여부
        """
        self.logger = get_logger(__name__)
        self.outlier_std_threshold = outlier_std_threshold
        self.missing_data_threshold = missing_data_threshold
        self.enable_normalization = enable_normalization
        
        # 전처리 통계
        self.preprocessing_stats: Dict[str, any] = {}
    
    def preprocess(
        self,
        df: pd.DataFrame,
        validate: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        데이터 전처리 메인 함수
        
        Args:
            df: 원본 DataFrame
            validate: 데이터 검증 수행 여부
            
        Returns:
            (전처리된 DataFrame, 전처리 통계)
        """
        if df is None or df.empty:
            self.logger.warning("빈 데이터프레임이 입력되었습니다.")
            return pd.DataFrame(), {}
        
        original_shape = df.shape
        self.logger.info(f"전처리 시작: {original_shape[0]}행, {original_shape[1]}열")
        
        # 1. 데이터 검증
        if validate:
            validation_result = self.validate_data(df)
            if not validation_result['is_valid']:
                self.logger.error(f"데이터 검증 실패: {validation_result['errors']}")
                return pd.DataFrame(), validation_result
        
        # 2. 타임스탬프 처리
        df = self._process_timestamp(df)
        
        # 3. 결측치 처리
        df, missing_stats = self._handle_missing_data(df)
        
        # 4. 이상치 제거
        df, outlier_stats = self._remove_outliers(df)
        
        # 5. 데이터 정규화 (선택적)
        if self.enable_normalization:
            df = self._normalize_data(df)
        
        # 6. 데이터 일관성 검사
        consistency_result = self._check_data_consistency(df)
        
        # 전처리 통계 수집
        self.preprocessing_stats = {
            'original_shape': original_shape,
            'processed_shape': df.shape,
            'missing_data': missing_stats,
            'outliers': outlier_stats,
            'data_consistency': consistency_result,
            'validation': validation_result if validate else None
        }
        
        self.logger.info(
            f"전처리 완료: {df.shape[0]}행, {df.shape[1]}열 "
            f"(결측치: {missing_stats.get('removed_count', 0)}, "
            f"이상치: {outlier_stats.get('removed_count', 0)})"
        )
        
        return df, self.preprocessing_stats
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        데이터 검증
        
        Args:
            df: 검증할 DataFrame
            
        Returns:
            검증 결과 딕셔너리
        """
        errors = []
        warnings = []
        
        # 기본 검증
        if df is None or df.empty:
            errors.append("데이터프레임이 비어있습니다.")
            return {
                'is_valid': False,
                'errors': errors,
                'warnings': warnings
            }
        
        # 필수 컬럼 확인
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(f"필수 컬럼 누락: {', '.join(missing_columns)}")
        
        # 데이터 타입 확인
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"컬럼 {col}이 숫자형이 아닙니다.")
        
        # 가격 데이터 유효성 확인
        if 'close' in df.columns:
            invalid_prices = df[df['close'] <= 0]
            if len(invalid_prices) > 0:
                errors.append(f"유효하지 않은 가격 데이터: {len(invalid_prices)}개")
        
        # OHLC 논리 검증
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = df[
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ]
            if len(invalid_ohlc) > 0:
                errors.append(f"OHLC 논리 오류: {len(invalid_ohlc)}개")
        
        # 경고: 결측치 비율
        if len(df) > 0:
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if missing_ratio > self.missing_data_threshold:
                warnings.append(f"결측치 비율 높음: {missing_ratio:.2%}")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'data_quality_score': 1.0 - min(missing_ratio, 1.0) if len(df) > 0 else 0.0
        }
    
    def _process_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """타임스탬프 처리"""
        result = df.copy()
        
        # 타임스탬프 컬럼이 있으면 인덱스로 설정
        if 'timestamp' in result.columns:
            try:
                result['timestamp'] = pd.to_datetime(result['timestamp'], unit='ms', errors='coerce')
                result.set_index('timestamp', inplace=True)
            except Exception as e:
                self.logger.warning(f"타임스탬프 처리 실패: {e}")
        
        # 인덱스가 DatetimeIndex가 아니면 시도
        if not isinstance(result.index, pd.DatetimeIndex):
            try:
                result.index = pd.to_datetime(result.index, errors='coerce')
            except Exception:
                pass
        
        return result
    
    def _handle_missing_data(
        self,
        df: pd.DataFrame,
        method: str = 'forward_fill'
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        결측치 처리
        
        Args:
            df: 입력 DataFrame
            method: 처리 방법 ('forward_fill', 'backward_fill', 'interpolate', 'drop')
            
        Returns:
            (처리된 DataFrame, 통계)
        """
        result = df.copy()
        original_count = len(result)
        
        missing_before = result.isnull().sum().sum()
        
        if method == 'forward_fill':
            result = result.ffill()
        elif method == 'backward_fill':
            result = result.bfill()
        elif method == 'interpolate':
            result = result.interpolate(method='linear')
        elif method == 'drop':
            result = result.dropna()
        
        missing_after = result.isnull().sum().sum()
        removed_count = original_count - len(result)
        
        stats = {
            'missing_before': missing_before,
            'missing_after': missing_after,
            'removed_count': removed_count,
            'method': method
        }
        
        if missing_after > 0:
            self.logger.warning(f"결측치 처리 후에도 {missing_after}개 결측치 남음")
        
        return result, stats
    
    def _remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        이상치 제거
        
        Args:
            df: 입력 DataFrame
            columns: 이상치 제거할 컬럼 목록 (None이면 자동 선택)
            
        Returns:
            (처리된 DataFrame, 통계)
        """
        result = df.copy()
        original_count = len(result)
        
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
            columns = [col for col in columns if col in result.columns]
        
        outlier_mask = pd.Series([False] * len(result), index=result.index)
        
        for col in columns:
            if col not in result.columns:
                continue
            
            # Z-score 기반 이상치 감지
            z_scores = np.abs((result[col] - result[col].mean()) / result[col].std())
            col_outliers = z_scores > self.outlier_std_threshold
            outlier_mask |= col_outliers
            
            # IQR 기반 이상치 감지 (추가)
            Q1 = result[col].quantile(0.25)
            Q3 = result[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = (result[col] < lower_bound) | (result[col] > upper_bound)
            outlier_mask |= iqr_outliers
        
        # 이상치 제거
        result = result[~outlier_mask]
        removed_count = original_count - len(result)
        
        stats = {
            'removed_count': removed_count,
            'removed_pct': (removed_count / original_count * 100) if original_count > 0 else 0,
            'columns_checked': columns
        }
        
        if removed_count > 0:
            self.logger.info(f"이상치 {removed_count}개 제거 ({stats['removed_pct']:.2f}%)")
        
        return result, stats
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 정규화 (Min-Max Scaling)
        
        Args:
            df: 입력 DataFrame
            
        Returns:
            정규화된 DataFrame
        """
        result = df.copy()
        
        # 정규화할 컬럼 선택 (가격 데이터만)
        normalize_columns = ['open', 'high', 'low', 'close']
        normalize_columns = [col for col in normalize_columns if col in result.columns]
        
        for col in normalize_columns:
            col_min = result[col].min()
            col_max = result[col].max()
            
            if col_max > col_min:
                result[col] = (result[col] - col_min) / (col_max - col_min)
                self.logger.debug(f"컬럼 {col} 정규화 완료")
        
        return result
    
    def _check_data_consistency(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        데이터 일관성 검사
        
        Args:
            df: 검사할 DataFrame
            
        Returns:
            일관성 검사 결과
        """
        issues = []
        
        # OHLC 일관성
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = df[
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ]
            if len(invalid_ohlc) > 0:
                issues.append(f"OHLC 일관성 문제: {len(invalid_ohlc)}개")
        
        # 가격 연속성 (큰 가격 변동 감지)
        if 'close' in df.columns and len(df) > 1:
            price_changes = df['close'].pct_change().abs()
            large_changes = price_changes[price_changes > 0.1]  # 10% 이상 변동
            if len(large_changes) > 0:
                issues.append(f"큰 가격 변동: {len(large_changes)}개")
        
        # 거래량 일관성
        if 'volume' in df.columns:
            negative_volume = df[df['volume'] < 0]
            if len(negative_volume) > 0:
                issues.append(f"음수 거래량: {len(negative_volume)}개")
        
        return {
            'is_consistent': len(issues) == 0,
            'issues': issues,
            'issue_count': len(issues)
        }
    
    def get_preprocessing_stats(self) -> Dict[str, any]:
        """전처리 통계 조회"""
        return self.preprocessing_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.preprocessing_stats = {}


if __name__ == "__main__":
    # 테스트 코드
    import pandas as pd
    import numpy as np
    
    logger = get_logger(__name__)
    
    logger.info("=== 데이터 전처리 테스트 ===")
    
    # 테스트 데이터 생성
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 50000,
        'high': np.random.randn(100).cumsum() + 50100,
        'low': np.random.randn(100).cumsum() + 49900,
        'close': np.random.randn(100).cumsum() + 50000,
        'volume': np.random.rand(100) * 1000
    })
    
    # 일부 결측치 추가
    test_data.loc[10:15, 'close'] = np.nan
    
    # 일부 이상치 추가
    test_data.loc[20, 'close'] = 100000  # 이상치
    
    # 전처리 실행
    preprocessor = DataPreprocessor(
        outlier_std_threshold=3.0,
        missing_data_threshold=0.1
    )
    
    processed_df, stats = preprocessor.preprocess(test_data, validate=True)
    
    logger.info(f"원본 데이터: {test_data.shape}")
    logger.info(f"전처리 후: {processed_df.shape}")
    logger.info(f"전처리 통계: {stats}")

