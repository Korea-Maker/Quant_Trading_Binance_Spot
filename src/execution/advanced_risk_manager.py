# src/execution/advanced_risk_manager.py

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

from src.utils.logger import get_logger


class AdvancedVaRCalculator:
    """고도화된 VaR 계산기"""

    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.logger = get_logger(__name__)
        self.confidence_levels = confidence_levels
        self.lookback_period = 252  # 1년

    def calculate_portfolio_var(self, positions: Dict[str, any],
                                price_data: Dict[str, pd.DataFrame],
                                method: str = 'historical') -> Dict[str, Dict[str, float]]:
        """
        고도화된 포트폴리오 VaR 계산

        Args:
            positions: 포지션 정보
            price_data: 각 자산의 가격 데이터
            method: 'historical', 'parametric', 'monte_carlo'

        Returns:
            각 신뢰수준별 VaR 값
        """
        try:
            if not positions or not price_data:
                return {str(cl): {'var': 0.0, 'cvar': 0.0} for cl in self.confidence_levels}

                # 1. 수익률 데이터 준비
            returns_data = self._prepare_returns_data(positions, price_data)

            if returns_data.empty:
                self.logger.warning("수익률 데이터가 없습니다.")
                return {str(cl): {'var': 0.0, 'cvar': 0.0} for cl in self.confidence_levels}

                # 2. 포트폴리오 가중치 계산
            weights = self._calculate_portfolio_weights(positions, price_data)

            # 3. 선택된 방법으로 VaR 계산
            if method == 'historical':
                var_results = self._historical_var(returns_data, weights)
            elif method == 'parametric':
                var_results = self._parametric_var(returns_data, weights)
            elif method == 'monte_carlo':
                var_results = self._monte_carlo_var(returns_data, weights)
            else:
                # 복합 방법 (3가지 방법의 평균)
                hist_var = self._historical_var(returns_data, weights)
                param_var = self._parametric_var(returns_data, weights)
                mc_var = self._monte_carlo_var(returns_data, weights)

                var_results = {}
                for cl_str in hist_var.keys():
                    var_results[cl_str] = {
                        'var': np.mean([hist_var[cl_str]['var'],
                                        param_var[cl_str]['var'],
                                        mc_var[cl_str]['var']]),
                        'cvar': np.mean([hist_var[cl_str]['cvar'],
                                         param_var[cl_str]['cvar'],
                                         mc_var[cl_str]['cvar']])
                    }

            self.logger.info(f"VaR 계산 완료 ({method} 방법)")
            return var_results

        except Exception as e:
            self.logger.error(f"VaR 계산 오류: {e}")
            return {str(cl): {'var': 0.0, 'cvar': 0.0} for cl in self.confidence_levels}

    def _prepare_returns_data(self, positions: Dict[str, any],
                              price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """수익률 데이터 준비"""
        returns_dict = {}

        for symbol in positions.keys():
            if symbol in price_data and not price_data[symbol].empty:
                # 일일 수익률 계산
                prices = price_data[symbol]['close'].tail(self.lookback_period)
                returns = prices.pct_change().dropna()

                if len(returns) > 30:  # 최소 30일 데이터 필요
                    returns_dict[symbol] = returns

        if not returns_dict:
            return pd.DataFrame()

            # 공통 날짜로 정렬
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()

        return returns_df

    def _calculate_portfolio_weights(self, positions: Dict[str, any],
                                     price_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """포트폴리오 가중치 계산"""
        weights = {}
        total_value = 0

        # 각 포지션의 현재 가치 계산
        for symbol, position in positions.items():
            if symbol in price_data and not price_data[symbol].empty:
                current_price = price_data[symbol]['close'].iloc[-1]
                position_value = abs(position.quantity) * current_price
                weights[symbol] = position_value
                total_value += position_value

                # 가중치 정규화
        if total_value > 0:
            for symbol in weights:
                weights[symbol] = weights[symbol] / total_value

        return pd.Series(weights)

    def _historical_var(self, returns_data: pd.DataFrame,
                        weights: pd.Series) -> Dict[str, Dict[str, float]]:
        """역사적 시뮬레이션 VaR"""
        # 포트폴리오 수익률 계산
        aligned_weights = weights.reindex(returns_data.columns, fill_value=0)
        portfolio_returns = (returns_data * aligned_weights).sum(axis=1)

        var_results = {}
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            var_percentile = portfolio_returns.quantile(alpha)

            # CVaR (조건부 VaR) 계산
            cvar = portfolio_returns[portfolio_returns <= var_percentile].mean()

            var_results[str(confidence_level)] = {
                'var': abs(var_percentile),
                'cvar': abs(cvar)
            }

        return var_results

    def _parametric_var(self, returns_data: pd.DataFrame,
                        weights: pd.Series) -> Dict[str, Dict[str, float]]:
        """모수적 VaR (분산-공분산 방법)"""
        # 포트폴리오 수익률 계산
        aligned_weights = weights.reindex(returns_data.columns, fill_value=0)
        portfolio_returns = (returns_data * aligned_weights).sum(axis=1)

        # 포트폴리오 통계량
        portfolio_mean = portfolio_returns.mean()
        portfolio_std = portfolio_returns.std()

        var_results = {}
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(alpha)

            # 정규분포 가정 VaR
            var_normal = abs(portfolio_mean + z_score * portfolio_std)

            # t-분포 가정 VaR (더 보수적)
            dof = len(portfolio_returns) - 1
            t_score = stats.t.ppf(alpha, dof)
            var_t = abs(portfolio_mean + t_score * portfolio_std)

            # CVaR 근사 계산
            cvar_normal = abs(portfolio_mean - portfolio_std * stats.norm.pdf(z_score) / alpha)

            var_results[str(confidence_level)] = {
                'var': max(var_normal, var_t),  # 보수적 접근
                'cvar': cvar_normal
            }

        return var_results

    def _monte_carlo_var(self, returns_data: pd.DataFrame,
                         weights: pd.Series,
                         num_simulations: int = 10000) -> Dict[str, Dict[str, float]]:
        """몬테카를로 시뮬레이션 VaR"""
        try:
            # 공분산 행렬 계산
            cov_matrix = returns_data.cov()
            mean_returns = returns_data.mean()

            # 가중치 정렬
            aligned_weights = weights.reindex(returns_data.columns, fill_value=0)
            aligned_means = mean_returns.reindex(returns_data.columns, fill_value=0)

            # 공분산 행렬이 양정정(positive definite)인지 확인
            try:
                np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                # 공분산 행렬을 정규화
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                eigenvals = np.maximum(eigenvals, 1e-8)  # 음수 고유값 제거
                cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

            # 몬테카를로 시뮬레이션
            portfolio_simulations = []

            for _ in range(num_simulations):
                # 다변량 정규분포에서 수익률 샘플링
                try:
                    simulated_returns = np.random.multivariate_normal(
                        aligned_means, cov_matrix
                    )

                    # 포트폴리오 수익률 계산
                    portfolio_return = np.dot(aligned_weights, simulated_returns)
                    portfolio_simulations.append(portfolio_return)
                except Exception:
                    # 시뮬레이션 실패 시 건너뛰기
                    continue

            if len(portfolio_simulations) < num_simulations * 0.9:  # 90% 이상 성공해야 함
                self.logger.warning(f"몬테카를로 시뮬레이션 성공률이 낮습니다: {len(portfolio_simulations)}/{num_simulations}")

            portfolio_simulations = np.array(portfolio_simulations)

            var_results = {}
            for confidence_level in self.confidence_levels:
                alpha = 1 - confidence_level
                var_mc = abs(np.percentile(portfolio_simulations, alpha * 100))

                # CVaR 계산 - 안전성 개선
                tail_losses = portfolio_simulations[portfolio_simulations <= -var_mc]
                cvar_mc = abs(np.mean(tail_losses)) if len(tail_losses) > 0 else var_mc

                var_results[str(confidence_level)] = {
                    'var': var_mc,
                    'cvar': cvar_mc
                }

            return var_results

        except Exception as e:
            self.logger.error(f"몬테카를로 VaR 계산 오류: {e}")
            # 폴백으로 historical VaR 사용
            return self._historical_var(returns_data, weights)

    def calculate_component_var(self, returns_data: pd.DataFrame,
                                weights: pd.Series) -> Dict[str, float]:
        """컴포넌트 VaR 계산 (각 자산이 포트폴리오 VaR에 기여하는 정도)"""
        try:
            # 포트폴리오 수익률
            aligned_weights = weights.reindex(returns_data.columns, fill_value=0)
            portfolio_returns = (returns_data * aligned_weights).sum(axis=1)
            portfolio_var = abs(portfolio_returns.quantile(0.05))

            component_vars = {}

            for asset in returns_data.columns:
                if asset in weights and weights[asset] > 0:
                    # 마진 기여도 계산
                    correlation = returns_data[asset].corr(portfolio_returns)
                    asset_std = returns_data[asset].std()
                    portfolio_std = portfolio_returns.std()

                    marginal_var = correlation * asset_std / portfolio_std * portfolio_var
                    component_var = weights[asset] * marginal_var

                    component_vars[asset] = component_var

            return component_vars

        except Exception as e:
            self.logger.error(f"컴포넌트 VaR 계산 오류: {e}")
            return {}


class AdvancedRiskMetrics:
    """고도화된 리스크 지표"""

    def __init__(self):
        self.logger = get_logger(__name__)

    def calculate_expected_shortfall(self, returns: pd.Series,
                                     confidence_level: float = 0.95) -> float:
        """기대 부족분 (Expected Shortfall/CVaR) 계산"""
        alpha = 1 - confidence_level
        var_threshold = returns.quantile(alpha)
        es = returns[returns <= var_threshold].mean()
        return abs(es)

    def calculate_maximum_drawdown(self, portfolio_values: pd.Series) -> Dict[str, any]:
        """최대 낙폭 계산"""
        cumulative = portfolio_values.cummax()
        drawdown = (portfolio_values - cumulative) / cumulative

        max_dd = drawdown.min()
        max_dd_duration = 0
        current_duration = 0

        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                current_duration = 0

        return {
            'max_drawdown': abs(max_dd),
            'max_duration': max_dd_duration,
            'current_drawdown': abs(drawdown.iloc[-1])
        }

    def calculate_calmar_ratio(self, returns: pd.Series,
                               portfolio_values: pd.Series) -> float:
        """칼마 비율 계산 (연간수익률 / 최대낙폭)"""
        annual_return = returns.mean() * 252
        max_dd = self.calculate_maximum_drawdown(portfolio_values)['max_drawdown']

        if max_dd == 0:
            return float('inf')

        return annual_return / max_dd

    def calculate_sortino_ratio(self, returns: pd.Series,
                                risk_free_rate: float = 0.02) -> float:
        """소티노 비율 계산"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf')

        downside_deviation = np.sqrt((downside_returns ** 2).mean())

        if downside_deviation == 0:
            return float('inf')

        return (excess_returns.mean() * 252) / (downside_deviation * np.sqrt(252))