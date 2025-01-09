import numpy as np
from typing import Dict, Union, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TimeSeriesMetrics:
    """시계열 예측 모델 평가를 위한 지표 계산 클래스"""
    
    @staticmethod
    def _prepare_data(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        """데이터 전처리: 3차원 데이터를 2차원으로 변환
        
        Args:
            y_true: 실제값 (batch_size, seq_len, num_features)
            y_pred: 예측값 (batch_size, seq_len, num_features)
            
        Returns:
            전처리된 (y_true, y_pred) 튜플
        """
        if y_true.ndim == 3:
            y_true = y_true.reshape(-1, y_true.shape[-1])
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        return y_true, y_pred
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray, multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
        """Mean Squared Error"""
        y_true, y_pred = TimeSeriesMetrics._prepare_data(y_true, y_pred)
        return mean_squared_error(y_true, y_pred, multioutput=multioutput)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray, multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
        """Root Mean Squared Error"""
        y_true, y_pred = TimeSeriesMetrics._prepare_data(y_true, y_pred)
        return np.sqrt(mean_squared_error(y_true, y_pred, multioutput=multioutput))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray, multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
        """Mean Absolute Error"""
        y_true, y_pred = TimeSeriesMetrics._prepare_data(y_true, y_pred)
        return mean_absolute_error(y_true, y_pred, multioutput=multioutput)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
        """Mean Absolute Percentage Error"""
        y_true, y_pred = TimeSeriesMetrics._prepare_data(y_true, y_pred)
        
        epsilon = 1e-10  # 0으로 나누기 방지
        mape_values = np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon)) * 100
        
        if multioutput == 'raw_values':
            return np.mean(mape_values, axis=0)
        return np.mean(mape_values)
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray, multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
        """Symmetric Mean Absolute Percentage Error"""
        y_true, y_pred = TimeSeriesMetrics._prepare_data(y_true, y_pred)
        
        epsilon = 1e-10
        smape_values = 200 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon)
        
        if multioutput == 'raw_values':
            return np.mean(smape_values, axis=0)
        return np.mean(smape_values)
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray, multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
        """R-squared Score"""
        y_true, y_pred = TimeSeriesMetrics._prepare_data(y_true, y_pred)
        return r2_score(y_true, y_pred, multioutput=multioutput)
    
    @staticmethod
    def nrmse(y_true: np.ndarray, y_pred: np.ndarray, multioutput: str = 'uniform_average') -> Union[float, np.ndarray]:
        """Normalized Root Mean Squared Error"""
        y_true, y_pred = TimeSeriesMetrics._prepare_data(y_true, y_pred)
        
        rmse_values = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
        y_std = np.std(y_true, axis=0)
        nrmse_values = rmse_values / (y_std + 1e-10)
        
        if multioutput == 'raw_values':
            return nrmse_values
        return np.mean(nrmse_values)
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        feature_names: List[str] = None,
        channel_independence: bool = False
    ) -> Dict:
        """모든 지표 계산
        
        Args:
            y_true: 실제값 (batch_size, seq_len, num_features)
            y_pred: 예측값 (batch_size, seq_len, num_features)
            feature_names: 특성 이름 리스트
            channel_independence: Channel independence 모드 여부
            
        Returns:
            계산된 메트릭을 포함하는 딕셔너리:
            {
                'overall': {metric_name: value, ...},
                'by_feature': {feature_name: {metric_name: value, ...}, ...}
            }
        """
        if channel_independence:
            # Channel independence mode: 각 feature별로 개별 계산 후 평균
            feature_metrics = {}
            overall_metrics = {
                'mse': 0, 'rmse': 0, 'mae': 0, 'mape': 0, 
                'smape': 0, 'r2': 0, 'nrmse': 0
            }
            
            for i, feature in enumerate(feature_names):
                feature_metrics[feature] = {
                    'mse': TimeSeriesMetrics.mse(y_true[..., i:i+1], y_pred[..., i:i+1]),
                    'rmse': TimeSeriesMetrics.rmse(y_true[..., i:i+1], y_pred[..., i:i+1]),
                    'mae': TimeSeriesMetrics.mae(y_true[..., i:i+1], y_pred[..., i:i+1]),
                    'mape': TimeSeriesMetrics.mape(y_true[..., i:i+1], y_pred[..., i:i+1]),
                    'smape': TimeSeriesMetrics.smape(y_true[..., i:i+1], y_pred[..., i:i+1]),
                    'r2': TimeSeriesMetrics.r2(y_true[..., i:i+1], y_pred[..., i:i+1]),
                    'nrmse': TimeSeriesMetrics.nrmse(y_true[..., i:i+1], y_pred[..., i:i+1])
                }
                
                # 전체 메트릭에 누적
                for metric_name, value in feature_metrics[feature].items():
                    overall_metrics[metric_name] += value
            
            # 전체 메트릭 평균 계산
            num_features = len(feature_names)
            overall_metrics = {
                metric_name: value / num_features 
                for metric_name, value in overall_metrics.items()
            }
        else:
            # Channel dependence mode: 모든 feature를 함께 계산
            overall_metrics = {
                'mse': TimeSeriesMetrics.mse(y_true, y_pred),
                'rmse': TimeSeriesMetrics.rmse(y_true, y_pred),
                'mae': TimeSeriesMetrics.mae(y_true, y_pred),
                'mape': TimeSeriesMetrics.mape(y_true, y_pred),
                'smape': TimeSeriesMetrics.smape(y_true, y_pred),
                'r2': TimeSeriesMetrics.r2(y_true, y_pred),
                'nrmse': TimeSeriesMetrics.nrmse(y_true, y_pred)
            }
            
            # Feature별 메트릭 계산
            feature_metrics = {}
            if feature_names is not None:
                individual_metrics = {
                    'mse': TimeSeriesMetrics.mse(y_true, y_pred, multioutput='raw_values'),
                    'rmse': TimeSeriesMetrics.rmse(y_true, y_pred, multioutput='raw_values'),
                    'mae': TimeSeriesMetrics.mae(y_true, y_pred, multioutput='raw_values'),
                    'mape': TimeSeriesMetrics.mape(y_true, y_pred, multioutput='raw_values'),
                    'smape': TimeSeriesMetrics.smape(y_true, y_pred, multioutput='raw_values'),
                    'r2': TimeSeriesMetrics.r2(y_true, y_pred, multioutput='raw_values'),
                    'nrmse': TimeSeriesMetrics.nrmse(y_true, y_pred, multioutput='raw_values')
                }
                
                for i, feature in enumerate(feature_names):
                    feature_metrics[feature] = {
                        metric_name: values[i]
                        for metric_name, values in individual_metrics.items()
                    }
        
        return {
            'overall': overall_metrics,
            'by_feature': feature_metrics
        }