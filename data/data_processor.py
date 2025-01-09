import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import logging

class DataProcessor:
    """시계열 데이터 전처리"""
    
    def __init__(self, exp_config: dict):
        """
        Args:
            exp_config: 실험 설정
        """
        self.logger = logging.getLogger(__name__)
        self.input_len = exp_config['input_len']
        self.pred_len = exp_config['pred_len']
        self.scale = exp_config.get('scale', True)
        
        self.scaler = StandardScaler() if self.scale else None
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 시퀀스 생성"""
        total_len = self.input_len + self.pred_len
        n_samples = len(data) - total_len + 1
        
        x = np.zeros((n_samples, self.input_len, data.shape[1]))
        y = np.zeros((n_samples, self.pred_len, data.shape[1]))
        
        for i in range(n_samples):
            x[i] = data[i:i + self.input_len]
            y[i] = data[i + self.input_len:i + total_len]
        
        return x, y
    
    def prepare_data(self, data: Dict[str, np.ndarray], channel_independence: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """데이터 준비
        
        Args:
            data: 'train', 'val', 'test' 키를 가진 데이터 딕셔너리
            channel_independence: Channel independence 모드 여부
        
        Returns:
            준비된 데이터 딕셔너리 (x, y 튜플 포함)
        """
        # 학습 데이터로 스케일러 학습
        if self.scale:
            self.scaler.fit(data['train'])
        
        processed_data = {}
        for split, split_data in data.items():
            # 스케일링 적용
            if self.scale:
                split_data = self.scaler.transform(split_data)
            
            # 시퀀스 생성
            x, y = self._create_sequences(split_data)
            processed_data[split] = (x, y)
        
        return processed_data
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """스케일링 역변환"""
        if self.scale and self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data