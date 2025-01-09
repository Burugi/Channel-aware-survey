import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import logging

class DataLoader:
    """시계열 데이터 로더"""
    
    def __init__(self, data_config: dict, exp_config: dict):
        """
        Args:
            data_config: 데이터 설정
            exp_config: 실험 설정
        """
        self.logger = logging.getLogger(__name__)
        self.data_config = data_config
        self.exp_config = exp_config
        
        # 데이터 경로 설정
        self.data_path = Path(data_config['data_path'])
        self.file_name = data_config['file_name']
        
        # 기본 feature 설정
        self.base_features = data_config['base_features']
        
        # 데이터 분할 비율
        self.train_ratio = exp_config['train_ratio']
        self.val_ratio = exp_config['val_ratio']
        
        # Scale 여부
        self.scale = exp_config['scale']
    
    def load_data(self) -> Dict[str, np.ndarray]:
        """데이터 로드"""
        # CSV 파일 로드
        df = pd.read_csv(self.data_path / self.file_name)
        
        # date 컬럼 제외하고 base feature만 선택
        data = df[self.base_features].values
        
        # 데이터 분할
        n = len(data)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def get_feature_info(self) -> Dict:
        """Feature 정보 반환"""
        return {
            'total_features': len(self.base_features),
            'base_features': len(self.base_features),
            'feature_names': self.base_features
        }