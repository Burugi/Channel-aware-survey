import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Union
import pandas as pd
from pathlib import Path

class TimeSeriesVisualizer:
    """시계열 예측 결과 시각화 클래스"""
    
    def __init__(self, save_dir: Optional[str] = None):
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 시각화 스타일 설정
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': [12, 6],
            'figure.dpi': 100,
            'font.size': 12,
            'lines.linewidth': 2,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True
        })
        
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        feature_names: List[str],
        scaler=None,
        timestamps: Optional[np.ndarray] = None,
        channel_independence: bool = False,
        title: str = "Predictions vs Actual Values",
        filename: Optional[str] = None
    ):
        # OT feature index 찾기
        ot_idx = feature_names.index('OT')
        
        # 역정규화 적용 (만약 scaler가 제공된 경우)
        if scaler is not None:
            # 3D -> 2D
            orig_shape = y_true.shape
            y_true_2d = y_true.reshape(-1, y_true.shape[-1])
            y_pred_2d = y_pred.reshape(-1, y_pred.shape[-1])
            
            # 역정규화
            y_true_2d = scaler.inverse_transform(y_true_2d)
            y_pred_2d = scaler.inverse_transform(y_pred_2d)
            
            # 2D -> 3D
            y_true = y_true_2d.reshape(orig_shape)
            y_pred = y_pred_2d.reshape(orig_shape)
            
        # OT feature만 선택
        y_true = y_true[..., [ot_idx]]
        y_pred = y_pred[..., [ot_idx]]
        feature_names = ['OT']
        n_features = 1
        
        # 배치 차원에 대한 평균과 표준편차 계산
        y_true_mean = np.mean(y_true, axis=0)
        y_pred_mean = np.mean(y_pred, axis=0)
        y_true_std = np.std(y_true, axis=0)
        y_pred_std = np.std(y_pred, axis=0)
        
        if channel_independence:
            # 각 feature별로 subplot 생성
            fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
            if n_features == 1:
                axes = [axes]
                
            for i, (ax, feature) in enumerate(zip(axes, feature_names)):
                x = range(len(y_true_mean)) if timestamps is None else timestamps
                
                # 실제값과 신뢰구간
                ax.plot(x, y_true_mean[:, i], color='#2ecc71', label='Actual')
                ax.fill_between(x, 
                              y_true_mean[:, i] - y_true_std[:, i],
                              y_true_mean[:, i] + y_true_std[:, i],
                              color='#2ecc71', alpha=0.2)
                
                # 예측값과 신뢰구간
                ax.plot(x, y_pred_mean[:, i], color='#e74c3c', 
                       label='Predicted', linestyle='--')
                ax.fill_between(x,
                              y_pred_mean[:, i] - y_pred_std[:, i],
                              y_pred_mean[:, i] + y_pred_std[:, i],
                              color='#e74c3c', alpha=0.2)
                
                ax.set_title(f'{feature}')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
        else:
            # 모든 feature를 하나의 그래프에 표시
            fig, ax = plt.subplots(figsize=(12, 6))
            x = range(len(y_true_mean)) if timestamps is None else timestamps
            
            colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
            for i, feature in enumerate(feature_names):
                color = colors[i % len(colors)]
                ax.plot(x, y_true_mean[:, i], color=color, 
                       label=f'Actual ({feature})')
                ax.plot(x, y_pred_mean[:, i], color=color, linestyle='--',
                       label=f'Predicted ({feature})')
            
            ax.set_title('All Features')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        
        if filename and hasattr(self, 'save_dir'):
            plt.savefig(self.save_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
            
    def plot_training_history(
        self,
        history: List[Dict],
        channel_independence: bool = False,
        title: str = "Training History",
        filename: Optional[str] = None
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(history) + 1)
        
        # 학습 및 검증 손실
        ax.plot(epochs, [h['train_loss'] for h in history], 
                color='#2ecc71', label='Train Loss')
                
        ax.plot(epochs, [h['val_loss'] for h in history],
                color='#e74c3c', label='Validation Loss', linestyle='--')
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        plt.tight_layout()
        
        if filename and hasattr(self, 'save_dir'):
            plt.savefig(self.save_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()