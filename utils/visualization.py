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
        channel_independence: bool = False,  # 파라미터는 유지하되 사용하지 않음
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
        
        # 첫 번째 샘플의 OT feature 선택
        y_true_sample = y_true[0, :, ot_idx]  # shape: (pred_len,)
        y_pred_sample = y_pred[0, :, ot_idx]  # shape: (pred_len,)
        
        # 시각화
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(y_true_sample)) if timestamps is None else timestamps
        
        # 실제값과 예측값 플롯
        ax.plot(x, y_true_sample, color='#2ecc71', label='Actual', marker='o')
        ax.plot(x, y_pred_sample, color='#e74c3c', label='Predicted', 
            linestyle='--', marker='x')
        
        ax.set_title('Occupancy Level (OT) - Single Sequence Comparison')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.suptitle(title, y=1.02)
        plt.tight_layout()
        
        if filename and hasattr(self, 'save_dir'):
            plt.savefig(self.save_dir / filename, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

        # 추가로 MSE, MAE 등의 오차 지표를 계산하여 출력
        mse = np.mean((y_true_sample - y_pred_sample) ** 2)
        mae = np.mean(np.abs(y_true_sample - y_pred_sample))
        print(f"Single Sequence Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
            
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