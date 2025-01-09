import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import time
from tqdm import tqdm
import json
from datetime import datetime
import torch.nn as nn

from utils.visualization import TimeSeriesVisualizer
from utils.metrics import TimeSeriesMetrics

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor],
        test_data: Tuple[torch.Tensor, torch.Tensor],
        config: dict,
        data_processor=None,
        results_dir: Optional[str] = None,
        visualization_dir: Optional[str] = None,
        visualization_epoch: int = 5
    ):
        """
        Args:
            results_dir: 실험 결과를 저장할 디렉토리
            기타 기존 파라미터들...
        """
        # 기존 초기화 코드...
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.config = config
        self.data_processor = data_processor
        self.channel_independence = self.model.channel_independence
        
        # 결과 저장 디렉토리 설정
        if results_dir:
            self.results_dir = Path(results_dir)
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
            # 예측 결과 저장을 위한 하위 디렉토리
            self.predictions_dir = self.results_dir / 'predictions'
            self.predictions_dir.mkdir(exist_ok=True)
            
            # 실험 결과 저장을 위한 하위 디렉토리
            self.metrics_dir = self.results_dir / 'metrics'
            self.metrics_dir.mkdir(exist_ok=True)
        
        # 시각화 설정
        self.visualization_dir = Path(visualization_dir) if visualization_dir else None
        self.visualization_epoch = visualization_epoch
        if self.visualization_dir:
            self.visualizer = TimeSeriesVisualizer(str(self.visualization_dir))
        
        # 학습 설정
        self.epochs = self.config.get('epochs', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 10)
        self.gradient_clip_val = self.config.get('gradient_clip_val', None)
        
        # Checkpoint 설정
        if self.config.get('save_checkpoint', False):
            self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 학습 상태 초기화
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_history = []

    def _save_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_type: str):
        """예측 결과 저장"""
        if not hasattr(self, 'results_dir'):
            return
            
        # 모델 이름과 채널 모드로 파일명 생성
        model_name = self.model.__class__.__name__
        channel_mode = 'CI' if self.channel_independence else 'CD'
        base_name = f"{model_name}_{channel_mode}"
        
        # 예측값 저장
        pred_path = self.predictions_dir / f"{base_name}_{dataset_type}_predictions.npy"
        np.save(pred_path, y_pred)
        
        # 실제값 저장 (처음 한 번만)
        true_path = self.predictions_dir / f"{dataset_type}_true.npy"
        if not true_path.exists():
            np.save(true_path, y_true)
            
    def _save_experiment_results(self, metrics: Dict, train_history: List[Dict]):
        """실험 결과 저장"""
        if not hasattr(self, 'results_dir'):
            return
            
        # 실험 정보 구성
        model_name = self.model.__class__.__name__
        channel_mode = 'CI' if self.channel_independence else 'CD'
        
        experiment_results = {
            'model_name': model_name,
            'channel_mode': channel_mode,
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'hyperparameters': {
                'model_config': self.model.config,
                'training_config': self.config
            },
            'metrics': metrics,
            'training_history': {
                'train_loss': [h['train_loss'] for h in train_history],
                'val_loss': [h['val_loss'] for h in train_history]
            }
        }
        
        # JSON 형식으로 저장
        results_path = self.metrics_dir / f"{model_name}_{channel_mode}_results.json"
        with open(results_path, 'w') as f:
            json.dump(experiment_results, f, indent=4)


        
    def _create_data_loader(self, data: Tuple[torch.Tensor, torch.Tensor]) -> torch.utils.data.DataLoader:
        """데이터 로더 생성"""
        x_data, y_data = data
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(x_data),
            torch.FloatTensor(y_data)
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def _visualize_predictions(self, epoch: int, batch_data: Tuple[torch.Tensor, torch.Tensor]):
        """학습 중 예측 결과 시각화"""
        if not self.visualization_dir or (epoch + 1) % self.visualization_epoch != 0:
            return
                
        x, y_true = batch_data
        x = x.float().to(self.model.device)
        y_true = y_true.float().to(self.model.device)
        
        with torch.no_grad():
            y_pred = self.model(x)
        
        # CPU로 이동 및 numpy 변환
        x = x.cpu().numpy()
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        
        # 첫 번째 배치의 결과만 시각화
        self.visualizer.plot_predictions(
            y_true=y_true,
            y_pred=y_pred,
            feature_names=self.model.config['feature_names'],
            scaler=self.data_processor.scaler if hasattr(self, 'data_processor') else None,
            channel_independence=self.channel_independence,
            title=f'Predictions at Epoch {epoch+1}',
            filename=f'predictions_epoch_{epoch+1}.png'
        )
    
    def _train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        epoch_losses = []
        feature_losses = {} if self.channel_independence else None
        
        for i, batch in enumerate(train_loader):
            # 학습 스텝 수행
            loss_dict = self.model.training_step(batch)
            epoch_losses.append(loss_dict['loss'])
            
            # Channel independence 모드에서 feature별 loss 저장
            if self.channel_independence and 'feature_losses' in loss_dict:
                for feature, loss in loss_dict['feature_losses'].items():
                    if feature not in feature_losses:
                        feature_losses[feature] = []
                    feature_losses[feature].append(loss)
            
            # 첫 번째 배치에 대해 시각화
            if i == 0:
                self._visualize_predictions(self.current_epoch, batch)
            
            # Gradient clipping
            if self.gradient_clip_val:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_val
                )
        
        # 평균 loss 계산
        results = {'train_loss': np.mean(epoch_losses)}
        
        # Channel independence 모드에서 feature별 평균 loss 추가
        if feature_losses:
            results['feature_losses'] = {
                feature: np.mean(losses)
                for feature, losses in feature_losses.items()
            }
        
        return results
    
    def _validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """검증 수행"""
        self.model.eval()
        val_losses = []
        feature_losses = {} if self.channel_independence else None
        
        with torch.no_grad():
            for batch in val_loader:
                # 검증 스텝 수행
                loss_dict = self.model.validation_step(batch)
                val_losses.append(loss_dict['val_loss'])
                
                # Channel independence 모드에서 feature별 loss 저장
                if self.channel_independence and 'feature_losses' in loss_dict:
                    for feature, loss in loss_dict['feature_losses'].items():
                        if feature not in feature_losses:
                            feature_losses[feature] = []
                        feature_losses[feature].append(loss)
        
        # 평균 loss 계산
        results = {'val_loss': np.mean(val_losses)}
        
        # Channel independence 모드에서 feature별 평균 loss 추가
        if feature_losses:
            results['feature_losses'] = {
                feature: np.mean(losses)
                for feature, losses in feature_losses.items()
            }
        
        return results
    
    def train(self) -> list:
        """전체 학습 과정 수행"""
        self.logger.info("Starting training...")
        
        # 데이터 로더 생성
        train_loader = self._create_data_loader(self.train_data)
        val_loader = self._create_data_loader(self.val_data)
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # 학습
            train_logs = self._train_epoch(train_loader)
            
            # 검증
            val_logs = self._validate(val_loader)
            
            # 로그 통합
            logs = {**train_logs, **val_logs}
            
            # Learning rate scheduler step
            self.model.on_epoch_end(logs)
            
            # 학습 히스토리 업데이트
            self.train_history.append({
                'epoch': epoch,
                **logs,
                'learning_rate': self.model.optimizer.param_groups[0]['lr']
            })
            
            # 로깅
            log_message = f"Epoch {epoch+1}/{self.epochs}"
            log_message += f" - train_loss: {logs['train_loss']:.4f}"
            log_message += f" - val_loss: {logs['val_loss']:.4f}"
            if 'feature_losses' in logs:
                for feature, loss in logs['feature_losses'].items():
                    log_message += f" - {feature}_loss: {loss:.4f}"
            self.logger.info(log_message)
            
            # 체크포인트 저장
            if hasattr(self, 'checkpoint_dir'):
                self._save_checkpoint(epoch, logs)
            
            # Early stopping 확인
            if logs['val_loss'] < self.best_val_loss:
                self.best_val_loss = logs['val_loss']
                self.patience_counter = 0
                # 최상의 모델 저장
                if hasattr(self, 'checkpoint_dir') and self.config.get('save_best', True):
                    self._save_checkpoint(epoch, logs, is_best=True)
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # 학습 과정 시각화
            if self.visualization_dir and (epoch + 1) % self.visualization_epoch == 0:
                self.visualizer.plot_training_history(
                    self.train_history,
                    channel_independence=self.channel_independence,
                    title=f'Training History (Epoch {epoch+1})',
                    filename=f'training_history_epoch_{epoch+1}.png'
                )

        # 학습 완료 후 테스트 데이터에 대한 예측 수행
        test_x, test_y = self.test_data
        with torch.no_grad():
            test_pred = self.model(torch.FloatTensor(test_x).to(self.model.device))
            test_pred = test_pred.cpu().numpy()
            
        # 예측 결과 저장
        self._save_predictions(test_y, test_pred, 'test')
        
        # 메트릭 계산
        metrics = TimeSeriesMetrics.calculate_metrics(
            test_y, test_pred,
            feature_names=self.model.config['feature_names'],
            channel_independence=self.channel_independence
        )
        
        # 실험 결과 저장
        self._save_experiment_results(metrics, self.train_history)
        
        return self.train_history
    
    def _save_checkpoint(self, epoch: int, logs: Dict[str, float], is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'scheduler_state_dict': self.model.scheduler.state_dict() if self.model.scheduler else None,
            'logs': logs,
            'train_history': self.train_history
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            
            # 마지막 체크포인트만 유지
            if self.config.get('save_last', True):
                prev_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
                if prev_path.exists():
                    prev_path.unlink()
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")