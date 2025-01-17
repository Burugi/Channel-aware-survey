import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple
import logging
import numpy as np

class BaseTimeSeriesModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 설정 저장
        self.config = config
        
        # 기본 설정
        self.input_len = self.config['input_len']
        self.pred_len = self.config['pred_len']
        self.num_features = self.config['num_features']
        self.base_features = self.config['base_features']
        self.channel_independence = self.config.get('channel_independence', False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # CI mode에서 feature별 독립적인 모델 구축을 위한 메서드
        self._build_model()
        self.to(self.device)
        
        # optimizer, scheduler, loss function 설정
        self._setup_training()

    def forward(self, x):
        """순전파"""
        if self.channel_independence:
            batch_size = x.size(0)
            outputs = []
            
            # 각 feature별로 독립적 처리
            for i in range(self.base_features):
                feature_input = x[..., i:i+1]  # (batch_size, seq_len, 1)
                feature_output = self._forward_single_feature(feature_input, i)  # (batch_size, pred_len, 1)
                outputs.append(feature_output)
            
            # 모든 feature의 출력을 결합
            return torch.cat(outputs, dim=-1)  # (batch_size, pred_len, num_features)
        else:
            return self._forward_all_features(x)

    def _forward_single_feature(self, x, feature_idx):
        """CI mode에서 단일 feature 처리 (하위 클래스에서 구현)"""
        raise NotImplementedError

    def _forward_all_features(self, x):
        """CD mode에서 전체 feature 처리 (하위 클래스에서 구현)"""
        raise NotImplementedError

    def _setup_training(self):
        """optimizer, scheduler, loss function 설정"""
        # Optimizer 설정
        optimizer_config = self.config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'Adam')
        optimizer_params = optimizer_config.get('params', {})
        
        optimizer_class = getattr(torch.optim, optimizer_name)
        self.optimizer = optimizer_class(self.parameters(), **optimizer_params)
        
        # Scheduler 설정
        scheduler_config = self.config.get('scheduler', {})
        if scheduler_config:
            scheduler_name = scheduler_config.get('name')
            scheduler_params = scheduler_config.get('params', {})
            
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
            self.scheduler = scheduler_class(self.optimizer, **scheduler_params)
        else:
            self.scheduler = None
            
        # Loss function 설정
        loss_name = self.config.get('loss', 'MSELoss')
        loss_params = self.config.get('loss_params', {})
        
        loss_class = getattr(torch.nn, loss_name)
        self.loss_fn = loss_class(**loss_params)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """단일 학습 스텝"""
        x, y = batch
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        
        # 순전파
        y_pred = self(x)
        
        if self.channel_independence:
            # Channel independence mode에서는 각 feature별 loss를 평균
            # 평균이 아니고 그냥 합으로 하기 
            loss = 0
            for i in range(self.base_features):
                loss += self.loss_fn(y_pred[..., i], y[..., i])
            # loss = loss / self.base_features
        else:
            loss = self.loss_fn(y_pred, y)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        
        if hasattr(self, 'gradient_clip_val') and self.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)
            
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    @abstractmethod
    def _build_model(self):
        """모델 아키텍처 구축 (하위 클래스에서 구현)"""
        pass

    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """단일 검증 스텝"""
        x, y = batch
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        
        with torch.no_grad():
            y_pred = self(x)
            if self.channel_independence:
                # Channel independence mode에서는 각 feature별 loss를 평균
                loss = 0
                for i in range(self.base_features):
                    loss += self.loss_fn(y_pred[..., i], y[..., i])
                # loss = loss / self.base_features
            else:
                loss = self.loss_fn(y_pred, y)
            
        return {'val_loss': loss.item()}
    
    def on_epoch_end(self, logs: Dict[str, float]):
        """에폭 종료 시 호출되는 메서드"""
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(logs['val_loss'])
            else:
                self.scheduler.step()
    
    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """예측 수행"""
        self.eval()
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        x = x.to(self.device).float()
        
        with torch.no_grad():
            y_pred = self(x)
            
        return y_pred.cpu().numpy()
    
    def save_model(self, path: str):
        """모델 저장"""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }
        torch.save(save_dict, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.config = checkpoint['config']
        self.logger.info(f"Model loaded from {path}")