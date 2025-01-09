import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Type
import json
from datetime import datetime
import pandas as pd
import copy
from utils.logging_utils import setup_logging, log_metrics
from trainers.trainer import Trainer

class HyperOptimizer:
    def __init__(
        self,
        model_class: Type[torch.nn.Module],
        model_name: str,
        train_data: tuple,
        val_data: tuple,
        test_data: tuple,
        base_config: dict,
        study_name: str,
        channel_independence: bool = False,
        n_trials: int = 100,
        study_dir: str = 'hyperopt_results'
    ):
        self.model_class = model_class
        self.model_name = model_name
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.base_config = base_config
        self.n_trials = n_trials
        self.channel_independence = channel_independence
        
        # 하이퍼파라미터 탐색 범위 로드
        hyperopt_config_path = Path('configs/hyperopt') / f'{model_name}.yaml'
        with open(hyperopt_config_path, 'r') as f:
            self.hyperopt_config = yaml.safe_load(f)
        
        # 결과 저장 디렉토리 설정
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = Path(base_config['data_config']['file_name']).stem
        result_dir = f"{model_name}_{dataset_name}_{current_time}"
        self.study_dir = Path(study_dir) / result_dir
        self.study_dir.mkdir(parents=True, exist_ok=True)
        
        # 로거 설정
        self.logger = setup_logging(
            str(self.study_dir),
            logger_name='HyperOptimizer',
            add_timestamp=False,
            channel_independence=channel_independence
        )
        
        # Optuna study 생성
        self.study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            sampler=TPESampler(),
            storage=f'sqlite:///{self.study_dir}/study.db',
            load_if_exists=True
        )
        
        # 결과 DataFrame 초기화
        self.results_df = pd.DataFrame(columns=[
            'trial_number', 'datetime', 'val_loss', 'channel_independence',
            'model_params', 'training_params', 'metrics'
        ])
        
        # 하이퍼파라미터 탐색 범위 로드
        hyperopt_config_path = Path('configs/hyperopt') / f'{model_name}.yaml'
        with open(hyperopt_config_path, 'r') as f:
            self.hyperopt_config = yaml.safe_load(f)
            
        # Study 디렉토리 설정
        self.study_dir = Path(study_dir)
        self.study_dir.mkdir(parents=True, exist_ok=True)
        
        # Optuna study 생성
        self.study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            sampler=TPESampler(),
            storage=f'sqlite:///{self.study_dir}/study.db',
            load_if_exists=True
        )
    
    def _suggest_value(self, trial: Trial, name: str, values: list) -> Any:
        """하이퍼파라미터 값 선택"""
        if isinstance(values[0], list):
            return trial.suggest_categorical(name, [tuple(v) for v in values])
        elif isinstance(values[0], (int, float)):
            return trial.suggest_categorical(name, values)
        else:
            return trial.suggest_categorical(name, values)
    
    def _suggest_params(self, trial: Trial) -> Dict[str, Any]:
        """하이퍼파라미터 샘플링"""
        params = {}
        
        # 모델 파라미터 샘플링
        for param_name, param_values in self.hyperopt_config['model_params'].items():
            params[param_name] = self._suggest_value(trial, param_name, param_values)
        
        # Channel independence 파라미터 추가
        params['channel_independence'] = self.channel_independence
        
        # 학습 파라미터 샘플링
        for param_name, param_values in self.hyperopt_config['training_params'].items():
            params[param_name] = self._suggest_value(trial, param_name, param_values)
        
        self.logger.info(f"Sampled parameters for trial {trial.number}:")
        for name, value in params.items():
            self.logger.info(f"  {name}: {value}")
            
        return params
    
    def _update_config(self, config: dict, params: Dict[str, Any]) -> dict:
        """설정 업데이트"""
        config = copy.deepcopy(config)
        
        # trainer 설정 업데이트
        config['trainer_config'].update({
            'batch_size': params['batch_size'],
            'optimizer': {
                'name': params['optimizer'],
                'params': {
                    'lr': params['learning_rate'],
                    'weight_decay': params['weight_decay']
                }
            }
        })
        
        # 모델 설정 업데이트
        for param_name, param_value in params.items():
            if param_name not in ['batch_size', 'optimizer', 'learning_rate', 'weight_decay']:
                config['model_config'][param_name] = param_value
                
        return config
    
    def objective(self, trial: Trial) -> float:
        """최적화 목적 함수"""
        # 하이퍼파라미터 샘플링
        params = self._suggest_params(trial)
        
        # 설정 업데이트
        config = self._update_config(self.base_config.copy(), params)
        
        try:
            # 모델 초기화
            self.logger.info(f"Initializing model for trial {trial.number}")
            model = self.model_class(config['model_config'])
            
            # 트레이너 초기화
            trainer = Trainer(
                model=model,
                train_data=self.train_data,
                val_data=self.val_data,
                test_data=self.test_data,
                config=config['trainer_config']
            )
            
            # 학습 수행
            history = trainer.train()
            
            # 검증 손실 계산
            val_loss = min(h['val_loss'] for h in history)
            
            # 결과 저장
            result = {
                'trial_number': trial.number,
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'val_loss': val_loss,
                'channel_independence': self.channel_independence,
                'model_params': params,
                'training_params': config['trainer_config'],
                'metrics': history[-1].get('metrics', {})  # 마지막 에폭의 메트릭
            }
            
            self.results_df = pd.concat([
                self.results_df,
                pd.DataFrame([result])
            ], ignore_index=True)
            
            # 중간 결과 저장
            self._save_results()
            
            return val_loss
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed with error: {str(e)}")
            raise optuna.exceptions.TrialPruned()

    def _save_results(self):
        """결과 저장"""
        # JSON 형식으로 저장
        results_dict = {
            'model_name': self.model_name,
            'channel_mode': 'CI' if self.channel_independence else 'CD',
            'best_trial': None,
            'all_trials': []
        }
        
        if not self.results_df.empty:
            best_idx = self.results_df['val_loss'].idxmin()
            best_trial = self.results_df.iloc[best_idx].to_dict()
            results_dict['best_trial'] = best_trial
            results_dict['all_trials'] = self.results_df.to_dict('records')
        
        results_path = self.study_dir / f"optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
    
    def _flatten_config(self, config: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """설정 딕셔너리를 평탄화"""
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def optimize(self) -> Dict:
        """하이퍼파라미터 최적화 수행"""
        self.logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        self.logger.info(f"Channel independence mode: {self.channel_independence}")
        
        # 최적화 실행
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        # 최종 결과 저장
        self._save_results()
        
        # 최적화 결과 로깅
        self.logger.info(f"Optimization completed")
        self.logger.info(f"Best trial: {self.study.best_trial.number}")
        self.logger.info(f"Best validation loss: {self.study.best_value:.4f}")
        self.logger.info("\nBest parameters:")
        for param, value in self.study.best_params.items():
            self.logger.info(f"{param}: {value}")
        
        return {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'study': self.study,
            'results_df': self.results_df
        }