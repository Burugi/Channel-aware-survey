import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json
import sys

def setup_logging(
    log_dir: str,
    logger_name: Optional[str] = None,
    log_level: int = logging.INFO,
    add_timestamp: bool = True,
    channel_independence: bool = False
) -> logging.Logger:
    """중앙화된 로깅 설정
    
    Args:
        log_dir: 로그 파일을 저장할 디렉토리
        logger_name: 로거 이름 (None이면 '__main__' 사용)
        log_level: 로깅 레벨 (기본값: INFO)
        add_timestamp: 로그 파일 이름에 타임스탬프 추가 여부
        channel_independence: Channel independence 모드 여부
    
    Returns:
        설정된 Logger 객체
    """
    logger_name = logger_name or '__main__'
    logger = logging.getLogger(logger_name)
    
    # 이미 핸들러가 설정되어 있다면 추가 설정하지 않음
    if logger.handlers:
        return logger
        
    logger.setLevel(log_level)
    
    # 로그 디렉토리 생성
    log_dir = Path(log_dir)
    mode_dir = log_dir / ('channel_independence' if channel_independence else 'channel_dependence')
    mode_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 핸들러 설정
    if add_timestamp:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = mode_dir / f'run_{current_time}.log'
    else:
        log_file = mode_dir / f'{logger_name}.log'
        
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # 포매터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_hyperparameters(
    logger: logging.Logger,
    config: dict,
    channel_independence: bool = False,
    prefix: str = ""
):
    """하이퍼파라미터 로깅
    
    Args:
        logger: Logger 객체
        config: 설정 딕셔너리
        channel_independence: Channel independence 모드 여부
        prefix: 로그 메시지 앞에 추가할 접두사
    """
    logger.info(f"{prefix}Training Mode: {'Channel Independence' if channel_independence else 'Channel Dependence'}")
    logger.info(f"{prefix}Hyperparameters:")
    
    # 설정을 카테고리별로 정리
    categories = {
        'Model Configuration': config.get('model_config', {}),
        'Training Configuration': config.get('trainer_config', {}),
        'Data Configuration': config.get('data_config', {}),
        'Experiment Configuration': config.get('exp_config', {})
    }
    
    for category, params in categories.items():
        logger.info(f"{prefix}{category}:")
        for key, value in params.items():
            if isinstance(value, dict):
                logger.info(f"{prefix}  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"{prefix}    {sub_key}: {sub_value}")
            else:
                logger.info(f"{prefix}  {key}: {value}")

def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    feature_names: Optional[list] = None,
    channel_independence: bool = False,
    epoch: Optional[int] = None,
    prefix: str = ""
):
    """메트릭 로깅
    
    Args:
        logger: Logger 객체
        metrics: 메트릭 딕셔너리
        feature_names: 특성 이름 리스트 (optional)
        channel_independence: Channel independence 모드 여부
        epoch: 현재 에폭 (optional)
        prefix: 로그 메시지 앞에 추가할 접두사
    """
    epoch_str = f" (Epoch {epoch})" if epoch is not None else ""
    logger.info(f"{prefix}Metrics{epoch_str}:")
    
    # 전체 메트릭 로깅
    if 'overall' in metrics:
        logger.info(f"{prefix}Overall Metrics:")
        for metric_name, value in metrics['overall'].items():
            if isinstance(value, (int, float)):
                logger.info(f"{prefix}  {metric_name}: {value:.4f}")
            else:
                logger.info(f"{prefix}  {metric_name}: {value}")
    
    # Feature별 메트릭 로깅
    if channel_independence and 'by_feature' in metrics and metrics['by_feature']:
        logger.info(f"{prefix}Feature-wise Metrics:")
        for feature, feature_metrics in metrics['by_feature'].items():
            logger.info(f"{prefix}  {feature}:")
            for metric_name, value in feature_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{prefix}    {metric_name}: {value:.4f}")
                else:
                    logger.info(f"{prefix}    {metric_name}: {value}")

def log_training_summary(
    logger: logging.Logger,
    train_history: list,
    best_metrics: Dict[str, Any],
    total_time: float,
    channel_independence: bool = False,
    prefix: str = ""
):
    """학습 결과 요약 로깅
    
    Args:
        logger: Logger 객체
        train_history: 학습 히스토리
        best_metrics: 최상의 메트릭
        total_time: 총 학습 시간 (초)
        channel_independence: Channel independence 모드 여부
        prefix: 로그 메시지 앞에 추가할 접두사
    """
    logger.info(f"{prefix}Training Summary:")
    logger.info(f"{prefix}Total Training Time: {total_time:.2f} seconds")
    logger.info(f"{prefix}Total Epochs: {len(train_history)}")
    
    # Best epoch 찾기
    best_epoch = min(range(len(train_history)), 
                    key=lambda i: train_history[i]['val_loss'])
    logger.info(f"{prefix}Best Epoch: {best_epoch + 1}")
    
    # 최상의 메트릭 로깅
    log_metrics(
        logger=logger,
        metrics=best_metrics,
        channel_independence=channel_independence,
        prefix=prefix + "  "
    )

def save_experiment_summary(
    save_dir: str,
    train_history: list,
    best_metrics: Dict[str, Any],
    config: dict,
    channel_independence: bool = False
):
    """실험 결과 요약 저장
    
    Args:
        save_dir: 저장 디렉토리
        train_history: 학습 히스토리
        best_metrics: 최상의 메트릭
        config: 설정 딕셔너리
        channel_independence: Channel independence 모드 여부
    """
    save_dir = Path(save_dir)
    mode_dir = save_dir / ('channel_independence' if channel_independence else 'channel_dependence')
    mode_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'channel_independence': channel_independence,
        'config': config,
        'train_history': train_history,
        'best_metrics': best_metrics
    }
    
    summary_file = mode_dir / 'experiment_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)