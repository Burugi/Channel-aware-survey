import argparse
from pathlib import Path
import yaml
from datetime import datetime
import numpy as np
from typing import Dict, Any
import logging

from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from trainers.trainer import Trainer
from trainers.hyperopt import HyperOptimizer
from utils.logging_utils import setup_logging
from utils.metrics import TimeSeriesMetrics
from utils.visualization import TimeSeriesVisualizer

def get_model_class(model_name: str):
    """모델 클래스 반환"""
    from models.implementations.lstm import LSTMModel
    from models.implementations.transformer import TransformerModel
    from models.implementations.tcn import TCNModel
    from models.implementations.autoformer import AutoformerModel
    from models.implementations.informer import InformerModel
    from models.implementations.rnn import RNNModel
    from models.implementations.adarnn import AdaRNNModel  # 추가
    from models.implementations.segrnn import SegRNNModel
    from models.implementations.dlinear import DLinearModel
    from models.implementations.scinet import SCINetModel
    from models.implementations.timemixer import TimeMixerModel
    models = {
        'lstm': LSTMModel,
        'transformer': TransformerModel,
        'tcn': TCNModel,
        'autoformer': AutoformerModel,
        'informer': InformerModel,
        'rnn': RNNModel,
        'adarnn': AdaRNNModel,  # 추가
        'segrnn': SegRNNModel,  # 추가
        'dlinear': DLinearModel,
        'scinet': SCINetModel,
        'timemixer': TimeMixerModel
    }
    return models.get(model_name.lower())

def load_config(model_name: str, dataset_name: str) -> dict:
    """설정 파일들을 로드하여 하나의 설정 딕셔너리로 반환"""
    config_dir = Path('configs')
    
    # 데이터 설정 로드
    with open(config_dir / 'data' / f'{dataset_name}.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    
    # 실험 설정 로드
    with open(config_dir / 'exp' / 'base.yaml', 'r') as f:
        exp_config = yaml.safe_load(f)
    
    # 모델 설정 로드
    with open(config_dir / 'model' / f'{model_name}.yaml', 'r') as f:
        model_config = yaml.safe_load(f)
    
    # 트레이너 설정 로드
    with open(config_dir / 'trainer' / 'base.yaml', 'r') as f:
        trainer_config = yaml.safe_load(f)
    
    return {
        'data_config': data_config,
        'exp_config': exp_config,
        'model_config': model_config,
        'trainer_config': trainer_config
    }

def run_hyperopt(args, logger):
    """하이퍼파라미터 최적화 실행"""
    # 설정 로드
    config = load_config(args.model, args.dataset)
    
    # Channel independence 설정 추가
    config['model_config']['channel_independence'] = args.channel_independence
    
    # 데이터 로더 초기화 및 특성 수 계산
    data_loader = DataLoader(config['data_config'], config['exp_config'])
    feature_info = data_loader.get_feature_info()
    
    config['model_config'].update({
        'num_features': feature_info['total_features'],
        'base_features': feature_info['base_features'],
        'feature_names': feature_info['feature_names'],
        'input_len': config['exp_config']['input_len'],
        'pred_len': config['exp_config']['pred_len']
    })
    
    # 데이터 로드 및 처리
    logger.info("Loading and processing data...")
    data = data_loader.load_data()
    
    data_processor = DataProcessor(config['exp_config'])
    processed_data = data_processor.prepare_data(
        data,
        channel_independence=args.channel_independence
    )
    
    # 하이퍼파라미터 최적화
    logger.info(f"Starting hyperparameter optimization for {args.model}...")
    model_class = get_model_class(args.model)
    
    study_name = f"{args.model}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study_dir = Path(args.output_dir)
    if args.channel_independence:
        study_dir = study_dir / 'channel_independence'
    else:
        study_dir = study_dir / 'channel_dependence'
    study_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = HyperOptimizer(
        model_class=model_class,
        model_name=args.model,
        train_data=processed_data['train'],
        val_data=processed_data['val'],
        test_data=processed_data['test'],
        base_config=config,
        study_name=study_name,
        channel_independence=args.channel_independence,
        n_trials=args.n_trials,
        study_dir=str(study_dir)
    )
    
    results = optimizer.optimize()
    return results

def run_experiment(args, logger):
    """단일 모델 실험 실행"""
    # 설정 로드
    config = load_config(args.model, args.dataset)
    
    # Channel independence 설정 추가
    config['model_config']['channel_independence'] = args.channel_independence
    
    # 데이터 로더 초기화 및 특성 수 계산
    data_loader = DataLoader(config['data_config'], config['exp_config'])
    feature_info = data_loader.get_feature_info()
    
    # 모델 설정 업데이트
    config['model_config'].update({
        'num_features': feature_info['total_features'],
        'base_features': feature_info['base_features'],
        'feature_names': feature_info['feature_names'],
        'input_len': config['exp_config']['input_len'],
        'pred_len': config['exp_config']['pred_len']
    })
    
    # 데이터 로드 및 처리
    logger.info("Loading and processing data...")
    data = data_loader.load_data()
    
    data_processor = DataProcessor(config['exp_config'])
    processed_data = data_processor.prepare_data(
        data,
        channel_independence=args.channel_independence
    )
    
    # 출력 디렉토리 설정
    base_output_dir = Path(args.output_dir) / args.model
    if args.channel_independence:
        base_output_dir = base_output_dir / 'channel_independence'
    else:
        base_output_dir = base_output_dir / 'channel_dependence'
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델 초기화
    logger.info("Initializing model...")
    model = get_model_class(args.model)(config['model_config'])
    
    # 시각화 설정
    viz_dir = base_output_dir / 'visualizations' if args.visualize else None
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 결과 저장 디렉토리 설정
    results_dir = base_output_dir / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = Trainer(
        model=model,
        train_data=processed_data['train'],
        val_data=processed_data['val'],
        test_data=processed_data['test'],
        config=config['trainer_config'],
        data_processor=data_processor,
        results_dir=str(results_dir),
        visualization_dir=viz_dir if args.visualize else None,
        visualization_epoch=5
    )
    
    logger.info("Starting training...")
    history = trainer.train()
    
    # 테스트 및 메트릭 계산
    test_x, test_y = processed_data['test']
    predictions = model.predict(test_x)
    
    metrics = TimeSeriesMetrics.calculate_metrics(
        test_y,
        predictions,
        feature_names=feature_info['feature_names'],
        channel_independence=args.channel_independence
    )
    
    # 결과 로깅
    logger.info("\nTest Results:")
    for metric_name, value in metrics['overall'].items():
        logger.info(f"{metric_name.upper()}: {value:.4f}")
    
    return {
        'history': history,
        'metrics': metrics
    }

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    parser.add_argument('--model', type=str, required=True,
                     help='Model type (Ex. transformer/lstm/tcn)')
    parser.add_argument('--dataset', type=str, required=True,
                     help='Dataset name (corresponding to config file name)')
    parser.add_argument('--channel_independence', action='store_true',
                     help='Enable channel independence mode')
    parser.add_argument('--optimize', action='store_true',
                     help='Run hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=100,
                     help='Number of optimization trials')
    parser.add_argument('--visualize', action='store_true',
                     help='Generate visualizations')
    parser.add_argument('--output_dir', type=str, default='outputs',
                     help='Directory for saving outputs')
    parser.add_argument('--log_dir', type=str, default='logs',
                     help='Directory for saving logs')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logger = setup_logging(args.log_dir, channel_independence=args.channel_independence)
    logger.info(f"Starting experiment with args: {args}")
    
    try:
        if args.optimize:
            run_hyperopt(args, logger)
        else:
            run_experiment(args, logger)
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()