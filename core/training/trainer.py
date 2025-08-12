"""
Trainer Module
=============

SFT 훈련을 담당하는 모듈
"""

import logging
from typing import Any, Optional
from trl import SFTConfig, SFTTrainer
from .monitor import MemoryMonitor
from ..config import ModelConfig

logger = logging.getLogger(__name__)


class Trainer:
    """SFT 훈련을 담당하는 클래스"""
    
    def __init__(self, model: Any, tokenizer: Any, config: ModelConfig):
        """
        Trainer 초기화
        
        Args:
            model: 훈련할 모델
            tokenizer: 토크나이저
            config: 모델 설정
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.trainer = None
        self.memory_monitor = MemoryMonitor()
        
    def create_sft_config(self) -> SFTConfig:
        """
        SFT 설정 생성 (원본 파라미터 보존)
        
        Returns:
            SFT 설정 객체
        """
        return SFTConfig(
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            seed=self.config.seed,
            output_dir=self.config.output_dir,
            report_to=self.config.report_to,
        )
    
    def setup_trainer(self, train_dataset: Any) -> SFTTrainer:
        """
        SFT 훈련자 설정 (원본 로직 보존)
        
        Args:
            train_dataset: 훈련 데이터셋
            
        Returns:
            설정된 SFT 훈련자
        """
        logger.info("SFT 훈련자 설정 시작")
        
        try:
            sft_config = self.create_sft_config()
            
            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                args=sft_config,
            )
            
            self.trainer = trainer
            
            logger.info("SFT 훈련자 설정 완료")
            return trainer
            
        except Exception as e:
            logger.error(f"훈련자 설정 실패: {e}")
            raise
    
    def train(self, train_dataset: Any, 
              monitor_memory: bool = True) -> Any:
        """
        모델 훈련 실행 (원본 로직 보존)
        
        Args:
            train_dataset: 훈련 데이터셋
            monitor_memory: 메모리 모니터링 여부
            
        Returns:
            훈련 통계
        """
        logger.info("모델 훈련 시작")
        
        try:
            # 1. 훈련자 설정
            if self.trainer is None:
                self.setup_trainer(train_dataset)
            
            # 2. 시작 메모리 기록
            if monitor_memory:
                self.memory_monitor.record_start_memory()
            
            # 3. 훈련 실행 (원본 코드)
            trainer_stats = self.trainer.train()
            
            # 4. 최종 통계 출력
            if monitor_memory:
                self.memory_monitor.print_training_stats(trainer_stats)
            
            logger.info("모델 훈련 완료")
            return trainer_stats
            
        except Exception as e:
            logger.error(f"훈련 실행 실패: {e}")
            raise
    
    def save_model(self, output_path: Optional[str] = None) -> str:
        """
        모델 저장
        
        Args:
            output_path: 저장 경로 (None인 경우 기본 경로 사용)
            
        Returns:
            저장된 경로
        """
        if self.trainer is None:
            raise ValueError("훈련자가 설정되지 않았습니다.")
        
        save_path = output_path or "./korean_textbook_model"
        
        logger.info(f"모델 저장 시작: {save_path}")
        
        try:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            logger.info(f"모델 저장 완료: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
            raise
    
    def get_training_config(self) -> dict:
        """
        훈련 설정 정보 반환
        
        Returns:
            훈련 설정 딕셔너리
        """
        return {
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "warmup_steps": self.config.warmup_steps,
            "max_steps": self.config.max_steps,
            "learning_rate": self.config.learning_rate,
            "optim": self.config.optim,
            "weight_decay": self.config.weight_decay,
            "lr_scheduler_type": self.config.lr_scheduler_type,
            "seed": self.config.seed,
            "output_dir": self.config.output_dir
        }
    
    def get_memory_report(self) -> dict:
        """
        메모리 사용량 보고서 반환
        
        Returns:
            메모리 사용량 딕셔너리
        """
        return self.memory_monitor.get_memory_report()
    
    def validate_training_setup(self) -> bool:
        """
        훈련 설정 검증
        
        Returns:
            검증 통과 여부
        """
        try:
            # 모델 상태 확인
            if self.model is None:
                logger.error("모델이 설정되지 않았습니다.")
                return False
            
            # 토크나이저 상태 확인
            if self.tokenizer is None:
                logger.error("토크나이저가 설정되지 않았습니다.")
                return False
            
            # 메모리 확인
            if self.memory_monitor.check_memory_usage(threshold=95.0):
                logger.warning("메모리 사용량이 높습니다.")
            
            # 설정 검증
            self.config.validate_config()
            
            logger.info("훈련 설정 검증 완료")
            return True
            
        except Exception as e:
            logger.error(f"훈련 설정 검증 실패: {e}")
            return False