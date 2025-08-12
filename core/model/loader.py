"""
Model Loader Module
==================

모델 로딩 및 LoRA 설정을 담당하는 모듈
"""

import logging
from typing import Tuple, Any
from unsloth import FastLanguageModel
from ..config import ModelConfig

logger = logging.getLogger(__name__)


class ModelLoader:
    """모델 로딩 및 설정을 담당하는 클래스"""
    
    def __init__(self, config: ModelConfig):
        """
        ModelLoader 초기화
        
        Args:
            config: 모델 설정 객체
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        
        # 설정 검증
        self.config.validate_config()
        
    def load_base_model(self) -> Tuple[Any, Any]:
        """
        베이스 모델과 토크나이저 로딩 (원본 로직 보존)
        
        Returns:
            Tuple[model, tokenizer]: 로딩된 모델과 토크나이저
        """
        logger.info(f"모델 로딩 시작: {self.config.model_name}")
        
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                dtype=self.config.dtype,  # None for auto detection
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
                full_finetuning=self.config.full_finetuning,
                # token="hf_...", # use one if using gated models
            )
            
            self.model = model
            self.tokenizer = tokenizer
            
            logger.info("베이스 모델 로딩 완료")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            raise
    
    def setup_peft_model(self) -> Any:
        """
        LoRA PEFT 모델 설정 (원본 파라미터 보존)
        
        Returns:
            설정된 PEFT 모델
        """
        if self.model is None:
            raise ValueError("먼저 베이스 모델을 로딩해야 합니다.")
        
        logger.info("LoRA PEFT 모델 설정 시작")
        
        try:
            model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.lora_r,
                target_modules=self.config.target_modules,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias=self.config.lora_bias,
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                random_state=self.config.random_state,
                use_rslora=self.config.use_rslora,
                loftq_config=self.config.loftq_config,
            )
            
            self.model = model
            
            logger.info(f"PEFT 모델 설정 완료: {self.config.get_model_info()}")
            return model
            
        except Exception as e:
            logger.error(f"PEFT 모델 설정 실패: {e}")
            raise
    
    def load_and_setup_model(self) -> Tuple[Any, Any]:
        """
        전체 모델 로딩 및 설정 프로세스
        
        Returns:
            Tuple[model, tokenizer]: 설정 완료된 모델과 토크나이저
        """
        logger.info("전체 모델 설정 프로세스 시작")
        
        # 1. 베이스 모델 로딩
        model, tokenizer = self.load_base_model()
        
        # 2. PEFT 설정
        model = self.setup_peft_model()
        
        logger.info("모델 설정 프로세스 완료")
        return model, tokenizer
    
    def get_model_info(self) -> dict:
        """
        현재 모델 정보 반환
        
        Returns:
            모델 정보 딕셔너리
        """
        return {
            "model_name": self.config.model_name,
            "lora_r": self.config.lora_r, 
            "lora_alpha": self.config.lora_alpha,
            "target_modules": self.config.target_modules,
            "max_seq_length": self.config.max_seq_length,
            "load_in_4bit": self.config.load_in_4bit
        }