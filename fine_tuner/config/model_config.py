"""
Model Configuration Module
=========================

중앙화된 모델 설정 관리
"""

from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class ModelConfig:
    """모델 설정을 관리하는 클래스"""
    
    # 기본 모델 설정
    max_seq_length: int = 1024
    dtype: Optional[Any] = None
    load_in_4bit: bool = True
    
    # 지원되는 4bit 모델들 (원본과 동일)
    fourbit_models: List[str] = None
    
    # 기본 모델
    model_name: str = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    
    # LoRA 설정 (원본 파라미터 유지)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0
    lora_bias: str = "none"
    target_modules: List[str] = None
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[Any] = None
    full_finetuning: bool = False
    
    # 훈련 설정
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 30
    learning_rate: float = 2e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs"
    report_to: str = "none"
    
    # 데이터셋 설정
    dataset_name: str = "maywell/korean_textbooks"
    dataset_config: str = "claude_evol"
    dataset_split: str = "train"
    
    # 추론 설정
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    do_sample: bool = True
    reasoning_effort: str = "medium"
    
    def __post_init__(self):
        """초기화 후 기본값 설정"""
        if self.fourbit_models is None:
            self.fourbit_models = [
                "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
                "unsloth/gpt-oss-120b-unsloth-bnb-4bit", 
                "unsloth/gpt-oss-20b",
                "unsloth/gpt-oss-120b",
            ]
        
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
    
    def get_model_info(self) -> str:
        """모델 정보 문자열 반환"""
        return f"Model: {self.model_name}, LoRA r={self.lora_r}, alpha={self.lora_alpha}"
    
    def validate_config(self) -> bool:
        """설정 유효성 검사"""
        if self.model_name not in self.fourbit_models:
            print(f"Warning: {self.model_name}이 지원 모델 목록에 없습니다.")
        
        if self.lora_r <= 0:
            raise ValueError("LoRA rank는 0보다 커야 합니다.")
        
        if self.max_steps <= 0:
            raise ValueError("훈련 스텝은 0보다 커야 합니다.")
        
        return True