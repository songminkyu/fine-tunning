"""
Data Processor Module
====================

데이터셋 로딩 및 처리를 담당하는 모듈
"""

import logging
from typing import Dict, List, Any
from datasets import load_dataset
from ..config import ModelConfig

logger = logging.getLogger(__name__)


class DataProcessor:
    """데이터셋 로딩 및 처리를 담당하는 클래스"""
    
    def __init__(self, config: ModelConfig):
        """
        DataProcessor 초기화
        
        Args:
            config: 모델 설정 객체
        """
        self.config = config
        self.dataset = None
        
    def load_dataset(self) -> Any:
        """
        데이터셋 로딩 (원본 로직 보존)
        
        Returns:
            로딩된 데이터셋
        """
        logger.info(f"데이터셋 로딩 시작: {self.config.dataset_name}")
        
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                name=self.config.dataset_config,
                split=self.config.dataset_split
            )
            
            self.dataset = dataset
            
            logger.info(f"데이터셋 로딩 완료 - 크기: {len(dataset)}")
            return dataset
            
        except Exception as e:
            logger.error(f"데이터셋 로딩 실패: {e}")
            raise
    
    def formatting_prompts_func_messages(self, examples: Dict[str, List]) -> Dict[str, List[str]]:
        """
        메시지 형태 데이터 포맷팅 함수 (원본 첫 번째 함수)
        
        Args:
            examples: 데이터셋 예제들
            
        Returns:
            포맷팅된 텍스트 딕셔너리
        """
        convos = examples["messages"]
        # 이 부분은 tokenizer가 필요하므로 별도 메서드로 처리
        # 원본 코드: texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"conversations": convos}
    
    def formatting_prompts_func_text(self, examples: Dict[str, List]) -> Dict[str, List[str]]:
        """
        텍스트 형태 데이터 포맷팅 함수 (원본 두 번째 함수)
        
        Args:
            examples: 데이터셋 예제들
            
        Returns:
            포맷팅된 텍스트 딕셔너리
        """
        # 이미 완성된 텍스트이므로 그대로 반환 (원본 로직)
        texts = examples["text"]
        return {"text": texts}
    
    def apply_chat_template_formatting(self, examples: Dict[str, List], 
                                     tokenizer: Any) -> Dict[str, List[str]]:
        """
        채팅 템플릿을 적용한 포맷팅 (tokenizer 필요)
        
        Args:
            examples: 데이터셋 예제들
            tokenizer: 토크나이저
            
        Returns:
            포맷팅된 텍스트 딕셔너리
        """
        if "messages" in examples:
            convos = examples["messages"]
            texts = [
                tokenizer.apply_chat_template(
                    convo, 
                    tokenize=False, 
                    add_generation_prompt=False
                ) for convo in convos
            ]
            return {"text": texts}
        else:
            # 이미 텍스트 형태인 경우
            return self.formatting_prompts_func_text(examples)
    
    def preview_data(self, num_samples: int = 1) -> None:
        """
        데이터 미리보기 (원본 print 문 기반)
        
        Args:
            num_samples: 미리볼 샘플 수
        """
        if self.dataset is None:
            logger.warning("데이터셋이 로딩되지 않았습니다.")
            return
        
        logger.info(f"데이터 미리보기 ({num_samples}개 샘플):")
        
        for i in range(min(num_samples, len(self.dataset))):
            print(f"\n=== 샘플 {i+1} ===")
            print(self.dataset[i]['text'])
            print("=" * 50)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        데이터셋 정보 반환
        
        Returns:
            데이터셋 정보 딕셔너리
        """
        if self.dataset is None:
            return {"status": "not_loaded"}
        
        return {
            "name": self.config.dataset_name,
            "config": self.config.dataset_config,
            "split": self.config.dataset_split,
            "size": len(self.dataset),
            "features": list(self.dataset.features.keys()) if hasattr(self.dataset, 'features') else [],
            "status": "loaded"
        }
    
    def prepare_for_training(self, tokenizer: Any) -> Any:
        """
        훈련용 데이터셋 준비
        
        Args:
            tokenizer: 토크나이저
            
        Returns:
            훈련용 데이터셋
        """
        if self.dataset is None:
            raise ValueError("데이터셋이 로딩되지 않았습니다.")
        
        logger.info("훈련용 데이터셋 준비 시작")
        
        try:
            # 데이터셋 특성에 따라 적절한 포맷팅 함수 선택
            if "messages" in self.dataset.column_names:
                formatted_dataset = self.dataset.map(
                    lambda examples: self.apply_chat_template_formatting(examples, tokenizer),
                    batched=True
                )
            else:
                formatted_dataset = self.dataset.map(
                    self.formatting_prompts_func_text,
                    batched=True
                )
            
            logger.info("훈련용 데이터셋 준비 완료")
            return formatted_dataset
            
        except Exception as e:
            logger.error(f"데이터셋 준비 실패: {e}")
            raise