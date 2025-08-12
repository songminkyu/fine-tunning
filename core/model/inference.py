"""
Model Inference Module
=====================

모델 추론 및 테스트를 담당하는 모듈
"""

import logging
from typing import List, Dict, Any, Optional
from transformers import TextStreamer
from ..config import ModelConfig

logger = logging.getLogger(__name__)


class ModelInference:
    """모델 추론 및 테스트를 담당하는 클래스"""
    
    def __init__(self, model: Any, tokenizer: Any, config: ModelConfig):
        """
        ModelInference 초기화
        
        Args:
            model: 훈련된 모델
            tokenizer: 토크나이저
            config: 모델 설정
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    
    def create_test_messages(self) -> List[Dict[str, str]]:
        """
        테스트용 메시지 생성 (원본 예제 기반)
        
        Returns:
            테스트 메시지 리스트
        """
        # 원본의 첫 번째 테스트 예제
        basic_test = [
            {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."}
        ]
        
        # 원본의 한국어 교육 테스트 예제
        korean_test = [
            {
                "role": "system", 
                "content": "당신은 한국어로 교육 내용을 설명하는 도움이 되는 어시스턴트입니다."
            },
            {
                "role": "user", 
                "content": "2의 거듭제곱에 대해 설명해주세요."
            }
        ]
        
        return {"basic": basic_test, "korean": korean_test}
    
    def prepare_inputs(self, messages: List[Dict[str, str]], 
                      reasoning_effort: Optional[str] = None) -> Dict[str, Any]:
        """
        입력 데이터 준비 (원본 로직 보존)
        
        Args:
            messages: 대화 메시지 리스트
            reasoning_effort: 추론 강도 설정
            
        Returns:
            모델 입력 데이터
        """
        effort = reasoning_effort or self.config.reasoning_effort
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=effort
        ).to(self.model.device)
        
        return inputs
    
    def generate_response(self, inputs: Dict[str, Any], 
                         max_new_tokens: Optional[int] = None,
                         temperature: Optional[float] = None,
                         use_streamer: bool = True,
                         **kwargs) -> Any:
        """
        응답 생성 (원본 파라미터 보존)
        
        Args:
            inputs: 모델 입력 데이터
            max_new_tokens: 최대 새 토큰 수
            temperature: 온도 파라미터
            use_streamer: 스트리머 사용 여부
            **kwargs: 추가 생성 파라미터
            
        Returns:
            생성된 응답
        """
        # 기본값 설정
        max_tokens = max_new_tokens or self.config.max_new_tokens
        temp = temperature or self.config.temperature
        
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temp,
            "top_p": self.config.top_p,
            "do_sample": self.config.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # 스트리머 설정
        if use_streamer:
            generation_kwargs["streamer"] = TextStreamer(self.tokenizer)
        
        logger.info(f"응답 생성 시작 - max_tokens: {max_tokens}, temp: {temp}")
        
        try:
            output = self.model.generate(**inputs, **generation_kwargs)
            logger.info("응답 생성 완료")
            return output
            
        except Exception as e:
            logger.error(f"응답 생성 실패: {e}")
            raise
    
    def test_basic_inference(self) -> None:
        """
        기본 추론 테스트 (원본 첫 번째 테스트)
        """
        logger.info("기본 추론 테스트 시작")
        
        test_messages = self.create_test_messages()
        inputs = self.prepare_inputs(test_messages["basic"])
        
        # 원본 설정으로 생성 (128 토큰, temperature 1.0)
        _ = self.generate_response(
            inputs, 
            max_new_tokens=128, 
            temperature=1.0,
            top_p=1.0
        )
        
        logger.info("기본 추론 테스트 완료")
    
    def test_korean_inference(self) -> None:
        """
        한국어 교육 추론 테스트 (원본 두 번째 테스트)
        """
        logger.info("한국어 교육 추론 테스트 시작")
        
        test_messages = self.create_test_messages()
        inputs = self.prepare_inputs(test_messages["korean"])
        
        # 원본 설정으로 생성 (512 토큰, temperature 0.7)
        _ = self.generate_response(
            inputs,
            max_new_tokens=512,
            temperature=0.7
        )
        
        logger.info("한국어 교육 추론 테스트 완료")
    
    def run_all_tests(self) -> None:
        """
        모든 테스트 실행
        """
        logger.info("전체 추론 테스트 시작")
        
        try:
            self.test_basic_inference()
            print("\n" + "="*50 + "\n")
            self.test_korean_inference()
            
            logger.info("전체 추론 테스트 완료")
            
        except Exception as e:
            logger.error(f"추론 테스트 실패: {e}")
            raise
    
    def custom_inference(self, messages: List[Dict[str, str]], 
                        **generation_kwargs) -> Any:
        """
        사용자 정의 추론
        
        Args:
            messages: 대화 메시지
            **generation_kwargs: 생성 파라미터
            
        Returns:
            생성된 응답
        """
        inputs = self.prepare_inputs(messages)
        return self.generate_response(inputs, **generation_kwargs)