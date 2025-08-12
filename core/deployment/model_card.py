"""
Model Card Generator Module
==========================

모델카드 생성을 담당하는 모듈
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ModelCardGenerator:
    """모델카드 생성을 담당하는 클래스"""
    
    def __init__(self):
        """ModelCardGenerator 초기화"""
        pass
    
    def create_model_card(self, model_path: str, repo_name: str) -> str:
        """
        한국어 모델카드 생성 (원본 함수 내용 보존)
        
        Args:
            model_path: 모델 저장 경로
            repo_name: 리포지토리 이름
            
        Returns:
            생성된 README.md 파일 경로
        """
        logger.info(f"모델카드 생성 시작: {repo_name}")
        
        # 원본 README 내용 그대로 보존
        readme_content = \
        f"""---
            license: apache-2.0
            base_model: unsloth/gpt-oss-20b
            tags:
            - unsloth
            - lora
            - korean
            - education
            - textbook
            - gpt-oss
            - 한국어
            - 교육
            - 파인튜닝
            language:
            - ko
            datasets:
            - maywell/korean_textbooks
            library_name: peft
            pipeline_tag: text-generation
            ---
            
            # 한국어 교육 자료 파인튜닝 모델 (Korean Textbook Fine-tuned Model)
            
            ## 📚 모델 소개
            
            이 모델은 **unsloth/gpt-oss-20b**를 기반으로 **maywell/korean_textbooks** 데이터셋으로 파인튜닝된 한국어 교육 전용 모델입니다.
            LoRA(Low-Rank Adaptation) 기술을 사용하여 효율적으로 학습되었으며, 한국어 교육 콘텐츠 생성에 특화되어 있습니다.
            
            ## 🎯 주요 특징
            
            - **베이스 모델**: unsloth/gpt-oss-20b (20B 파라미터)
            - **훈련 방법**: LoRA (Low-Rank Adaptation)
            - **특화 분야**: 한국어 교육 콘텐츠 생성
            - **데이터셋**: maywell/korean_textbooks
            - **언어**: 한국어 (Korean)
            
            ## 🚀 사용 방법
            
            ### 모델 로드
            
            ```python
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch
            
            # 베이스 모델 로드
            base_model = AutoModelForCausalLM.from_pretrained(
                "unsloth/gpt-oss-20b",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # LoRA 어댑터 로드
            model = PeftModel.from_pretrained(base_model, "{repo_name}")
            
            # 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
            ```
            
            ### 사용 예시
            
            ```python
            messages = [
                {{"role": "system", "content": "당신은 한국어로 교육 내용을 설명하는 도움이 되는 어시스턴트입니다."}},
                {{"role": "user", "content": "2의 거듭제곱에 대해 설명해주세요."}}
            ]
            
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(response)
            ```
            
            ## 📊 훈련 정보
            
            - **베이스 모델**: unsloth/gpt-oss-20b-unsloth-bnb-4bit
            - **훈련 스텝**: 30 steps
            - **LoRA Rank**: 8
            - **LoRA Alpha**: 16
            - **타겟 모듈**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
            - **데이터셋**: maywell/korean_textbooks
            
            ## 🎓 활용 분야
            
            이 모델은 다음 분야에서 우수한 성능을 보입니다:
            
            ### 수학 (Mathematics)
            - 기초 수학 개념 설명
            - 대수, 기하, 미적분 문제 해설
            - 수학 공식의 직관적 이해
            
            ### 과학 (Science)
            - 물리, 화학, 생물학 원리 설명
            - 실험 과정 및 결과 해석
            - 과학적 현상의 이해
            
            ### 언어 (Language)
            - 한국어 문법 및 어휘 설명
            - 문학 작품 분석 및 해석
            - 글쓰기 기법 안내
            
            ### 사회 (Social Studies)
            - 역사적 사건 및 인물 설명
            - 지리적 개념 및 현상
            - 사회 제도 및 문화 이해
            
            ## 💻 시스템 요구사항
            
            - **GPU 메모리**: 최소 16GB (권장 24GB+)
            - **시스템 RAM**: 최소 16GB
            - **Python**: 3.8+
            - **주요 라이브러리**: transformers, peft, torch
            
            ## ⚠️ 주의사항
            
            1. **교육 목적 특화**: 이 모델은 교육 콘텐츠 생성에 최적화되어 있습니다.
            2. **한국어 중심**: 한국어 외의 언어에서는 성능이 제한적일 수 있습니다.
            3. **사실 확인 필요**: 생성된 내용은 항상 검토하고 사실 확인이 필요합니다.
            4. **윤리적 사용**: 교육적이고 건전한 목적으로만 사용해주세요.
            
            ## 🔗 관련 링크
            
            - **베이스 모델**: [unsloth/gpt-oss-20b](https://huggingface.co/unsloth/gpt-oss-20b)
            - **데이터셋**: [maywell/korean_textbooks](https://huggingface.co/datasets/maywell/korean_textbooks)
            
            ## 📜 라이선스
            
            이 모델은 베이스 모델인 unsloth/gpt-oss-20b의 라이선스를 따릅니다.
            """

        try:
            # README.md 파일 저장
            readme_path = os.path.join(model_path, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_content)

            logger.info("모델카드(README.md) 생성 완료")
            print("✅ 모델카드(README.md) 생성 완료")
            
            return readme_path
            
        except Exception as e:
            logger.error(f"모델카드 생성 실패: {e}")
            raise
    
    def generate_model_card(self, model_path: str, repo_name: str, 
                          custom_description: Optional[str] = None) -> str:
        """
        모델카드 생성 (확장 가능한 버전)
        
        Args:
            model_path: 모델 저장 경로
            repo_name: 리포지토리 이름
            custom_description: 사용자 정의 설명
            
        Returns:
            생성된 README.md 파일 경로
        """
        return self.create_model_card(model_path, repo_name)