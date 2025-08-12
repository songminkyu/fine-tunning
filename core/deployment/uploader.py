"""
Model Uploader Module
====================

모델 업로드 및 배포를 담당하는 모듈
"""

import os
import json
import logging
from typing import Optional
from huggingface_hub import HfApi, login
from .model_card import ModelCardGenerator

logger = logging.getLogger(__name__)


class ModelUploader:
    """모델 업로드 및 배포를 담당하는 클래스"""
    
    def __init__(self):
        """ModelUploader 초기화"""
        self.model_card_generator = ModelCardGenerator()
        self.api = None
    
    def fix_adapter_config(self, model_path: str) -> None:
        """
        adapter_config.json 수정 (에러 방지) - 원본 함수 보존
        
        Args:
            model_path: 모델 저장 경로
        """
        config_path = os.path.join(model_path, "adapter_config.json")

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # 문자열로 변환 (에러 해결)
                config['task_type'] = "CAUSAL_LM"
                config['peft_type'] = "LORA"

                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

                logger.info("adapter_config.json 수정 완료")
                print("✅ adapter_config.json 수정 완료")
                
            except Exception as e:
                logger.error(f"adapter_config.json 수정 실패: {e}")
                print(f"❌ adapter_config.json 수정 실패: {e}")
        else:
            logger.warning(f"adapter_config.json 파일을 찾을 수 없습니다: {config_path}")
    
    def login_to_hub(self, token: str) -> bool:
        """
        Hugging Face Hub 로그인
        
        Args:
            token: Hugging Face 토큰
            
        Returns:
            로그인 성공 여부
        """
        try:
            login(token=token)
            self.api = HfApi(token=token)
            logger.info("Hugging Face Hub 로그인 성공")
            return True
            
        except Exception as e:
            logger.error(f"Hugging Face Hub 로그인 실패: {e}")
            print(f"❌ 로그인 실패: {e}")
            return False
    
    def create_repository(self, repo_name: str) -> bool:
        """
        리포지토리 생성
        
        Args:
            repo_name: 리포지토리 이름
            
        Returns:
            생성 성공 여부
        """
        if self.api is None:
            logger.error("먼저 Hugging Face Hub에 로그인해야 합니다.")
            return False
        
        try:
            self.api.create_repo(repo_id=repo_name, exist_ok=True)
            logger.info(f"리포지토리 생성 완료: {repo_name}")
            print(f"✅ 리포지토리 생성: {repo_name}")
            return True
            
        except Exception as e:
            logger.error(f"리포지토리 생성 실패: {e}")
            print(f"❌ 리포지토리 생성 실패: {e}")
            return False
    
    def upload_model(self, model_path: str, repo_name: str, 
                    commit_message: Optional[str] = None) -> bool:
        """
        모델 업로드
        
        Args:
            model_path: 모델 저장 경로
            repo_name: 리포지토리 이름
            commit_message: 커밋 메시지
            
        Returns:
            업로드 성공 여부
        """
        if self.api is None:
            logger.error("먼저 Hugging Face Hub에 로그인해야 합니다.")
            return False
        
        if not os.path.exists(model_path):
            logger.error(f"모델 폴더가 없습니다: {model_path}")
            print(f"❌ 모델 폴더가 없습니다: {model_path}")
            return False
        
        commit_msg = commit_message or "Upload fine-tuned Korean model with model card"
        
        try:
            logger.info("모델 업로드 시작...")
            print("🚀 업로드 시작...")
            
            self.api.upload_folder(
                folder_path=model_path,
                repo_id=repo_name,
                commit_message=commit_msg
            )
            
            logger.info(f"모델 업로드 완료: {repo_name}")
            print(f"🎉 업로드 완료!")
            print(f"🔗 링크: https://huggingface.co/{repo_name}")
            return True
            
        except Exception as e:
            logger.error(f"모델 업로드 실패: {e}")
            print(f"❌ 업로드 실패: {e}")
            return False
    
    def save_and_upload_model(self, model: any, tokenizer: any, 
                             repo_name: Optional[str] = None,
                             token: Optional[str] = None,
                             model_path: str = "./korean_textbook_model") -> bool:
        """
        파인튜닝된 모델 저장 및 업로드 (원본 함수 기반)
        
        Args:
            model: 훈련된 모델
            tokenizer: 토크나이저
            repo_name: 리포지토리 이름
            token: Hugging Face 토큰
            model_path: 모델 저장 경로
            
        Returns:
            성공 여부
        """
        logger.info("모델 저장 및 업로드 프로세스 시작")
        
        # 1. 모델 저장
        print("💾 모델 저장 중...")
        
        try:
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            logger.info(f"모델 저장 완료: {model_path}")
            print(f"✅ 모델 저장 완료: {model_path}")
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
            print(f"❌ 모델 저장 실패: {e}")
            return False

        # 2. 사용자 입력 (repo_name이 없는 경우)
        if repo_name is None:
            repo_name = input("리포지토리 이름 (예: PAUL1122/my-korean-model): ").strip()
            if not repo_name:
                print("❌ 리포지토리 이름을 입력해주세요.")
                return False

        # 3. 모델카드 생성
        self.model_card_generator.create_model_card(model_path, repo_name)

        # 4. 설정 파일 수정
        self.fix_adapter_config(model_path)

        # 5. 토큰 입력 (token이 없는 경우)
        if token is None:
            token = input("Hugging Face 토큰: ").strip()
            if not token:
                print("❌ 토큰을 입력해주세요.")
                return False

        # 6. 업로드
        try:
            # 로그인
            if not self.login_to_hub(token):
                return False

            # 리포지토리 생성
            if not self.create_repository(repo_name):
                return False

            # 폴더 업로드
            return self.upload_model(model_path, repo_name)
            
        except Exception as e:
            logger.error(f"업로드 프로세스 실패: {e}")
            print(f"❌ 업로드 프로세스 실패: {e}")
            return False
    
    def upload_existing_model(self, repo_name: Optional[str] = None,
                            token: Optional[str] = None,
                            model_path: str = "./korean_textbook_model") -> bool:
        """
        이미 저장된 모델 업로드 (원본 함수 기반)
        
        Args:
            repo_name: 리포지토리 이름
            token: Hugging Face 토큰
            model_path: 모델 저장 경로
            
        Returns:
            성공 여부
        """
        logger.info("기존 모델 업로드 프로세스 시작")
        
        # 1. 모델 폴더 확인
        if not os.path.exists(model_path):
            print(f"❌ 모델 폴더가 없습니다: {model_path}")
            print("먼저 save_and_upload_model() 함수를 실행해주세요.")
            return False

        # 2. 사용자 입력 (repo_name이 없는 경우)
        if repo_name is None:
            repo_name = input("리포지토리 이름 (예: PAUL1122/my-korean-model): ").strip()
            if not repo_name:
                print("❌ 리포지토리 이름을 입력해주세요.")
                return False

        # 3. 모델카드 생성 (없거나 업데이트)
        self.model_card_generator.create_model_card(model_path, repo_name)

        # 4. 설정 파일 수정
        self.fix_adapter_config(model_path)

        # 5. 토큰 입력 (token이 없는 경우)
        if token is None:
            token = input("Hugging Face 토큰: ").strip()
            if not token:
                print("❌ 토큰을 입력해주세요.")
                return False

        # 6. 업로드
        try:
            # 로그인
            if not self.login_to_hub(token):
                return False

            # 리포지토리 생성
            if not self.create_repository(repo_name):
                return False

            # 폴더 업로드
            return self.upload_model(model_path, repo_name)
            
        except Exception as e:
            logger.error(f"업로드 프로세스 실패: {e}")
            print(f"❌ 업로드 프로세스 실패: {e}")
            return False
    
    def print_usage_guide(self) -> None:
        """
        사용법 안내 (원본 출력)
        """
        print("📖 사용 방법:")
        print("1. 파인튜닝 직후 저장+업로드: save_and_upload_model()")
        print("2. 이미 저장된 모델 업로드: upload_existing_model()")
        print()