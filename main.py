"""
Main Execution Script
====================

한국어 교육용 GPT-OSS 모델 파인튜닝 메인 스크립트
원본 gpt-oss-kor-fine-tune.py의 모든 기능을 모듈화하여 재구성
"""

import logging
from core import (
    ModelConfig, ModelLoader, ModelInference, 
    DataProcessor, Trainer, ModelUploader
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """메인 실행 함수"""
    
    print("=" * 60)
    print("🚀 한국어 교육용 GPT-OSS 모델 파인튜닝")
    print("=" * 60)
    
    try:
        # 1. 설정 로딩
        logger.info("설정 초기화")
        config = ModelConfig()
        print(f"📋 설정 로딩 완료: {config.get_model_info()}")
        
        # 2. 모델 로딩 및 설정
        logger.info("모델 로딩 시작")
        model_loader = ModelLoader(config)
        model, tokenizer = model_loader.load_and_setup_model()
        print("✅ 모델 로딩 및 PEFT 설정 완료")
        
        # 3. 기본 추론 테스트 (원본 첫 번째 테스트)
        logger.info("기본 추론 테스트 실행")
        inference = ModelInference(model, tokenizer, config)
        print("\n🔍 기본 추론 테스트:")
        inference.test_basic_inference()
        
        # 4. 데이터셋 로딩 및 처리
        logger.info("데이터셋 처리 시작")
        data_processor = DataProcessor(config)
        dataset = data_processor.load_dataset()
        print(f"✅ 데이터셋 로딩 완료: {len(dataset)}개 샘플")
        
        # 데이터 미리보기 (원본 print문 재현)
        print("\n📊 데이터 미리보기:")
        data_processor.preview_data(num_samples=1)
        
        # 5. 훈련 준비 및 실행
        logger.info("훈련 시작")
        trainer = Trainer(model, tokenizer, config)
        
        # 훈련 설정 검증
        if not trainer.validate_training_setup():
            print("❌ 훈련 설정 검증 실패")
            return False
        
        # 데이터셋 준비
        train_dataset = data_processor.prepare_for_training(tokenizer)
        
        # 훈련 실행 (메모리 모니터링 포함)
        print("\\n🏋️ 모델 훈련 시작...")
        trainer_stats = trainer.train(train_dataset)
        print("✅ 모델 훈련 완료")
        
        # 6. 훈련 후 추론 테스트 (원본 두 번째 테스트)
        print("\\n🔍 한국어 교육 추론 테스트:")
        inference.test_korean_inference()
        
        # 7. 모델 저장
        model_path = trainer.save_model()
        print(f"💾 모델 저장 완료: {model_path}")
        
        # 8. 업로드 옵션 제공 (원본 사용법 안내)
        print("\\n" + "=" * 50)
        uploader = ModelUploader()
        uploader.print_usage_guide()
        
        # 실행 선택 (원본 로직)
        choice = input("선택하세요 (1: 저장+업로드, 2: 업로드만, 3: 종료): ").strip()
        
        if choice == "1":
            print("\\n📤 모델 업로드 프로세스 시작...")
            success = uploader.save_and_upload_model(model, tokenizer)
            if success:
                print("🎉 모델 업로드 성공!")
            else:
                print("❌ 모델 업로드 실패")
                
        elif choice == "2":
            print("\\n📤 기존 모델 업로드 프로세스 시작...")
            success = uploader.upload_existing_model()
            if success:
                print("🎉 모델 업로드 성공!")
            else:
                print("❌ 모델 업로드 실패")
                
        elif choice == "3":
            print("👋 프로그램을 종료합니다.")
        else:
            print("올바른 선택지를 입력해주세요 (1, 2, 또는 3)")
        
        print("\\n" + "=" * 60)
        print("🎯 한국어 교육용 모델 파인튜닝 완료!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"메인 프로세스 실패: {e}")
        print(f"❌ 오류 발생: {e}")
        return False


def run_inference_only():
    """추론만 실행하는 함수 (훈련된 모델 테스트용)"""
    
    print("🔍 추론 전용 모드")
    
    try:
        config = ModelConfig()
        
        # 이미 훈련된 모델이 있다면 로딩
        # 이 부분은 실제 사용 시 저장된 모델 경로로 수정 필요
        model_loader = ModelLoader(config)
        model, tokenizer = model_loader.load_and_setup_model()
        
        inference = ModelInference(model, tokenizer, config)
        
        print("\\n전체 추론 테스트 실행:")
        inference.run_all_tests()
        
        return True
        
    except Exception as e:
        logger.error(f"추론 실행 실패: {e}")
        print(f"❌ 추론 실행 실패: {e}")
        return False


def run_upload_only():
    """업로드만 실행하는 함수"""
    
    print("📤 업로드 전용 모드")
    
    try:
        uploader = ModelUploader()
        uploader.print_usage_guide()
        
        success = uploader.upload_existing_model()
        
        if success:
            print("🎉 모델 업로드 성공!")
        else:
            print("❌ 모델 업로드 실패")
            
        return success
        
    except Exception as e:
        logger.error(f"업로드 실행 실패: {e}")
        print(f"❌ 업로드 실행 실패: {e}")
        return False


if __name__ == "__main__":
    
    print("실행 모드를 선택하세요:")
    print("1. 전체 파인튜닝 파이프라인")
    print("2. 추론만 실행")
    print("3. 업로드만 실행")
    
    mode = input("모드 선택 (1/2/3): ").strip()
    
    if mode == "1":
        main()
    elif mode == "2":
        run_inference_only()
    elif mode == "3":
        run_upload_only()
    else:
        print("올바른 모드를 선택해주세요 (1, 2, 또는 3)")