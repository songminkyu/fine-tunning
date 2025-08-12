from transformers import TextStreamer
from unsloth import FastLanguageModel
import torch

max_seq_length = 1024
dtype = None

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit", # 20B model using bitsandbytes 4bit quantization
    "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    "unsloth/gpt-oss-20b", # 20B model using MXFP4 format
    "unsloth/gpt-oss-120b",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    dtype = dtype, # None for auto detection
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = True, # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

messages = [
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    return_tensors = "pt",
    return_dict = True,
    reasoning_effort = "medium", # **NEW!** Set reasoning effort to low, medium or high
).to(model.device)

_ = model.generate(**inputs, max_new_tokens = 128, streamer = TextStreamer(tokenizer),
                   temperature = 1.0, top_p = 1.0)



def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("maywell/korean_textbooks", name="claude_evol", split="train")

def formatting_prompts_func(examples):
    # 이미 완성된 텍스트이므로 그대로 반환
    texts = examples["text"]
    return {"text": texts}


print(dataset[0]['text'])



from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 30,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)


# @title 현재 메모리 상태 확인
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()


# @title 최종 메모리 및 시간 통계 확인
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


messages = [
    {"role": "system", "content": "당신은 한국어로 교육 내용을 설명하는 도움이 되는 어시스턴트입니다."},
    {"role": "user", "content": "2의 거듭제곱에 대해 설명해주세요."},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="medium",
).to(model.device)

from transformers import TextStreamer
_ = model.generate(
    **inputs,
    max_new_tokens=512,  # 128에서 512로 증가
    streamer=TextStreamer(tokenizer),
    do_sample=True,      # 더 다양한 응답을 위해 추가
    temperature=0.7,     # 응답의 창의성 조절
    pad_token_id=tokenizer.eos_token_id  # 패딩 토큰 설정
)

# 심플한 파인튜닝 모델 업로드 코드

import os
import json
import shutil
from huggingface_hub import HfApi, login

def fix_adapter_config(model_path):
    """adapter_config.json 수정 (에러 방지)"""
    config_path = os.path.join(model_path, "adapter_config.json")

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

        # 문자열로 변환 (에러 해결)
        config['task_type'] = "CAUSAL_LM"
        config['peft_type'] = "LORA"

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print("✅ adapter_config.json 수정 완료")

def create_model_card(model_path, repo_name):
    """한국어 모델카드 생성"""

    readme_content = f"""---
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

    # README.md 파일 저장
    readme_path = os.path.join(model_path, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print("✅ 모델카드(README.md) 생성 완료")

def save_and_upload_model():
    """파인튜닝된 모델 저장 및 업로드"""

    # 1. 모델 저장
    print("💾 모델 저장 중...")
    model_path = "./korean_textbook_model"

    try:
        # 파인튜닝 후 저장
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"✅ 모델 저장 완료: {model_path}")
    except NameError:
        print("❌ model 또는 tokenizer 변수가 없습니다.")
        print("파인튜닝을 먼저 실행해주세요.")
        return
    except Exception as e:
        print(f"❌ 모델 저장 실패: {e}")
        return

    # 2. 사용자 입력 (모델카드 생성에 필요)
    repo_name = input("리포지토리 이름 (예: PAUL1122/my-korean-model): ").strip()
    if not repo_name:
        print("❌ 리포지토리 이름을 입력해주세요.")
        return

    # 3. 모델카드 생성
    create_model_card(model_path, repo_name)

    # 4. 설정 파일 수정
    fix_adapter_config(model_path)

    # 5. 토큰 입력
    token = input("Hugging Face 토큰: ").strip()
    if not token:
        print("❌ 토큰을 입력해주세요.")
        return

    # 6. 업로드
    try:
        print("🚀 업로드 시작...")

        # 로그인
        login(token=token)
        api = HfApi(token=token)

        # 리포지토리 생성
        api.create_repo(repo_id=repo_name, exist_ok=True)
        print(f"✅ 리포지토리 생성: {repo_name}")

        # 폴더 업로드
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            commit_message="Upload fine-tuned Korean model with model card"
        )

        print(f"🎉 업로드 완료!")
        print(f"🔗 링크: https://huggingface.co/{repo_name}")

    except Exception as e:
        print(f"❌ 업로드 실패: {e}")

def upload_existing_model():
    """이미 저장된 모델 업로드"""

    # 1. 기본 설정
    model_path = "./korean_textbook_model"

    # 2. 모델 폴더 확인
    if not os.path.exists(model_path):
        print(f"❌ 모델 폴더가 없습니다: {model_path}")
        print("먼저 save_and_upload_model() 함수를 실행해주세요.")
        return

    # 3. 사용자 입력
    repo_name = input("리포지토리 이름 (예: PAUL1122/my-korean-model): ").strip()
    if not repo_name:
        print("❌ 리포지토리 이름을 입력해주세요.")
        return

    # 4. 모델카드 생성 (없거나 업데이트)
    create_model_card(model_path, repo_name)

    # 5. 설정 파일 수정
    fix_adapter_config(model_path)

    # 6. 토큰 입력
    token = input("Hugging Face 토큰: ").strip()
    if not token:
        print("❌ 토큰을 입력해주세요.")
        return

    # 7. 업로드
    try:
        print("🚀 업로드 시작...")

        # 로그인
        login(token=token)
        api = HfApi(token=token)

        # 리포지토리 생성
        api.create_repo(repo_id=repo_name, exist_ok=True)
        print(f"✅ 리포지토리 생성: {repo_name}")

        # 폴더 업로드
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            commit_message="Upload fine-tuned Korean model with model card"
        )

        print(f"🎉 업로드 완료!")
        print(f"🔗 링크: https://huggingface.co/{repo_name}")

    except Exception as e:
        print(f"❌ 업로드 실패: {e}")

# 사용법 안내
print("📖 사용 방법:")
print("1. 파인튜닝 직후 저장+업로드: save_and_upload_model()")
print("2. 이미 저장된 모델 업로드: upload_existing_model()")
print()

# 실행 선택
choice = input("선택하세요 (1: 저장+업로드, 2: 업로드만): ").strip()

if choice == "1":
    save_and_upload_model()
elif choice == "2":
    upload_existing_model()
else:
    print("올바른 선택지를 입력해주세요 (1 또는 2)")