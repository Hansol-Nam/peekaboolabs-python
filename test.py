import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoModelForImageClassification
from PIL import Image
import numpy as np
import time
import os
from torchvision.transforms import Compose, Normalize, ToTensor


# 모델 클래스 정의
class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits  # logits만 반환 (추론 결과)

# 소프트맥스 함수 정의
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # 안정적인 계산을 위해 최대값을 뺌
    return exp_logits / np.sum(exp_logits)

float32_model_path = "coreml_model_test.mlpackage"
float16_model_path = "coreml_model_test_float16.mlpackage"

# PyTorch 모델 로드
model_name = "dima806/facial_emotions_image_detection"
pretrained_model = AutoModelForImageClassification.from_pretrained(model_name)
wrapped_model = WrappedModel(pretrained_model)
wrapped_model.eval()

# 샘플 입력 정의
example_input = torch.rand(1, 3, 224, 224)

# TorchScript 변환 (Tracing)
traced_model = torch.jit.trace(wrapped_model, example_input)

# 클래스 레이블 정의
labels = ["sad", "disgust", "angry", "neutral", "fear", "surprise", "happy"]

# Core ML 변환 (float 32)
coreml_model_test = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)],
    compute_precision=ct.precision.FLOAT32  # 정밀도 설정
)
coreml_model_test.save("coreml_model_test.mlpackage")

# Core ML 변환 (float 16)
coreml_model_test_float16 = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[ct.TensorType(shape=example_input.shape)],
    compute_precision=ct.precision.FLOAT16  # 정밀도 설정
)

coreml_model_test.save(float32_model_path)
coreml_model_test_float16.save(float16_model_path)

# 테스트용 이미지 경로
test_image_path = "test_image.png"

# 테스트 이미지 전처리
test_image = Image.open(test_image_path).convert("RGB")
test_image_resized = test_image.resize((224, 224))

# 이미지 전처리 파이프라인
transform = Compose([
    ToTensor(),  # [0, 255] -> [0, 1] 범위로 변환
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
])

# PyTorch 입력 데이터 생성
pytorch_input = transform(test_image_resized).unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

# PyTorch 모델로 추론
with torch.no_grad():
    pytorch_logits = pretrained_model(pytorch_input).logits.numpy().flatten()
    print("PyTorch Logits:", pytorch_logits)

# 소프트맥스 적용
pytorch_probs = softmax(pytorch_logits)

# PyTorch 결과 출력
print("\nPyTorch Model Prediction:")
for label, prob in zip(labels, pytorch_probs):
    print(f"{label}: {prob:.4f}")
print(f"Predicted Emotion (PyTorch): {labels[np.argmax(pytorch_probs)]}")

# Core ML 입력 데이터 생성
coreml_input = pytorch_input.squeeze().numpy()  # (1, C, H, W) -> (C, H, W)
coreml_input = np.expand_dims(coreml_input, axis=0)  # (C, H, W) -> (1, C, H, W)

# Core ML 모델로 추론

# FLOAT32 추론 시간 측정
start_time = time.time()

predictions = coreml_model_test.predict({"x_1": coreml_input})

float32_time = time.time() - start_time

# FLOAT16 추론 시간 측정
start_time = time.time()

predictions_float16 = coreml_model_test_float16.predict({"x_1": coreml_input})

float16_time = time.time() - start_time

float32_size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(float32_model_path) for f in filenames)
float16_size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(float16_model_path) for f in filenames)

# Core ML logits 출력 및 소프트맥스 적용
if "linear_72" in predictions:
    coreml_logits = predictions["linear_72"].flatten()
    print("\nCore ML (Float32) Logits:", coreml_logits)

    # 소프트맥스 적용
    coreml_probs = softmax(coreml_logits)
    print("\nCore ML Model (Float32) Probabilities (Softmax Applied):")
    for label, prob in zip(labels, coreml_probs):
        print(f"{label}: {prob:.4f}")
    print(f"\nPredicted Emotion (Core ML) (Float32): {labels[np.argmax(coreml_probs)]}")
else:
    print("Core ML logits key 'linear_72' not found.")

# Core ML logits 출력 및 소프트맥스 적용
if "linear_72" in predictions_float16:
    coreml_logits = predictions_float16["linear_72"].flatten()
    print("\nCore ML (Float16) Logits:", coreml_logits)

    # 소프트맥스 적용
    coreml_probs = softmax(coreml_logits)
    print("\nCore ML Model (Float16) Probabilities (Softmax Applied):")
    for label, prob in zip(labels, coreml_probs):
        print(f"{label}: {prob:.4f}")
    print(f"\nPredicted Emotion (Core ML) (Float16): {labels[np.argmax(coreml_probs)]}")
else:
    print("Core ML logits key 'linear_72' not found.")


print("=== MODEL COMPARISON LOG ===\n")
print(f"FLOAT32 Inference Time: {float32_time:.4f} seconds\n")
print(f"FLOAT16 Inference Time: {float16_time:.4f} seconds\n")
print(f"FLOAT32 Model Size: {float32_size / (1024 * 1024):.2f} MB\n")
print(f"FLOAT16 Model Size: {float16_size / (1024 * 1024):.2f} MB\n")

if float32_time > float16_time:
    print(f"FLOAT16 is {float32_time / float16_time:.2f}x faster than FLOAT32.\n")
else:
    print(f"FLOAT32 is {float16_time / float32_time:.2f}x faster than FLOAT16.\n")

if float32_size > float16_size:
    print(f"FLOAT16 is {float32_size / float16_size:.2f}x smaller than FLOAT32.\n")
else:
    print(f"FLOAT32 is {float16_size / float32_size:.2f}x smaller than FLOAT16.\n")
