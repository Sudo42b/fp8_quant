# FP8 Quantization

## 소개

**8FP Quantization**은 딥러닝 모델의 연산 및 저장 효율을 극대화하기 위해 8비트 부동소수점(FP8) 형식으로 가중치와 활성값을 변환하는 양자화(Quantization) 기법입니다. 기존의 FP32, FP16에 비해 메모리 사용량과 연산량을 크게 줄이면서도, 모델의 정확도를 최대한 유지하는 것이 목표입니다.

## 8FP(8-bit Floating Point)란?

FP8은 8비트로 표현되는 부동소수점 데이터 타입입니다. 일반적으로 다음과 같이 구성됩니다:

- 1비트: 부호(Sign)
- 4비트: 지수(Exponent)
- 3비트: 가수(Mantissa)

이 구조는 다양한 값의 범위를 표현할 수 있으면서도, 8비트라는 작은 크기로 인해 메모리와 연산 효율이 매우 높습니다.

## 장점

- **메모리 절감**: FP32 대비 1/4, FP16 대비 1/2의 메모리 사용량
- **연산 속도 향상**: 하드웨어 지원 시 연산 속도 대폭 증가
- **에너지 효율**: 적은 비트수로 인한 전력 소모 감소
- **대규모 모델 배포 용이**: 경량화된 모델로 모바일, 엣지 디바이스 배포에 유리

## 사용 예시

```python
from fp8_quant import quantize, dequantize

# FP32 텐서를 FP8로 양자화
fp8_tensor = quantize(fp32_tensor)

# FP8 텐서를 다시 FP32로 복원
fp32_tensor = dequantize(fp8_tensor)
```

## 지원 환경

- Python 3.10 이상
- PyTorch 등 주요 딥러닝 프레임워크와 호환
- (선택) FP8 연산을 지원하는 GPU/가속기


## 참고 자료

- [NVIDIA FP8 Formats](https://developer.nvidia.com/blog/introducing-the-tensorfloat-32-precision-format/)
- [Google: 8-bit Floating Point Inference](https://arxiv.org/abs/2209.05433)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)

## 라이선스

MIT License