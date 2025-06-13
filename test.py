import numpy as np
import matplotlib.pyplot as plt

# FP8 E4M3 format 파라미터 (NVIDIA H100에서 사용)
FP8_E4M3_MAX = 448.0  # 최대 표현 가능한 값
FP8_E4M3_MIN = -448.0  # 최소 표현 가능한 값

def fp8_e4m3_quantize(x, scale):
    """FP8 E4M3 양자화 시뮬레이션"""
    # 스케일링
    x_scaled = x / scale
    
    # FP8 범위로 클리핑
    x_clipped = np.clip(x_scaled, FP8_E4M3_MIN, FP8_E4M3_MAX)
    
    # 실제로는 하드웨어에서 FP8로 저장되지만, 여기서는 시뮬레이션
    # 다시 원래 스케일로 복원
    return x_clipped * scale

def per_channel_quantization(weight_matrix, axis=0):
    """채널별 양자화 수행"""
    # 각 채널별로 최대 절댓값 계산
    max_vals = np.max(np.abs(weight_matrix), axis=axis, keepdims=True)
    
    # 각 채널별 스케일링 팩터 계산
    scales = max_vals / FP8_E4M3_MAX
    scales = np.maximum(scales, 1e-8)  # 0으로 나누기 방지
    
    # 채널별로 양자화 적용
    quantized_weight = fp8_e4m3_quantize(weight_matrix, scales)
    
    return quantized_weight, scales

def tensor_wise_quantization(weight_matrix):
    """텐서 전체 양자화 (비교용)"""
    max_val = np.max(np.abs(weight_matrix))
    scale = max_val / FP8_E4M3_MAX
    scale = max(scale, 1e-8)
    
    quantized_weight = fp8_e4m3_quantize(weight_matrix, scale)
    return quantized_weight, scale

# 예제 가중치 행렬 생성 (다양한 채널별 분포를 가지도록)
np.random.seed(42)
weight_shape = (128, 256)  # (출력 채널, 입력 채널)
weights = np.random.randn(*weight_shape)

# 일부 채널에 더 큰 값들을 추가하여 채널별 분포 차이 만들기
weights[:32, :] *= 3.0   # 첫 32개 채널은 더 큰 값
weights[32:64, :] *= 0.5  # 다음 32개 채널은 더 작은 값
weights[64:96, :] *= 2.0  # 다음 32개 채널은 중간 값

print("=== FP8 채널별 양자화 vs 텐서 전체 양자화 비교 ===\n")

print(f"원본 가중치 행렬 형태: {weights.shape}")
print(f"원본 가중치 범위: [{weights.min():.3f}, {weights.max():.3f}]")
print(f"원본 가중치 표준편차: {weights.std():.3f}\n")

# 채널별 양자화 (출력 채널 기준)
quant_per_channel, scales_per_channel = per_channel_quantization(weights, axis=1)
per_channel_error = np.mean((weights - quant_per_channel) ** 2)

print("=== 채널별 양자화 결과 ===")
print(f"스케일링 팩터 개수: {len(scales_per_channel)}")
print(f"스케일링 팩터 범위: [{scales_per_channel.min():.6f}, {scales_per_channel.max():.6f}]")
print(f"양자화 오차 (MSE): {per_channel_error:.6f}")

# 텐서 전체 양자화
quant_tensor_wise, scale_tensor_wise = tensor_wise_quantization(weights)
tensor_wise_error = np.mean((weights - quant_tensor_wise) ** 2)

print("\n=== 텐서 전체 양자화 결과 ===")
print(f"스케일링 팩터: {scale_tensor_wise:.6f}")
print(f"양자화 오차 (MSE): {tensor_wise_error:.6f}")

print(f"\n=== 성능 비교 ===")
print(f"채널별 양자화 오차: {per_channel_error:.6f}")
print(f"텐서 전체 양자화 오차: {tensor_wise_error:.6f}")
print(f"오차 개선률: {(tensor_wise_error - per_channel_error) / tensor_wise_error * 100:.2f}%")

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 원본 가중치 분포
axes[0, 0].hist(weights.flatten(), bins=50, alpha=0.7, color='blue')
axes[0, 0].set_title('원본 가중치 분포')
axes[0, 0].set_xlabel('값')
axes[0, 0].set_ylabel('빈도')

# 채널별 스케일링 팩터
axes[0, 1].plot(scales_per_channel.flatten())
axes[0, 1].set_title('채널별 스케일링 팩터')
axes[0, 1].set_xlabel('채널 인덱스')
axes[0, 1].set_ylabel('스케일링 팩터')

# 채널별 양자화 결과
axes[0, 2].hist(quant_per_channel.flatten(), bins=50, alpha=0.7, color='green')
axes[0, 2].set_title('채널별 양자화 결과')
axes[0, 2].set_xlabel('값')
axes[0, 2].set_ylabel('빈도')

# 텐서 전체 양자화 결과
axes[1, 0].hist(quant_tensor_wise.flatten(), bins=50, alpha=0.7, color='red')
axes[1, 0].set_title('텐서 전체 양자화 결과')
axes[1, 0].set_xlabel('값')
axes[1, 0].set_ylabel('빈도')

# 오차 비교 (채널별)
channel_errors_per = np.mean((weights - quant_per_channel) ** 2, axis=1)
channel_errors_tensor = np.mean((weights - quant_tensor_wise) ** 2, axis=1)

axes[1, 1].plot(channel_errors_per, label='채널별 양자화', color='green')
axes[1, 1].plot(channel_errors_tensor, label='텐서 전체 양자화', color='red')
axes[1, 1].set_title('채널별 양자화 오차 비교')
axes[1, 1].set_xlabel('채널 인덱스')
axes[1, 1].set_ylabel('MSE')
axes[1, 1].legend()

# 채널별 가중치 범위
channel_ranges = np.max(np.abs(weights), axis=1)
axes[1, 2].plot(channel_ranges)
axes[1, 2].set_title('채널별 가중치 최대 절댓값')
axes[1, 2].set_xlabel('채널 인덱스')
axes[1, 2].set_ylabel('최대 절댓값')

plt.tight_layout()
plt.show()

# 구체적인 채널 몇 개의 예시
print("\n=== 개별 채널 예시 ===")
for i in [0, 32, 64, 96]:
    original_range = np.max(np.abs(weights[i, :]))
    scale = scales_per_channel[i, 0]
    quantized_range = np.max(np.abs(quant_per_channel[i, :]))
    error = np.mean((weights[i, :] - quant_per_channel[i, :]) ** 2)
    
    print(f"채널 {i:2d}: 원본 범위={original_range:6.3f}, "
          f"스케일={scale:8.6f}, 양자화 후 범위={quantized_range:6.3f}, "
          f"MSE={error:8.6f}")

print("\n=== 채널별 양자화의 장점 ===")
print("1. 각 채널의 동적 범위를 최대한 활용")
print("2. 채널별로 다른 분포 특성에 적응")
print("3. 전체적인 양자화 오차 감소")
print("4. 모델 정확도 유지에 유리")

print("\n=== 메모리 오버헤드 ===")
print(f"채널별 스케일링 팩터: {len(scales_per_channel)} 개")
print(f"텐서 전체 스케일링 팩터: 1 개")
print(f"추가 메모리: {len(scales_per_channel) * 4} bytes (float32 기준)")