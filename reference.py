# Example code here: https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_w8a8_fp8

import torch
e4m3_type = torch.float8_e4m3fn
e5m2_type = torch.float8_e5m2


def generate_input(m: int, n: int, k: int, seed: int):
    """
    Generate random input and weights for Blockwise W8A8 Matmul scaled to FP32.
    
    Returns:
        Tuple of (
            a: torch.Tensor[float8_e4m3fnuz] of shape [m, k], 
            b: torch.Tensor[float8_e4m3fnuz] of shape [n, k], 
            a_scale: torch.Tensor[float32] of shape [m, k // 128], 
            b_scale: torch.Tensor[float32] of shape [n // 128, k // 128], 
            c: torch.Tensor[bfloat16] of shape [m, n]
        )
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)

    # Generate random inputs with FP8 quantization
    a = (torch.randn((m, k), dtype=torch.float16, device="cuda", generator=gen)).to(e4m3_type)
    b = (torch.randn((k, n), dtype=torch.float16, device="cuda", generator=gen)).to(e4m3_type)

    # Generate scaling factors with FP32
    a_scale = torch.randn([m, k], dtype=torch.float16, device="cuda", generator=gen)
    b_scale = torch.randn([k, n], dtype=torch.float16, device="cuda", generator=gen)

    c = torch.zeros((m, n), dtype=torch.float16, device="cuda")
    
    # Dequantize 'a', in your implementation you should do this at the end.
    a = a.to(a_scale.dtype) * a_scale 
    # Dequantize 'b', in your implementation you should do this at the end.
    b = b.to(b_scale.dtype) * b_scale

    # Compute FP8 GEMM and write to 'c'. 
    c = (a @ b).to(e4m3_type)
    
    return c

print(generate_input(1024, 1024, 1024, 0))
