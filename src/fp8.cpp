#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iostream>
#include <random>
#include <cassert>
#include <fstream>
using namespace std;

// Type
#define TYPE float_type::E4M3
// Matrix size
constexpr size_t M = 2;
constexpr size_t N = 4;
constexpr size_t K = 3;
constexpr float eps = 1e-12;
// float_8 상수들
constexpr float float_8_E4M3_MAX = 448.0f;
constexpr float float_8_E5M2_MAX = 57344.0f;

// float_8 타입 정의
enum class float_type {
    E4M3,  // 4비트 지수, 3비트 가수
    E5M2   // 5비트 지수, 2비트 가수
};

// float_8 값을 저장하는 구조체
struct float_8 {
    uint8_t data;

    float_8(uint8_t d = 0) : data(d) {}
};
typedef float_8 float_8;
typedef float float_32;

template<typename T>
    std::vector<T> generate_random_data(const float_type type, const int row, const int col) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dist(type == float_type::E4M3 ? -float_8_E4M3_MAX : -float_8_E5M2_MAX, 
                                        type == float_type::E4M3 ? float_8_E4M3_MAX : float_8_E5M2_MAX);
    std::vector<T> data(row * col);
    for (int i = 0; i < row * col; ++i) {
        data[i] = dist(gen);
    }
    return data;
}


template<typename T>
T calculate_channel_scale(const vector<T>& data, 
                        const vector<size_t>& shape, 
                        float_type type) {
    T scale = (type == float_type::E4M3 ? -float_8_E4M3_MAX : -float_8_E5M2_MAX); // m x n
    for (size_t i = 0; i < shape[0]; ++i) { // m
        for (size_t j = 0; j < shape[1]; ++j) { // n 
            T val = data[i * shape[1] + j];
            scale = std::max(scale, std::abs(val));
        }   
    }
    return scale; //채널별 하나
}

// 텐서별 스케일 계산
template<typename T>
T calculate_tensor_scale(const vector<T>& data,      
                        const vector<size_t>& shape,
                        float_type type) {
    vector<T> scales(shape[1]*shape[0], type == float_type::E4M3 ? -float_8_E4M3_MAX : -float_8_E5M2_MAX); // m x n
    
    for (size_t c = 0; c < shape[1]; ++c) { // n
        T min_val = type == float_type::E4M3 ? float_8_E4M3_MAX : float_8_E5M2_MAX;
        T max_val = type == float_type::E4M3 ? -float_8_E4M3_MAX : -float_8_E5M2_MAX;
        
        // 해당 채널의 모든 요소에 대해 최소/최대값 찾기
        for (size_t i = 0; i < shape[0]; ++i) {
            T val = data[i * channels + c];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        
        scales[c] = (type == float_type::E4M3) ? float_8_E4M3_MAX : float_8_E5M2_MAX / std::max(max_val - min_val, static_cast<T>(eps));
    }
    return scales;
}

template<typename T>
void save_result(const string& input_name= "input_data.txt",
                const std::vector<T>& input_data, 
                const string& weight_name= "weight_data.txt",
                const std::vector<T>& weight_data, 
                const string& output_name= "output_data.txt",
                const std::vector<T>& output_data) {

    auto input_shape = input_data.shape();
    auto weight_shape = weight_data.shape();
    auto output_shape = output_data.shape();

    // 결과를 파일로 저장
    std::ofstream input_file(input_name);
    input_file << "Input Shape: " << input_shape[0] << " " << input_shape[1] << "\n";
    for (size_t i = 0; i < input_shape[0]; ++i) {
        for (size_t j = 0; j < input_shape[1]; ++j) {
            input_file << input_data[i * input_shape[1] + j] << " ";
        }
        input_file << "\n";
    }
    input_file.close();

    std::ofstream weight_file(weight_name);
    weight_file << "Weight Shape: " << weight_shape[0] << " " << weight_shape[1] << "\n";
    for (size_t i = 0; i < weight_shape[0]; ++i) {
        for (size_t j = 0; j < weight_shape[1]; ++j) {
            weight_file << weight_data[i * weight_shape[1] + j] << " ";
        }
        weight_file << "\n";
    }
    weight_file.close();

    std::ofstream output_file(output_name);
    output_file << "Output Shape: " << M << " " << K << "\n";
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            output_file << dynamic_output[i * K + j] << " ";
        }
        output_file << "\n";
    }
    
    output_file.close();
}


auto quantized_gemm(const vector<float_32>& A, 
                    const vector<float_32>& B, 
                    const vector<size_t>& A_shape, 
                    const vector<size_t>& B_shape,
                    float_type type, 
                    bool use_channel_wise = false) {

    if (use_channel_wise){

        auto A_fp8 = calculate_channel_scale(A, A_shape, type);
        auto B_fp8 = calculate_channel_scale(B, B_shape, type);
    }
    else{
        auto A_fp8 = calculate_tensor_scale(A, A_shape, type);
        auto B_fp8 = calculate_tensor_scale(B, B_shape, type);
    }
    float_32 C_scale = A_fp8 * B_fp8;
    
}
int main() {
    auto input_data = generate_random_data<float_32>(TYPE, M, N);
    auto weight_data = generate_random_data<float_32>(TYPE, N, K);
    
    quantized_gemm(input_data, weight_data, {M, N}, {N, K}, TYPE, false);
    
    // 채널별 스케일링 사용
    auto channel_output = quantized_gemm(input_data, weight_data, {M, N}, {N, K}, TYPE, true);
    
    // 결과 저장
    save_result("input_data.txt", input_data, "weight_data.txt", weight_data, "tensor_output.txt", tensor_output);


    return 0;
}