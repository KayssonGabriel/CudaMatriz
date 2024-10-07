#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip> // Para std::setprecision
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>

// Função de multiplicação paralela na GPU
__global__ void matrixMultiplyParallel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

void fillMatrix(float* matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

void verificarCuda(cudaError_t status, const char* operacao) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operacao) + ": " + cudaGetErrorString(status));
    }
}

using MatrizCuda = std::unique_ptr<float, decltype(&cudaFree)>;

MatrizCuda alocarMatriz(size_t bytes) {
    float* matriz = nullptr;
    verificarCuda(cudaMalloc(&matriz, bytes), "alocar matriz CUDA");
    return MatrizCuda(matriz, cudaFree);
}

int main() try {
    srand(time(nullptr));

    // Solicitar o tamanho da matriz ao usuário
    int N;
    std::cout << "Digite o tamanho da matriz (NxN): ";
    if (!(std::cin >> N) || N <= 0
            || N > std::numeric_limits<int>::max() / N) {
        throw std::invalid_argument("Informe uma dimensão inteira positiva compatível com os índices da matriz.");
    }

    size_t elementos = static_cast<size_t>(N) * N;
    size_t bytes = elementos * sizeof(float);

    std::vector<float> h_A(elementos);
    std::vector<float> h_B(elementos);
    std::vector<float> h_C_parallel(elementos);
    fillMatrix(h_A.data(), N);
    fillMatrix(h_B.data(), N);

    auto d_A = alocarMatriz(bytes);
    auto d_B = alocarMatriz(bytes);
    auto d_C = alocarMatriz(bytes);
    verificarCuda(cudaMemcpy(d_A.get(), h_A.data(), bytes, cudaMemcpyHostToDevice), "enviar matriz A");
    verificarCuda(cudaMemcpy(d_B.get(), h_B.data(), bytes, cudaMemcpyHostToDevice), "enviar matriz B");

    // Definir dimensões da grade e blocos para a execução na GPU
    int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Multiplicação Paralela
    auto start_parallel = std::chrono::high_resolution_clock::now();
    matrixMultiplyParallel<<<blocks, threads>>>(d_A.get(), d_B.get(), d_C.get(), N);
    verificarCuda(cudaGetLastError(), "lançar multiplicação CUDA");
    verificarCuda(cudaDeviceSynchronize(), "executar multiplicação CUDA");
    auto end_parallel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_parallel = end_parallel - start_parallel;

    // Copiar o resultado de volta para o host (CPU)
    verificarCuda(cudaMemcpy(h_C_parallel.data(), d_C.get(), bytes, cudaMemcpyDeviceToHost), "receber matriz C");

    // Imprimir os resultados em segundos com 6 casas decimais
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Tempo Paralelo: " << time_parallel.count() << " segundos\n";

    return 0;
} catch (const std::exception& erro) {
    std::cerr << erro.what() << std::endl;
    return 1;
}
