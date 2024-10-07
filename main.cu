#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip> // Para std::setprecision
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

int main() {
    srand(time(0));

    // Solicitar o tamanho da matriz ao usuário
    int N;
    std::cout << "Digite o tamanho da matriz (NxN): ";
    std::cin >> N;

    size_t bytes = N * N * sizeof(float);

    // Alocação de memória para as matrizes no host (CPU)
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C_parallel = (float*)malloc(bytes); // Resultado Paralelo

    // Preencher as matrizes A e B com números aleatórios
    fillMatrix(h_A, N);
    fillMatrix(h_B, N);

    // Alocação de memória na GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copiar dados do host (CPU) para o dispositivo (GPU)
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Definir dimensões da grade e blocos para a execução na GPU
    int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Multiplicação Paralela
    auto start_parallel = std::chrono::high_resolution_clock::now();
    matrixMultiplyParallel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // Garantir que a execução esteja completa
    auto end_parallel = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_parallel = end_parallel - start_parallel;

    // Copiar o resultado de volta para o host (CPU)
    cudaMemcpy(h_C_parallel, d_C, bytes, cudaMemcpyDeviceToHost);

    // Imprimir os resultados em segundos com 6 casas decimais
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Tempo Paralelo: " << time_parallel.count() << " segundos\n";

    // Liberar memória
    free(h_A);
    free(h_B);
    free(h_C_parallel);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
