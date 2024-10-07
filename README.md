# Multiplicação de matrizes em CUDA

Exercício de multiplicação de duas matrizes quadradas de números de ponto flutuante em GPU. [main.cu](main.cu) preenche as entradas aleatoriamente na CPU, copia os dados para o dispositivo e lança blocos de `16 × 16` threads. Cada thread calcula um elemento da matriz resultante.

## Como executar

Use CMake 3.29 ou superior, CUDA Toolkit com suporte ao padrão CUDA C++20, compilador anfitrião compatível e uma GPU NVIDIA.

```bash
cmake -S . -B build
cmake --build build --config Release
```

Execute o alvo `untitled4` definido em [CMakeLists.txt](CMakeLists.txt). Com Visual Studio, procure `build/Release/untitled4.exe`. Informe uma dimensão inteira positiva, por exemplo `32`.

A saída mostra o tempo entre o lançamento e a sincronização do kernel. A alocação e as cópias entre CPU e GPU ficam fora desse intervalo. O programa copia a matriz resultante de volta à CPU e exibe o tempo no terminal.
