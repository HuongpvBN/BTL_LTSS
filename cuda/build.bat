@echo off
echo Compiling C files...
gcc -c main.c -O2
gcc -c funcs.c -O2

echo Compiling CUDA file...
nvcc -c cuda_convolute.cu

echo Linking...
nvcc -o cuda_conv main.o funcs.o cuda_convolute.o -O2

echo Done!
