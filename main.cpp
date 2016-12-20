#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <random>
#include "main.h"

#define VERIFY(err)             \
do                              \
{                               \
    if (err != cudaSuccess)     \
    {                           \
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));\
        exit(EXIT_FAILURE);     \
    }                           \
}while(0)

extern unsigned int NextPow2(const unsigned int& x);
extern unsigned char log2(const unsigned int& x);


template <typename T>
void CPUReference(T* output, const unsigned int& size)
{
    StopWatchInterface *hTimer = 0;
    double timer = 0;
    sdkCreateTimer(&hTimer);

    sdkStartTimer(&hTimer);

    unsigned long offset = 1;
    for(unsigned long i = size >> 1; i > 0; i = i >> 1, offset = offset << 1)              //level of the tree
    {       
        for(unsigned long tid = 0; tid < size/2; tid++)
        {
            if(tid < i)
            {
                unsigned int index1 = offset*(2*tid+1) - 1;
                unsigned int index2 = offset*(2*tid+2) - 1;
                output[index2] = output[index1] + output[index2];
            }
        }
    }

    offset = size / 2;
    for(unsigned long i = 1; i < size; i = i << 1, offset = offset >> 1)
    {
        for(unsigned long tid = 0; tid < size/2; tid++)
        {
            if(tid < i)
            {
                if(i == 1)
                {
                    output[size -1] = 0;
                }
                unsigned int index1 = offset*(2*tid + 1) - 1;
                unsigned int index2 = offset*(2*tid + 2) - 1;
                T temp = output[index2];
                output[index2] = output[index1] + output[index2];
                output[index1] = temp;
            }
        }
    }
    sdkStopTimer(&hTimer);
    timer = sdkGetTimerValue(&hTimer);
    printf("CPU time = %.8f ms\r\n",timer);
}


template <typename T, bool IsFloat = std::is_floating_point<T>::value >
struct GenerateInput
{
    static void Generate(T* input, const unsigned int& size)
    {}
};

template <typename T >
struct GenerateInput<T,true>
{
    static void Generate(T* input, const unsigned int& size)
    {
        std::mt19937 rng( std::time(NULL));
        std::normal_distribution<T> normal( 0, 1.0 );
        for(unsigned int i = 0; i < size; i++)
        {
            input[i] = static_cast<T>(normal(rng));
        }
    }
};

template <typename T >
struct GenerateInput<T,false>
{
    static void Generate(T* input, const unsigned int& size)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> uniform(1, 6);
 
        for (unsigned int i=0; i<size; i++)
            input[i] = static_cast<T>(uniform(gen));
    }
};

template <typename T>
bool ResultCheck(const T* gpuResult, const T* cpuResult, const unsigned int& size)
{
    for(unsigned int i = 0; i < size; i++)
    {
        if(!(std::is_integral<T>::value))
        {
            if(abs(gpuResult[i] - cpuResult[i]) > 1e-6)
            {
                printf("gpuResult[%d] = %f, cpuResult[%d] = %f\n", i, gpuResult[i], i, cpuResult[i]);
                return false;
            }
        }
        else
        {
            if(gpuResult[i] != cpuResult[i])
            {
                printf("gpuResult[%d] = %u, cpuResult[%d] = %u\n", i, gpuResult[i], i, cpuResult[i]);
                return false;
            }
        }
    }
    return true;
}

template <typename T>
void SetupKernel(const unsigned int& inputSize)
{
    unsigned int roundSize = NextPow2(inputSize);
    printf("array size = %d \n",roundSize );
    T* pHostInput = (T*)malloc(roundSize*sizeof(T));
    T* pHostOutput = (T*)malloc(roundSize*sizeof(T));
    
    GenerateInput<T>::Generate(pHostInput, inputSize);

    T* pDevInput;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&pDevInput,roundSize*sizeof(T));
    VERIFY(err);

    err = cudaMemcpy(pDevInput, pHostInput, roundSize*sizeof(T), cudaMemcpyHostToDevice);
    VERIFY(err);

    T* pDevOutput;
    err = cudaMalloc(&pDevOutput,roundSize*sizeof(T));
    VERIFY(err);

    //warm up
    RunKernel1<T>(pDevInput,pDevOutput,roundSize);

    StopWatchInterface *hTimer = 0;
    double timer = 0;
    sdkCreateTimer(&hTimer);

    sdkStartTimer(&hTimer);

    RunKernel1<T>(pDevInput,pDevOutput,roundSize);
    cudaDeviceSynchronize();

    sdkStopTimer(&hTimer);

    cudaMemcpy(pHostOutput, pDevOutput, roundSize*sizeof(T), cudaMemcpyDeviceToHost);

    timer = sdkGetTimerValue(&hTimer);
    printf("GPU time = %.8f ms\r\n",timer);

    CPUReference(pHostInput, roundSize);


    if(ResultCheck<T>(pHostOutput, pHostInput, inputSize))
    {
        printf("Pass\n");
    }
    else
    {
        printf("Failed\n");
    }

    cudaFree(pDevInput);
    cudaFree(pDevOutput);

    free(pHostInput);
    free(pHostOutput);
    sdkDeleteTimer(&hTimer);
}


int main(int argc, char* argv[])
{
    SetupKernel<float>(1024*1024*128);
    return 1;
}