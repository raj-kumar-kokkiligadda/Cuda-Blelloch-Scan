#include <cuda_runtime.h>
#include <helper_cuda.h>
#define NUM_BANKS 32  
#define LOG_NUM_BANKS 4  
#define CONFLICT_FREE_OFFSET(n) 0
//((n) / NUM_BANKS)
#define BLOCKWIDTH  512

#define VERIFY(err)             \
do                              \
{                               \
    if (err != cudaSuccess)     \
    {                           \
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));\
        exit(EXIT_FAILURE);     \
    }                           \
}while(0)

template<typename T>
struct SharedMemory
{
    __device__ inline operator T *()
    {
        __shared__ T __smem[1024*2];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        __shared__ T __smem[1024*2];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator double *()
    {
        __shared__ double __smem_d[1024*2];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        __shared__ double __smem_d[1024*2];
        return (double *)__smem_d;
    }
};

template<typename T>
__device__ void ReduceStep1(const unsigned int& tid, const unsigned int& roundSize, T* input, T* auxArry)
{
    unsigned int offset = 1;
    for(unsigned int i = roundSize >> 1; i > 0; i = i >> 1, offset = offset << 1)              //level of the tree
    {       
        if(tid < i)
        {
            unsigned int index1 = offset*(2*tid+1) - 1;  //The stride between 2 index will be 1, 2, 4, 6, 8, .... will cause bank-conflict
            unsigned int index2 = offset*(2*tid+2) - 1;
            input[index2 + CONFLICT_FREE_OFFSET(index2)] = input[index1 + CONFLICT_FREE_OFFSET(index1)] + 
                                                           input[index2 + CONFLICT_FREE_OFFSET(index2)];
        }
        __syncthreads();   
    }
}

template<typename T>
__device__ void DownsweepStep2(const unsigned int& tid,const unsigned int& blockId, const unsigned int& size, T* input, T* auxArry)
{
    unsigned int offset = size / 2;
    for(unsigned int i = 1; i < size; i = i << 1, offset = offset >> 1)
    {
        if(tid < i)
        {
            if(i == 1)
            {
                auxArry[blockId] = input[size -1 + CONFLICT_FREE_OFFSET(size -1)];  //save sum reduce result before clear it to 0
                //printf("auxArry[%d] = %d\n", blockId, auxArry[blockId]);
                input[size -1 + CONFLICT_FREE_OFFSET(size -1)] = 0;
            }
            unsigned int index1 = offset*(2*tid + 1) - 1;
            unsigned int index2 = offset*(2*tid + 2) - 1;
            T temp = input[index2 + CONFLICT_FREE_OFFSET(index2)];
            input[index2 + CONFLICT_FREE_OFFSET(index2)] = input[index1 + CONFLICT_FREE_OFFSET(index1)] + input[index2 + CONFLICT_FREE_OFFSET(index2)];
            input[index1 + CONFLICT_FREE_OFFSET(index1)] = temp;
        }
        __syncthreads(); 
    }
}


template<typename T>
__global__ void BlellochScan(T* input, T* output, T* auxArry, const unsigned int roundSize)
{
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int globalIdx = bid * blockDim.x + tid;
    

    T* sData = SharedMemory<T>();
    unsigned int index1 = 2*tid;
    unsigned int index2 = 2*tid + 1;
    sData[index1 + CONFLICT_FREE_OFFSET(index1) ] = input[2*globalIdx];                //copy data to share memory
    sData[index2 + CONFLICT_FREE_OFFSET(index2) ] = input[2*globalIdx+1];              //copy data to share memory

    ReduceStep1(tid, roundSize, sData, auxArry);                                        //leaf to root sum reduce
    DownsweepStep2(tid, bid, roundSize, sData, auxArry);                         //root to leaf down sweep

    output[2*globalIdx] = sData[index1 + CONFLICT_FREE_OFFSET(index1)];                 //copy share memory to global memory
    output[2*globalIdx+1] = sData[index2 + CONFLICT_FREE_OFFSET(index2)];

}
template<typename T>
__global__ void Add(T* inout, T* add)
{
    unsigned int bid = blockIdx.y *gridDim.x + blockIdx.x;
    unsigned int tid = bid * blockDim.x + threadIdx.x;
    inout[tid] = inout[tid] + add[bid];
}


template <typename T>
void RunKernel1(T* pDevInput, T* pDevOutput, const unsigned int& roundSize)
{
    if(roundSize<=1024)         //if data size <= 1024, then 1 block can deal with it
    {
        T* pDummy;
        cudaMalloc(&pDummy,roundSize*sizeof(T));
        //unsigned int blockSize = roundSize/(BLOCKWIDTH*2);

        unsigned int threadNum = roundSize/2;//(roundSize/2) > BLOCKWIDTH ? BLOCKWIDTH : roundSize/2;
        BlellochScan<<<1, threadNum>>>(pDevInput, pDevOutput, pDummy, threadNum*2);   //Do scan for aux array

        cudaFree(pDummy);
        return;
    }

    unsigned int blockSize = roundSize/(BLOCKWIDTH*2);
    //dim3 blockDims(32768,1,1);
    dim3 blockDims;
    if(blockSize >= 65536)
    {
        blockDims.x = 32768;
        blockDims.y = blockSize/32768;
        blockDims.z = 1;
    }  
    else
    {
        blockDims.x = blockSize;
    }
    unsigned int threadNum = BLOCKWIDTH;//(roundSize/2) > BLOCKWIDTH ? BLOCKWIDTH : roundSize/2;

    T* pAuxArray;   //for each block, save the sum reduce result
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&pAuxArray,blockSize*sizeof(T));
    VERIFY(err);
    BlellochScan<<<blockDims, threadNum>>>(pDevInput, pDevOutput, pAuxArray, threadNum*2);  //each block deal with 1024 data

    T* pAuxScan;
    err = cudaMalloc(&pAuxScan,blockSize*sizeof(T));

    RunKernel1(pAuxArray,pAuxScan,blockSize);   //deal with Aux Array data, since this data has size bigger than 1024, than



    Add<<<blockDims, threadNum*2>>>(pDevOutput,pAuxScan);       //each block deal with 1024 data


    cudaFree(pAuxArray);
    cudaFree(pAuxScan);
    
}

template <typename T>
void RunKernel0(T* pDevInput, T* pDevOutput, const unsigned int& roundSize)
{
    printf("array length = %d\n", roundSize);

    unsigned int blockSize = roundSize/(BLOCKWIDTH*2);
    unsigned int threadNum = BLOCKWIDTH;//(roundSize/2) > BLOCKWIDTH ? BLOCKWIDTH : roundSize/2;

    T* pAuxArray;   //for each block, save the sum reduce result
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&pAuxArray,blockSize*sizeof(T));
    VERIFY(err);

    BlellochScan<<<blockSize, threadNum>>>(pDevInput, pDevOutput, pAuxArray, threadNum*2);  //each block deal with 1024 data

    T* pAuxScan;
    err = cudaMalloc(&pAuxScan,blockSize*sizeof(T));

    T* pDummy;
    err = cudaMalloc(&pDummy,blockSize*sizeof(T));

    BlellochScan<<<1, blockSize/2>>>(pAuxArray, pAuxScan, pDummy, blockSize);   //Do scan for aux array

    Add<<<blockSize, threadNum*2>>>(pDevOutput,pAuxScan);       //each block deal with 1024 data

    cudaFree(pAuxArray);
    cudaFree(pAuxScan);
    cudaFree(pDummy);
}

template 
void RunKernel0<double>(double* pDevInput, double* pDevOutput, const unsigned int& roundSize);

template 
void RunKernel0<float>(float* pDevInput, float* pDevOutput, const unsigned int& roundSize);

template 
void RunKernel0<int>(int* pDevInput, int* pDevOutput, const unsigned int& roundSize);
template 
void RunKernel0<unsigned int>(unsigned int* pDevInput, unsigned int* pDevOutput, const unsigned int& roundSize);

template 
void RunKernel1<double>(double* pDevInput, double* pDevOutput, const unsigned int& roundSize);

template 
void RunKernel1<float>(float* pDevInput, float* pDevOutput, const unsigned int& roundSize);

template 
void RunKernel1<int>(int* pDevInput, int* pDevOutput, const unsigned int& roundSize);

template 
void RunKernel1<unsigned int>(unsigned int* pDevInput, unsigned int* pDevOutput, const unsigned int& roundSize);

template 
void RunKernel1<unsigned long>(unsigned long* pDevInput, unsigned long* pDevOutput, const unsigned int& roundSize);