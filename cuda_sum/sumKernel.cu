#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <ctime>
#include <iostream>
#include <vector>
#include "inc/helper_cuda.h"
using namespace std;

const size_t N = 5e8;       //数组长度


__global__ void d_ParallelTest(float *Para)
{
	size_t tid = threadIdx.x;
	//----随循环次数的增加，stride逐次翻倍（乘以2）-----------------------------------------------------
	for (size_t stride = 1; stride < blockDim.x; stride *= 2)
	{
		if (tid % (2 * stride) == 0)
		{
			Para[tid] += Para[tid + stride];
		}
		cudaDeviceSynchronize();
	}

}


// this kernel computes, per-block, the sum
// of a block-sized portion of the input
// using a block-wide reduction
__global__ void block_sum(float *input,
	float *per_block_results,
	const size_t n)
{
	extern __shared__ float sdata[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// load input size_to __shared__ memory //一个线程负责把一个元素从全局内存载入到共享内存
	float x = 0;
		
	if (i < n)
	{
		x = input[i];
		//if(n == 1954)
		//	printf("%.5f\t%d\n", x, i);
	}
	sdata[threadIdx.x] = x;
	__syncthreads(); //等待所有线程把自己负责的元素载入到共享内存

							 // contiguous range pattern//块内进行合并操作，每次合并变为一半.注意threadIdx.x是块内的偏移，上面算出的i是全局的偏移。
	for (int offset = blockDim.x / 2;
		offset > 0;
		offset >>= 1)
	{
		if (threadIdx.x < offset)//控制只有某些线程才进行操作
		{
			// add a partial sum upstream to our own
			sdata[threadIdx.x] += sdata[threadIdx.x + offset];
		}
		// wait until all threads in the block have
		// updated their partial sums
		__syncthreads();
	}
	// thread 0 writes the final result//每个块的线程0负责存放块内求和的结果
	if (threadIdx.x == 0)
	{
		per_block_results[blockIdx.x] = sdata[0];
		//printf("%.5f\n", per_block_results[blockIdx.x]);
	}
}


extern "C"
void ParallelTest()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	for (size_t i = 0; i < deviceCount; i++)
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		std::cout << "使用GPU device " << i << ": " << devProp.name << std::endl;
		std::cout << "设备全局内存总量： " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
		std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
		std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
		std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
		std::cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << devProp.regsPerBlock << std::endl;
		std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
		std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
		std::cout << "设备上多处理器的数量： " << devProp.multiProcessorCount << std::endl;
		std::cout << "======================================================" << std::endl;
	}

	float *pcpu = new float[N];
	for (size_t i = 0; i<N; i++)
	{
		pcpu[i] = 1.0;	//数组赋值
	}
	//CPU
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	double ParaSum = 0;
	for (size_t i = 0; i<N; i++)
	{
		ParaSum += pcpu[i];	//CPU端数组累加
	}
	endTime = clock();//计时结束
	cout << "CPU run time is: " << (float)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout << "CPU result = " << ParaSum << endl;	//显示CPU端结果

	//GPU
	startTime = clock();//计时开始
	float *d_input;
	checkCudaErrors(cudaMalloc((void **)&d_input, sizeof(float) * N));
	checkCudaErrors(cudaMemcpy(d_input, pcpu, N * sizeof(float), cudaMemcpyHostToDevice));
	delete[]pcpu;

	size_t block_size = 1024;//线程块的大小。目前有些gpu的线程块最大为512，有些为1024.
	size_t num_blocks = (N / block_size) + ((N%block_size) ? 1 : 0);
	size_t it = 0, len = num_blocks, plusBlocks = 0;
	vector<size_t> vBlocks, vN;
	while (len > block_size){
		++it;
		vN.push_back(len);
		len = ((len / block_size) + ((len%block_size) ? 1 : 0));
		plusBlocks += len;
		vBlocks.push_back(len);
	}
	
	
	// allocate space to hold one partial sum per block, plus one additional
	// slot to store the total sum
	float *d_partial_sums_and_total;//一个线程块一个和，另外加一个元素，存放所有线程块的和。
	checkCudaErrors(cudaMalloc((void**)&d_partial_sums_and_total, sizeof(float) * (num_blocks + 1 + plusBlocks)));

	// launch one kernel to compute, per-block, a partial sum//把每个线程块的和求出来
	block_sum << <num_blocks, block_size, block_size * sizeof(float) >> >(d_input, d_partial_sums_and_total, N);
	cudaFree(d_input);

	// launch a single block to compute the sum of the partial sums
	size_t num = vN.size();
	float *start = d_partial_sums_and_total, *end = d_partial_sums_and_total + num_blocks;
	while (it){
		size_t blocks = vBlocks[num - it];
		size_t n = vN[num - it];
		block_sum << <blocks, block_size, block_size * sizeof(float) >> >(start, end, n);

		start = end;
		end = end + blocks;
		--it;
	}

	size_t new_blocks = pow(2, ceil(log2(vBlocks[vBlocks.size() - 1])));
	block_sum << <1, new_blocks, new_blocks * sizeof(float) >> >(start, end, vBlocks[vBlocks.size()-1]);

	float d_ParaSum;
	checkCudaErrors(cudaMemcpy(&d_ParaSum, d_partial_sums_and_total + num_blocks + plusBlocks, sizeof(float), cudaMemcpyDeviceToHost));	//从累加过后数组的0号元素得出结果
	cudaFree(d_partial_sums_and_total);

	endTime = clock();//计时结束
	cout << "GPU run time is: " << (float)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout << "GPU result = " << d_ParaSum << endl;	//显示GPU端结果
	
}