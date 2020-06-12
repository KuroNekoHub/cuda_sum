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

	float *in = new float[N];
	for (size_t i = 0; i<N; i++)
	{
		in[i] = 1.0;	//数组赋值
	}
	//CPU
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	double ParaSum = 0;
	for (size_t i = 0; i<N; i++)
	{
		ParaSum += in[i];	//CPU端数组累加
	}
	endTime = clock();//计时结束
	cout << "CPU run time is: " << (float)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout << "CPU result = " << ParaSum << endl;	//显示CPU端结果

	//GPU
	startTime = clock();//计时开始

	size_t block_size = 1024;//线程块的大小。目前有些gpu的线程块最大为512，有些为1024.
	size_t num_blocks = N;
	size_t it = 0, plusBlocks = 0;
	vector<size_t> vBlocks, vLen, vBlockSize;
	while (num_blocks > block_size){
		++it;
		vLen.push_back(num_blocks);
		num_blocks = ((num_blocks / block_size) + ((num_blocks%block_size) ? 1 : 0));
		plusBlocks += num_blocks;
		vBlocks.push_back(num_blocks);
		vBlockSize.push_back(block_size);
	}
	//final operation
	vBlocks.push_back(1);
	vLen.push_back(num_blocks);
	++it;
	vBlockSize.push_back(pow(2, ceil(log2(vBlocks[vBlocks.size() - 2]))));

	// allocate space to hold one partial sum per block, plus one additional
	// slot to store the total sum
	float *d_partial_sums_and_total;//一个线程块一个和，另外加一个元素，存放所有线程块的和。
	checkCudaErrors(cudaMalloc((void**)&d_partial_sums_and_total, sizeof(float) * (N + plusBlocks + 1)));
	checkCudaErrors(cudaMemcpy(d_partial_sums_and_total, in, N * sizeof(float), cudaMemcpyHostToDevice));

	delete[]in;
	
	// launch sevral blocks to compute the sum of the partial sums
	size_t total = vLen.size();
	float *start = d_partial_sums_and_total, *end = d_partial_sums_and_total+ vLen[0];
	while (it){
		size_t block_nums = vBlocks[total - it];
		size_t len = vLen[total - it];
		size_t thread_nums = vBlockSize[total - it];
		block_sum << <block_nums, thread_nums, thread_nums * sizeof(float) >> >(start, end, len);
		start = end;
		end = end + block_nums;
		--it;
	}

	float d_ParaSum;
	checkCudaErrors(cudaMemcpy(&d_ParaSum, d_partial_sums_and_total + plusBlocks + N, sizeof(float), cudaMemcpyDeviceToHost));	//从累加过后数组的0号元素得出结果
	cudaFree(d_partial_sums_and_total);

	endTime = clock();//计时结束
	cout << "GPU run time is: " << (float)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout << "GPU result = " << d_ParaSum << endl;	//显示GPU端结果
	
}