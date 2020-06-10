#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <ctime>
#include <iostream>
#include "inc/helper_cuda.h"
using namespace std;

const int N = 1e8;        //数组长度


__global__ void d_ParallelTest(double *Para)
{
	int tid = threadIdx.x;
	//----随循环次数的增加，stride逐次翻倍（乘以2）-----------------------------------------------------
	for (int stride = 1; stride < blockDim.x; stride *= 2)
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
__global__ void block_sum(double *input,
	double *per_block_results,
	const size_t n)
{
	extern __shared__ double sdata[];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	// load input into __shared__ memory //一个线程负责把一个元素从全局内存载入到共享内存
	double x = 0;

	if (i < n)
	{
		x = input[i];
	}
	sdata[threadIdx.x] = x;
	cudaDeviceSynchronize(); //等待所有线程把自己负责的元素载入到共享内存

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
		cudaDeviceSynchronize();
	}
	// thread 0 writes the final result//每个块的线程0负责存放块内求和的结果
	if (threadIdx.x == 0)
	{
		per_block_results[blockIdx.x] = sdata[0];
		//printf("%.5f\t%d\n", per_block_results[blockIdx.x], blockDim.x);
	}
}


extern "C"
void ParallelTest()
{
	
	double *pcpu = new double[N];
	for (int i = 0; i<N; i++)
	{
		pcpu[i] = 0.1;	//数组赋值
	}
	//CPU
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	double ParaSum = 0;
	for (int i = 0; i<N; i++)
	{
		ParaSum += pcpu[i];	//CPU端数组累加
	}
	endTime = clock();//计时结束
	cout << "CPU run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout << " CPU result = " << ParaSum << endl;	//显示CPU端结果

	//GPU
	startTime = clock();//计时开始
	double *d_input;
	checkCudaErrors(cudaMalloc((void **)&d_input, sizeof(double) * N));
	checkCudaErrors(cudaMemcpy(d_input, pcpu, N * sizeof(double), cudaMemcpyHostToDevice));

	size_t block_size = 512;//线程块的大小。目前有些gpu的线程块最大为512，有些为1024.
	size_t num_blocks = (N - 1) / block_size + 1;
	// allocate space to hold one partial sum per block, plus one additional
	// slot to store the total sum
	double *d_partial_sums_and_total;//一个线程块一个和，另外加一个元素，存放所有线程块的和。
	checkCudaErrors(cudaMalloc((void**)&d_partial_sums_and_total, sizeof(double) * (num_blocks + 1)));

	// launch one kernel to compute, per-block, a partial sum//把每个线程块的和求出来
	block_sum << <num_blocks, block_size, block_size * sizeof(double) >> >(d_input, d_partial_sums_and_total, N);
	//cudaDeviceSynchronize();
	// launch a single block to compute the sum of the partial sums
	//再次用一个线程块把上一步的结果求和。
	//注意这里有个限制，上一步线程块的数量，必须不大于一个线程块线程的最大数量，因为这一步得把上一步的结果放在一个线程块操作。
	//即num_blocks不能大于线程块的最大线程数量。
	int new_blocks = pow(2, ceil(log2(num_blocks)));

	block_sum << <1, new_blocks, new_blocks * sizeof(double) >> >(d_partial_sums_and_total, d_partial_sums_and_total + num_blocks, num_blocks);
	//cudaDeviceSynchronize();

	double d_ParaSum;
	checkCudaErrors(cudaMemcpy(&d_ParaSum, d_partial_sums_and_total + num_blocks, sizeof(double), cudaMemcpyDeviceToHost));	//从累加过后数组的0号元素得出结果
	endTime = clock();//计时结束
	cout << "GPU run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout << " GPU result = " << d_ParaSum << endl;	//显示GPU端结果

	cudaFree(d_input);
	delete[]pcpu;
}