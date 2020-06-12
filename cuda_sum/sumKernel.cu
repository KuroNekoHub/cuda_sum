#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <ctime>
#include <iostream>
#include <vector>
#include "inc/helper_cuda.h"
using namespace std;

const size_t N = 5e8;       //���鳤��


__global__ void d_ParallelTest(float *Para)
{
	size_t tid = threadIdx.x;
	//----��ѭ�����������ӣ�stride��η���������2��-----------------------------------------------------
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
	// load input size_to __shared__ memory //һ���̸߳����һ��Ԫ�ش�ȫ���ڴ����뵽�����ڴ�
	float x = 0;
		
	if (i < n)
	{
		x = input[i];
		//if(n == 1954)
		//	printf("%.5f\t%d\n", x, i);
	}
	sdata[threadIdx.x] = x;
	__syncthreads(); //�ȴ������̰߳��Լ������Ԫ�����뵽�����ڴ�

							 // contiguous range pattern//���ڽ��кϲ�������ÿ�κϲ���Ϊһ��.ע��threadIdx.x�ǿ��ڵ�ƫ�ƣ����������i��ȫ�ֵ�ƫ�ơ�
	for (int offset = blockDim.x / 2;
		offset > 0;
		offset >>= 1)
	{
		if (threadIdx.x < offset)//����ֻ��ĳЩ�̲߳Ž��в���
		{
			// add a partial sum upstream to our own
			sdata[threadIdx.x] += sdata[threadIdx.x + offset];
		}
		// wait until all threads in the block have
		// updated their partial sums
		__syncthreads();
	}
	// thread 0 writes the final result//ÿ������߳�0�����ſ�����͵Ľ��
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
		std::cout << "ʹ��GPU device " << i << ": " << devProp.name << std::endl;
		std::cout << "�豸ȫ���ڴ������� " << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
		std::cout << "SM��������" << devProp.multiProcessorCount << std::endl;
		std::cout << "ÿ���߳̿�Ĺ����ڴ��С��" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
		std::cout << "ÿ���߳̿������߳�����" << devProp.maxThreadsPerBlock << std::endl;
		std::cout << "�豸��һ���߳̿飨Block���ֿ��õ�32λ�Ĵ��������� " << devProp.regsPerBlock << std::endl;
		std::cout << "ÿ��EM������߳�����" << devProp.maxThreadsPerMultiProcessor << std::endl;
		std::cout << "ÿ��EM������߳�������" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
		std::cout << "�豸�϶ദ������������ " << devProp.multiProcessorCount << std::endl;
		std::cout << "======================================================" << std::endl;
	}

	float *in = new float[N];
	for (size_t i = 0; i<N; i++)
	{
		in[i] = 1.0;	//���鸳ֵ
	}
	//CPU
	clock_t startTime, endTime;
	startTime = clock();//��ʱ��ʼ
	double ParaSum = 0;
	for (size_t i = 0; i<N; i++)
	{
		ParaSum += in[i];	//CPU�������ۼ�
	}
	endTime = clock();//��ʱ����
	cout << "CPU run time is: " << (float)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout << "CPU result = " << ParaSum << endl;	//��ʾCPU�˽��

	//GPU
	startTime = clock();//��ʱ��ʼ

	size_t block_size = 1024;//�߳̿�Ĵ�С��Ŀǰ��Щgpu���߳̿����Ϊ512����ЩΪ1024.
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
	float *d_partial_sums_and_total;//һ���߳̿�һ���ͣ������һ��Ԫ�أ���������߳̿�ĺ͡�
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
	checkCudaErrors(cudaMemcpy(&d_ParaSum, d_partial_sums_and_total + plusBlocks + N, sizeof(float), cudaMemcpyDeviceToHost));	//���ۼӹ��������0��Ԫ�صó����
	cudaFree(d_partial_sums_and_total);

	endTime = clock();//��ʱ����
	cout << "GPU run time is: " << (float)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout << "GPU result = " << d_ParaSum << endl;	//��ʾGPU�˽��
	
}