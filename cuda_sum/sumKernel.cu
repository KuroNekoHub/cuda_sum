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

	float *pcpu = new float[N];
	for (size_t i = 0; i<N; i++)
	{
		pcpu[i] = 1.0;	//���鸳ֵ
	}
	//CPU
	clock_t startTime, endTime;
	startTime = clock();//��ʱ��ʼ
	double ParaSum = 0;
	for (size_t i = 0; i<N; i++)
	{
		ParaSum += pcpu[i];	//CPU�������ۼ�
	}
	endTime = clock();//��ʱ����
	cout << "CPU run time is: " << (float)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout << "CPU result = " << ParaSum << endl;	//��ʾCPU�˽��

	//GPU
	startTime = clock();//��ʱ��ʼ
	float *d_input;
	checkCudaErrors(cudaMalloc((void **)&d_input, sizeof(float) * N));
	checkCudaErrors(cudaMemcpy(d_input, pcpu, N * sizeof(float), cudaMemcpyHostToDevice));
	delete[]pcpu;

	size_t block_size = 1024;//�߳̿�Ĵ�С��Ŀǰ��Щgpu���߳̿����Ϊ512����ЩΪ1024.
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
	float *d_partial_sums_and_total;//һ���߳̿�һ���ͣ������һ��Ԫ�أ���������߳̿�ĺ͡�
	checkCudaErrors(cudaMalloc((void**)&d_partial_sums_and_total, sizeof(float) * (num_blocks + 1 + plusBlocks)));

	// launch one kernel to compute, per-block, a partial sum//��ÿ���߳̿�ĺ������
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
	checkCudaErrors(cudaMemcpy(&d_ParaSum, d_partial_sums_and_total + num_blocks + plusBlocks, sizeof(float), cudaMemcpyDeviceToHost));	//���ۼӹ��������0��Ԫ�صó����
	cudaFree(d_partial_sums_and_total);

	endTime = clock();//��ʱ����
	cout << "GPU run time is: " << (float)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout << "GPU result = " << d_ParaSum << endl;	//��ʾGPU�˽��
	
}