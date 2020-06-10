#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <ctime>
#include <iostream>
#include "inc/helper_cuda.h"
using namespace std;

const int N = 12000;        //���鳤��


__global__ void d_ParallelTest(float *Para)
{
	int tid = threadIdx.x;
	//----��ѭ�����������ӣ�stride��η���������2��-----------------------------------------------------
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
__global__ void block_sum(float *input,
	float *per_block_results,
	const size_t n)
{
	extern __shared__ float sdata[];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	// load input into __shared__ memory //һ���̸߳����һ��Ԫ�ش�ȫ���ڴ����뵽�����ڴ�
	float x = 0;

	if (i < n)
	{
		x = input[i];
	}
	sdata[threadIdx.x] = x;
	cudaDeviceSynchronize(); //�ȴ������̰߳��Լ������Ԫ�����뵽�����ڴ�

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
		cudaDeviceSynchronize();
	}
	// thread 0 writes the final result//ÿ������߳�0�����ſ�����͵Ľ��
	if (threadIdx.x == 0)
	{
		per_block_results[blockIdx.x] = sdata[0];
		//printf("%.5f\t%d\n", per_block_results[blockIdx.x], blockDim.x);
	}
}


extern "C"
void ParallelTest()
{
	float *pcpu = new float[N];
	for (int i = 0; i<N; i++)
	{
		pcpu[i] = 1.0;	//���鸳ֵ
	}
	//CPU
	clock_t startTime, endTime;
	startTime = clock();//��ʱ��ʼ
	float ParaSum = 0;
	for (int i = 0; i<N; i++)
	{
		ParaSum += pcpu[i];	//CPU�������ۼ�
	}
	endTime = clock();//��ʱ����
	cout << "CPU run time is: " << (float)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout << " CPU result = " << ParaSum << endl;	//��ʾCPU�˽��

	//GPU
	startTime = clock();//��ʱ��ʼ
	float *d_input;
	checkCudaErrors(cudaMalloc((void **)&d_input, sizeof(float) * N));
	checkCudaErrors(cudaMemcpy(d_input, pcpu, N * sizeof(float), cudaMemcpyHostToDevice));

	size_t block_size = 512;//�߳̿�Ĵ�С��Ŀǰ��Щgpu���߳̿����Ϊ512����ЩΪ1024.
	size_t num_blocks = (N / block_size) + ((N%block_size) ? 1 : 0);
	// allocate space to hold one partial sum per block, plus one additional
	// slot to store the total sum
	float *d_partial_sums_and_total;//һ���߳̿�һ���ͣ������һ��Ԫ�أ���������߳̿�ĺ͡�
	checkCudaErrors(cudaMalloc((void**)&d_partial_sums_and_total, sizeof(float) * (num_blocks + 1)));

	// launch one kernel to compute, per-block, a partial sum//��ÿ���߳̿�ĺ������
	block_sum << <num_blocks, block_size, block_size * sizeof(float) >> >(d_input, d_partial_sums_and_total, N);
	cudaDeviceSynchronize();
	cudaFree(d_input);
	// launch a single block to compute the sum of the partial sums
	//�ٴ���һ���߳̿����һ���Ľ����͡�
	//ע�������и����ƣ���һ���߳̿�����������벻����һ���߳̿��̵߳������������Ϊ��һ���ð���һ���Ľ������һ���߳̿������
	//��num_blocks���ܴ����߳̿������߳�������
	float *b = new float[num_blocks];
	checkCudaErrors(cudaMemcpy(b, d_partial_sums_and_total, num_blocks*sizeof(float), cudaMemcpyDeviceToHost));	//���ۼӹ��������0��Ԫ�صó����
	float a = 0;
	for (int i = 0; i < num_blocks; ++i)
		a += b[i];
	delete[]b;
	int new_blocks = pow(2, ceil(log2(num_blocks)));

	block_sum << <1, new_blocks, new_blocks * sizeof(float) >> >(d_partial_sums_and_total, d_partial_sums_and_total + num_blocks, num_blocks);
	cudaDeviceSynchronize();

	float d_ParaSum;
	checkCudaErrors(cudaMemcpy(&d_ParaSum, d_partial_sums_and_total + num_blocks, sizeof(float), cudaMemcpyDeviceToHost));	//���ۼӹ��������0��Ԫ�صó����
	endTime = clock();//��ʱ����
	cout << "GPU run time is: " << (float)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout << " GPU result = " << d_ParaSum << endl;	//��ʾGPU�˽��
	
	cudaFree(d_partial_sums_and_total);
	delete[]pcpu;
}