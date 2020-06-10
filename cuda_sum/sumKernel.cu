#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <ctime>
#include <iostream>
#include "inc/helper_cuda.h"
using namespace std;

const int N = 1e8;        //���鳤��


__global__ void d_ParallelTest(double *Para)
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
__global__ void block_sum(double *input,
	double *per_block_results,
	const size_t n)
{
	extern __shared__ double sdata[];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	// load input into __shared__ memory //һ���̸߳����һ��Ԫ�ش�ȫ���ڴ����뵽�����ڴ�
	double x = 0;

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
	
	double *pcpu = new double[N];
	for (int i = 0; i<N; i++)
	{
		pcpu[i] = 0.1;	//���鸳ֵ
	}
	//CPU
	clock_t startTime, endTime;
	startTime = clock();//��ʱ��ʼ
	double ParaSum = 0;
	for (int i = 0; i<N; i++)
	{
		ParaSum += pcpu[i];	//CPU�������ۼ�
	}
	endTime = clock();//��ʱ����
	cout << "CPU run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout << " CPU result = " << ParaSum << endl;	//��ʾCPU�˽��

	//GPU
	startTime = clock();//��ʱ��ʼ
	double *d_input;
	checkCudaErrors(cudaMalloc((void **)&d_input, sizeof(double) * N));
	checkCudaErrors(cudaMemcpy(d_input, pcpu, N * sizeof(double), cudaMemcpyHostToDevice));

	size_t block_size = 512;//�߳̿�Ĵ�С��Ŀǰ��Щgpu���߳̿����Ϊ512����ЩΪ1024.
	size_t num_blocks = (N - 1) / block_size + 1;
	// allocate space to hold one partial sum per block, plus one additional
	// slot to store the total sum
	double *d_partial_sums_and_total;//һ���߳̿�һ���ͣ������һ��Ԫ�أ���������߳̿�ĺ͡�
	checkCudaErrors(cudaMalloc((void**)&d_partial_sums_and_total, sizeof(double) * (num_blocks + 1)));

	// launch one kernel to compute, per-block, a partial sum//��ÿ���߳̿�ĺ������
	block_sum << <num_blocks, block_size, block_size * sizeof(double) >> >(d_input, d_partial_sums_and_total, N);
	//cudaDeviceSynchronize();
	// launch a single block to compute the sum of the partial sums
	//�ٴ���һ���߳̿����һ���Ľ����͡�
	//ע�������и����ƣ���һ���߳̿�����������벻����һ���߳̿��̵߳������������Ϊ��һ���ð���һ���Ľ������һ���߳̿������
	//��num_blocks���ܴ����߳̿������߳�������
	int new_blocks = pow(2, ceil(log2(num_blocks)));

	block_sum << <1, new_blocks, new_blocks * sizeof(double) >> >(d_partial_sums_and_total, d_partial_sums_and_total + num_blocks, num_blocks);
	//cudaDeviceSynchronize();

	double d_ParaSum;
	checkCudaErrors(cudaMemcpy(&d_ParaSum, d_partial_sums_and_total + num_blocks, sizeof(double), cudaMemcpyDeviceToHost));	//���ۼӹ��������0��Ԫ�صó����
	endTime = clock();//��ʱ����
	cout << "GPU run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cout << " GPU result = " << d_ParaSum << endl;	//��ʾGPU�˽��

	cudaFree(d_input);
	delete[]pcpu;
}