#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>


#include <iostream>

using namespace std;

const int N = 128;        //数组长度


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

extern "C"
void ParallelTest()
{
	double *Para;
	double *pcpu = new double[N];
	cudaMalloc((void **)&Para, sizeof(double) * N);        //统一内存寻址，CPU和GPU都可以使用的数组

	double ParaSum = 0;
	for (int i = 0; i<N; i++)
	{
		pcpu[i] = (i + 1) * 0.1;	//数组赋值
		ParaSum += pcpu[i];	//CPU端数组累加
	}
	cudaMemcpy(Para, pcpu, N * sizeof(double), cudaMemcpyHostToDevice);
	cout << " CPU result = " << ParaSum << endl;	//显示CPU端结果
	double d_ParaSum;

	d_ParallelTest << < 1, N >> > (Para);	//调用核函数（一个包含N个线程的线程块）

	cudaDeviceSynchronize();	//同步
	d_ParaSum = Para[0];	//从累加过后数组的0号元素得出结果
	cout << " GPU result = " << d_ParaSum << endl;	//显示GPU端结果

}