#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>


#include <iostream>

using namespace std;

const int N = 128;        //���鳤��


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

extern "C"
void ParallelTest()
{
	double *Para;
	double *pcpu = new double[N];
	cudaMalloc((void **)&Para, sizeof(double) * N);        //ͳһ�ڴ�Ѱַ��CPU��GPU������ʹ�õ�����

	double ParaSum = 0;
	for (int i = 0; i<N; i++)
	{
		pcpu[i] = (i + 1) * 0.1;	//���鸳ֵ
		ParaSum += pcpu[i];	//CPU�������ۼ�
	}
	cudaMemcpy(Para, pcpu, N * sizeof(double), cudaMemcpyHostToDevice);
	cout << " CPU result = " << ParaSum << endl;	//��ʾCPU�˽��
	double d_ParaSum;

	d_ParallelTest << < 1, N >> > (Para);	//���ú˺�����һ������N���̵߳��߳̿飩

	cudaDeviceSynchronize();	//ͬ��
	d_ParaSum = Para[0];	//���ۼӹ��������0��Ԫ�صó����
	cout << " GPU result = " << d_ParaSum << endl;	//��ʾGPU�˽��

}