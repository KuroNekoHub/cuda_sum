// cuda_sum.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <iostream>
using namespace std;

extern "C"
void ParallelTest();
int main() {
	//���й�Լ
	ParallelTest();	//���ù�Լ����

	system("pause");
	return 0;
}