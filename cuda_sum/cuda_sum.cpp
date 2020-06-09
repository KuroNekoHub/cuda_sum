// cuda_sum.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
using namespace std;

extern "C"
void ParallelTest();
int main() {
	//并行归约
	ParallelTest();	//调用归约函数

	system("pause");
	return 0;
}