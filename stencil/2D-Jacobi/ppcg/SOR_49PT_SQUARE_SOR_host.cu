#include <assert.h>
#include <stdio.h>
#include "SOR_kernel.hu"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<string.h>
#include<errno.h>

const int n1 = 4096, n2 = 4096;
const int nn1 = 4108, nn2 = 4108;

void SOR(int len1, int len2, int arr1[nn1][nn2], int arr2[nn1][nn2], int padd, int trial){
	struct timeval tbegin, tend;
	gettimeofday(&tbegin, NULL);
	
	#define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
	#define ppcg_fdiv_q(n,d) (((n)<0) ? -((-(n)+(d)-1)/(d)) : (n)/(d))
	if (trial >= 1 && len1 >= padd + 1 && len2 >= padd + 1) {
#define cudaCheckReturn(ret) \
  do { \
    cudaError_t cudaCheckReturn_e = (ret); \
    if (cudaCheckReturn_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
      fflush(stderr); \
    } \
    assert(cudaCheckReturn_e == cudaSuccess); \
  } while(0)
#define cudaCheckKernel() \
  do { \
    cudaCheckReturn(cudaGetLastError()); \
  } while(0)

	  int *dev_arr1;
	  int *dev_arr2;
	  
	  cudaCheckReturn(cudaMalloc((void **) &dev_arr1, (len1 + 3) * (4108) * sizeof(int)));
	  cudaCheckReturn(cudaMalloc((void **) &dev_arr2, (len1 + 3) * (4108) * sizeof(int)));
	  
	  if (padd <= 4110) {
	    cudaCheckReturn(cudaMemcpy(dev_arr1, arr1, (len1 + 3) * (4108) * sizeof(int), cudaMemcpyHostToDevice));
	    cudaCheckReturn(cudaMemcpy(dev_arr2, arr2, (len1 + 3) * (4108) * sizeof(int), cudaMemcpyHostToDevice));
	  }
	struct timeval t1, t2;
	gettimeofday(&t1, NULL);
	  for (int c0 = padd; c0 < 6 * trial + len1 - 3; c0 += 1) {
	    if (6 * trial + len1 >= c0 + 7 && c0 + 6 * ppcg_fdiv_q(len1 - c0 - 1, 6) >= padd)
	      {
	        dim3 k0_dimBlock(16, 31);
	        dim3 k0_dimGrid(len2 + 30 >= ((len2 + 31) % 8192) + padd ? 256 : (len2 + 31) / 32 - 256 * ((len2 + 31) / 8192), ppcg_min(256, (trial + 31) / 32));
	        kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_arr1, dev_arr2, trial, padd, len1, len2, c0);
	        cudaCheckKernel();
	      }
	    	cudaDeviceSynchronize();
	    if (c0 >= padd + 3 && c0 + 6 * ppcg_fdiv_q(len1 - c0 + 2, 6) >= padd + 3)
	      {
	        dim3 k1_dimBlock(16, 31);
	        dim3 k1_dimGrid(len2 + 30 >= ((len2 + 31) % 8192) + padd ? 256 : (len2 + 31) / 32 - 256 * ((len2 + 31) / 8192), ppcg_min(256, (trial + 31) / 32));
	        kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_arr1, dev_arr2, trial, padd, len1, len2, c0);
	        cudaCheckKernel();
	      }
	      
	  }
	  cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);
	double t3 = (double)(t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec) / 1000000.0;
	printf("execution time: %lf\n", t3);
	  if (padd <= 4110) {
	    cudaCheckReturn(cudaMemcpy(arr1, dev_arr1, (len1 + 3) * (4108) * sizeof(int), cudaMemcpyDeviceToHost));
	    cudaCheckReturn(cudaMemcpy(arr2, dev_arr2, (len1 + 3) * (4108) * sizeof(int), cudaMemcpyDeviceToHost));
	  }
	  cudaCheckReturn(cudaFree(dev_arr1));
	  cudaCheckReturn(cudaFree(dev_arr2));
	}
	
	gettimeofday(&tend, NULL);
	double tt = (double)(tend.tv_sec - tbegin.tv_sec) + (double)(tend.tv_usec - tbegin.tv_usec) / 1000000.0;
	printf("execution time: %lf s\n", tt);
}

int main(){
	int trial = 64;
     	int padd = 6;
	static int arr1[nn1][nn2];
	static int arr2[nn1][nn2];

	for (int row = 0; row < nn1; row++){
		for (int col = 0; col < nn2; col++){
			arr1[row][col] = rand() % 100;
			arr2[row][col] = arr1[row][col];
		}
	}	


	SOR(n1 + padd, n2 + padd, arr1, arr2, padd, trial);

	return 0;
}
