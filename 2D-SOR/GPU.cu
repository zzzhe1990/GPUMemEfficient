#include<iostream>
#include<cstdlib>
#include<fstream>
#include<string>
#include<sys/time.h>

//#define PRINT_FINAL_RESULT
typedef unsigned long long int UINT;

using namespace std;

__global__ void GPU(int *dev_arr1, int *dev_arr2, const int rowsize, 
			const int colsize, const int n1, const int threadsPerBlock, int padd){
	int offset = rowsize * blockIdx.x + padd;
	int idx = threadIdx.x + offset;
	while (idx < n1 + padd + offset){
		dev_arr2[idx] = (dev_arr1[idx-1] + dev_arr1[idx-rowsize] + dev_arr1[idx]
				+ dev_arr1[idx+1] + dev_arr1[idx+rowsize]) / 5;	
		idx += threadsPerBlock;
	}	
	__threadfence();	
}

void checkGPUError(cudaError err){
	if (cudaSuccess != err){
		printf("CUDA error in file %s, in line %i: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void SOR(int n1, int n2, int padd, int *arr1, int *arr2, int MAXTRIAL){
	int rowsize = n1 + 2 * padd;
	int colsize = n2 + 2 * padd;

	int *dev_arr1, *dev_arr2, *tmp;
	int tablesize = rowsize * colsize;
	
//	size_t freeMem, totalMem;
//	cudaMemGetInfo(&freeMem, &totalMem);
//	cout << "current GPU memory info FREE: " << freeMem << " Bytes, Total: " << totalMem << " Bytes.";
//	cout << "colsize: " << colsize << ", rowsize: " << rowsize << ", allocates: " << tablesize * sizeof(int)<< " Bytes." << endl;
	cudaError err = cudaMalloc(&dev_arr1, tablesize * sizeof(int));
	checkGPUError(err);
	err = cudaMalloc(&dev_arr2, tablesize * sizeof(int));
	checkGPUError(err);
	
	cudaMemcpy(dev_arr1, arr1, tablesize * sizeof(int), cudaMemcpyHostToDevice);

	int threadsPerBlock = min(1024, n1);
	int blocksPerGrid = n2;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	struct timeval tbegin, tend;
	gettimeofday(&tbegin, NULL);

	//suppose n1 is the row size and the longer array
	for (int t = 0; t < MAXTRIAL; t++){
		GPU<<<blocksPerGrid, threadsPerBlock>>>(&dev_arr1[padd * rowsize], &dev_arr2[padd * rowsize], rowsize, colsize, n1, threadsPerBlock, padd);		
		cudaDeviceSynchronize();
		tmp = dev_arr1;
		dev_arr1 = dev_arr2;
		dev_arr2 = tmp;
	}

	gettimeofday(&tend, NULL);
	double s = (double)(tend.tv_sec - tbegin.tv_sec) + (double)(tend.tv_usec - tbegin.tv_usec) / 1000000.0;
	cout << "execution time: " << s << " second." << endl;

	cudaMemcpy(arr1, dev_arr1, tablesize*sizeof(int), cudaMemcpyDeviceToHost);
#ifdef PRINT_FINAL_RESULT
	//display table
	cout << "full table: " << endl;
	for (int i=0; i<colsize; i++){
		for (int j=0; j<rowsize; j++){
			cout << arr1[i * rowsize + j] << " ";
		}
		cout << endl;
	}
#endif
	cout << "The last element: " << arr1[n2*rowsize + n1] << endl;
	
	cudaFree(dev_arr1);
	cudaFree(dev_arr2);
}

