#include<iostream>
#include<cstdlib>
#include<fstream>
#include<string>
#include<sys/time.h>

typedef unsigned long long int UINT;

using namespace std;

__global__ void GPU(int *dev_table, int startIdx, int curjobs, const int rowsize, int startx, int starty){
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (thread < curjobs){
		int idx = startIdx + (thread * rowsize - thread);
		dev_table[idx] = (dev_table[idx-1] + dev_table[idx-rowsize] + dev_table[idx]
				+ dev_table[idx+1] + dev_table[idx+rowsize]) / 5;	
	}		
}

void checkGPUError(cudaError err){
	if (cudaSuccess != err){
		printf("CUDA error in file %s, in line %i: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void SOR(int n1, int n2, int *arr){
	int paddsize = 1;
	int rowsize = n1 + 2 * paddsize;
	int colsize = n2 + 2 * paddsize;

	int *dev_table;
	size_t freeMem, totalMem;

	cudaMemGetInfo(&freeMem, &totalMem);
	int tablesize = rowsize * colsize;
	cout << "current GPU memory info FREE: " << freeMem << " Bytes, Total: " << totalMem << " Bytes.";
	cout << "colsize: " << colsize << ", rowsize: " << rowsize << ", allocates: " << tablesize * sizeof(int)<< " Bytes." << endl;
	cudaError err = cudaMalloc(&dev_table, tablesize * sizeof(int));
	checkGPUError(err);
	
	cudaMemcpy(dev_table, arr, tablesize * sizeof(int), cudaMemcpyHostToDevice);

	int maxthreads = min(n1 ,n2);
	int maxlevel = n1 + n2 - 1;	
	int curlevel = 1;
	int curjobs = 1;
	int startx, starty;
	int threadPerBlock = 128, blockPerGrid;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	//suppose n1 is the row size and the longer array
	while(curlevel <= maxlevel){
//		cout << "level: " << curlevel << endl;
		int startIdx;
		if (curlevel <= n1){
			startIdx = curlevel - 1; 
			curjobs = curlevel;
			startx = startIdx;
			starty = 0;
		}
		else{
			startIdx = n1 - 1 + rowsize * (curlevel - n1);
			curjobs = 2 * n1 - curlevel;
			startx = n1 - 1;
			starty = curlevel - n1;
		}

		int numthreads = (curjobs + 31) / 32;
		numthreads *= 32;
	
		blockPerGrid = (numthreads + threadPerBlock - 1) / threadPerBlock;

		GPU<<<blockPerGrid, threadPerBlock>>>(&dev_table[paddsize*rowsize+paddsize], startIdx, curjobs, rowsize, startx, starty);		
		
		cudaDeviceSynchronize();		

		curlevel++;
	}

//	cudaMemcpy(table, dev_table, (n1+paddsize)*rowsize*sizeof(int), cudaMemcpyDeviceToHost);
/*
	//display table
	cout << "full table: " << endl;
	for (int i=0; i<n1+paddsize; i++){
		for (int j=0; j<n2+paddsize; j++){
			cout << table[i * rowsize + j] << " ";
		}
		cout << endl;
	}
*/	

	cudaFree(dev_table);
}

