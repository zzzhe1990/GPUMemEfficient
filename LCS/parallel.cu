#include<iostream>
#include<cstdlib>
#include<fstream>
#include<string>
#include<sys/time.h>

typedef unsigned long long int UINT;

using namespace std;

__global__ void GPU(int *dev_table, int *dev_arr1, int *dev_arr2, int startIdx, int curjobs, const int rowsize, int startx, int starty){
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (thread < curjobs){
		int idx = startIdx + (thread * rowsize - thread);
		int x = startx - thread;
		int y = starty + thread;
		
		*(dev_table + idx) = max(*(dev_table+idx-1), *(dev_table+idx-rowsize));
		
		if (dev_arr1[x] == dev_arr2[y]){
//			printf("if equal, before, idx: %d, dev_table[idx]: %d; pre: %d, dev_table[idx-rowsize-1]: %d \n", idx, dev_table[idx], idx-rowsize-1, dev_table[idx-rowsize-1]);
			*(dev_table+idx) = *(dev_table+ (idx - rowsize - 1) ) + 1;
//			printf("after, dev_table[idx]: %d\n", dev_table[idx]);
		}		
//		printf("thread: %d, idx: %d, x: %d, y: %d, arr1[x]: %d, arr2[y]: %d \n", thread, idx, x, y, dev_arr1[x], dev_arr2[y]);
	}
//	__threadfence();
}

void checkGPUError(cudaError err){
	if (cudaSuccess != err){
		printf("CUDA error in file %s, in line %i: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

int LCS(int n1, int n2, int *arr1, int *arr2){
	int lcslength;
	int poolsize = 32;
	int rowsize = poolsize + n2;
	int colsize = poolsize + n1;

	int *dev_table, *dev_arr1, *dev_arr2;

	//int *table;

	//table = new int[(n1+poolsize) * rowsize];

	size_t freeMem, totalMem;

	cudaMemGetInfo(&freeMem, &totalMem);
	int tablesize = colsize * rowsize;
	cout << "current GPU memory info FREE: " << freeMem << " Bytes, Total: " << totalMem << " Bytes.";
	cout << "colsize: " << colsize << ", rowsize: " << rowsize << ", allocates: " << tablesize * sizeof(int)<< " Bytes." << endl;
	cudaError err = cudaMalloc(&dev_table, tablesize * sizeof(int));
	checkGPUError(err);
	
	cudaMalloc(&dev_arr1, n1*sizeof(int));
	cudaMalloc(&dev_arr2, n2*sizeof(int));

	cudaMemset(dev_table, 0, tablesize * sizeof(int));
	cudaMemcpy(dev_arr1, arr1, n1*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_arr2, arr2, n2*sizeof(int), cudaMemcpyHostToDevice);

	int maxthreads = min(n1, n2);
	int maxlevel = n1 + n2 - 1;	
	int curlevel = 1;
	int curjobs = 1;
	int startx, starty;
	int threadPerBlock = 96, blockPerGrid;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	//suppose n2 is the row size and the longer array
	while(curlevel <= maxlevel){
//		cout << "level: " << curlevel << endl;
		int startIdx;
		if (curlevel <= n2){
			startIdx = curlevel - 1;
			curjobs = curlevel;
			startx = startIdx;
			starty = 0;
		}
		else{
			startIdx = n2 - 1 + rowsize * (curlevel - n2);
			curjobs = 2 * n2 - curlevel;
			startx = n2 - 1;
			starty = curlevel - n2;
		}

		int numthreads = (curjobs + 31) / 32;
		numthreads *= 32;
	
		blockPerGrid = (numthreads + threadPerBlock - 1) / threadPerBlock;

		GPU<<<blockPerGrid, threadPerBlock>>>(&dev_table[poolsize*rowsize+poolsize], dev_arr1, dev_arr2, startIdx, curjobs, rowsize, startx, starty);		
		
		cudaDeviceSynchronize();		

		curlevel++;
	}

	cudaMemcpy(&lcslength, &dev_table[tablesize-1], sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(table, dev_table, (n1+poolsize)*rowsize*sizeof(int), cudaMemcpyDeviceToHost);
/*
	//display table
	cout << "full table: " << endl;
	for (int i=0; i<n1+poolsize; i++){
		for (int j=0; j<n2+poolsize; j++){
			cout << table[i * rowsize + j] << " ";
		}
		cout << endl;
	}
*/	

	cudaFree(dev_arr1);
	cudaFree(dev_arr2);
	cudaFree(dev_table);

//	delete[] table;

	return lcslength;
}

