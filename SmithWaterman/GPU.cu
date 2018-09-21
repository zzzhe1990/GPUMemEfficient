#include<iostream>
#include<cstdlib>
#include<fstream>
#include<string>
#include<sys/time.h>

//#define debug
typedef unsigned long long int UINT;

using namespace std;

__device__ int s(int a, int b){
	return a==b?3:-3;
}

__global__ void GPU(int *dev_table, int *dev_arr1, int *dev_arr2, int startIdx, int curjobs, const int rowsize, int startx, int starty){
	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (thread < curjobs){
		int idx = startIdx + (thread * rowsize - thread);
		int x = startx - thread;
		int y = starty + thread;
		
		dev_table[idx] = max(dev_table[idx-1]-2,
				max(dev_table[idx-rowsize]-2,
				max(dev_table[idx-rowsize-1] + s(dev_arr1[x], dev_arr2[y]), 0)));
	}
//	__threadfence();
}

void checkGPUError(cudaError err){
	if (cudaSuccess != err){
		printf("CUDA error in file %s, in line %i: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

int SW(int n1, int n2, int *arr1, int *arr2){
	int last;
	int paddsize = 1;
	int rowsize = paddsize + n2;
	int colsize = paddsize + n1;

	int *dev_table, *dev_arr1, *dev_arr2;

	int *table;

	table = new int[colsize * rowsize];

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
	int threadPerBlock = 32, blockPerGrid;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

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

		GPU<<<blockPerGrid, threadPerBlock>>>(&dev_table[paddsize*rowsize+paddsize], dev_arr1, dev_arr2, startIdx, curjobs, rowsize, startx, starty);		
		
		cudaDeviceSynchronize();		

		curlevel++;
	}

	cudaMemcpy(&last, &dev_table[tablesize-1], sizeof(int), cudaMemcpyDeviceToHost);
#ifdef debug
	cudaMemcpy(table, dev_table, (n1+paddsize)*rowsize*sizeof(int), cudaMemcpyDeviceToHost);
	//display table
	cout << "full table: " << endl;
	for (int i=0; i<n1+paddsize; i++){
		for (int j=0; j<n2+paddsize; j++){
			cout << table[i * rowsize + j] << " ";
		}
		cout << endl;
	}
#endif	

	cudaFree(dev_arr1);
	cudaFree(dev_arr2);
	cudaFree(dev_table);

	delete[] table;

	return last;
}

