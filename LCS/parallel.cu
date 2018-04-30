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



__global__ void GPU(int tilesize, const int threadNum, const int maxThreads, int *dev_table, const int rowsize, const int maxlevel){
	//This code has to ensure n2 size is the multiple of 128. And n2 is no smaller than n1, where n2 is row array size, n1 is colum array size
	//on K40, tile size is max to 48K, which is 128*96; on pascal and volta, tile size is max to 64K which is 128*128
	__shared__ int tile[12288];

	int thread = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = thread;
	int address = thread;
	int curjobs = 0;
	int curlevel = 1;
	int startIdx, startx, starty;

	while(idx < tilesize){
		tile[idx] = dev_table[address];
		address += rowsize;
		idx += threadNum;
	}
	
	while (curlevel <= maxlevel){
		if (curjobs < lenY)
			curjobs++;

		startx = curlevel - 1;
		starty = 0;

		if (curlevel > tileX){
			curjobs--;	
			startx = tileX -1;
			starty = curlevel - tileX;
		}
	
		if (thread < curjobs){
			startx -= thread;
			starty += thread;
			startIdx = startx + starty * tileX;
			tile[startIdx] = 
		}

		curlevel++;
		
		__syncthreads();	
	}
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
	int tileX = 128;
	int tileY = 96;
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

	int maxthreads;
	int maxlevel;
	int curlevel = 1;
	int curjobs = 1;
	int startx, starty;
	int threadPerBlock = 128, blockPerGrid;
	int numStream = 16;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	int xseg = (n1+tileX-1) / tileX;
	int yseg = (n2+tileY-1) / tileY;
	int maxSegThreads = min(xseg, yseg);		//max number of segs at either of the seg levels.
	int maxSegLevel = xseg + yseg - 1;
	int curSegLevel = 1;
	int curSegJobs = 1;
	int startSegX, startSegY;	

	cudaStream_t stream[numStream];
	for (int s=0; s<numStream; s++)
		cudaStreamCreate(&stream[s]);

	while(curSegLevel <= maxSegLevel){
		int startSegIdx;
		int segIdx = 0;
		if (curSegLevel <= xseg){
			startSegIdx = curSegLevel - 1;
			curSegJobs = curSegLevel;
			startSegX = startSegIdx;
			startSegY = 0;
		}	
		else{
			startSegIdx = xseg - 1 + xseg * (curSegLevel - xseg);
			curSegJobs = 2 * segx - curSegLevel;
			startSegX = xseg - 1;
			startSegY = curSegLevel - xseg;
		}
		
		while ( startSegX > 0 && startSegY <= yseg - 1){
			//suppose n2 is the row size and the longer array
			int i = poolsize + startSegX * tileX;
			int j = poolsize + startSegY * tileY;
			int startSegAdd = j * (n2 + poolsize) + i;
			int s = segIdx % numStream;
			//resY is used to determine the rest size of Y. This is used to check if the rest size of Y is smaller than tileY.
			int resY = n1 - startSegY * tileY;
			int lenY = min(resY, tileY);
			maxlevel = tileX + lenY - 1;
			maxThreads = min(tileX, lenY);
			int tilesize = tileX * lenY;

			GPU<<<1, threadPerBlock, 0, stream[s]>>>(tilesize, threadPerBlock, maxThreads, &dev_table[startSegAdd], rowsize);

			//This code has to ensure n2 size is the multiple of 128.
			while (curlevel <= maxlevel){				
				if (curlevel <= tileX){
					startIdx = startSegAdd + curlevel - 1;
					curjobs = min(curlevel, maxthreads);
					startx = startIdx;
					starty = 0;
				}
				else{
					startx = tileX -1;
					starty = curlevel - tileX;
					startIdx = startSegAdd + tileX - 1 + rowsize * starty;
					curjobs = tileX + lenY - curlevel;
				}
				int numthreads = (curjobs + 31) / 32;
				numthreads *= 32;
				blockPerGrid = (numthreads + threadPerBlock - 1) / threadPerBlock;

				GPU<<<blockPerGrid, threadPerBlock, 0, stream[s]>>>(&dev_table[], dev_arr1, dev_arr2, startIdx, curjobs, rowsize, startx, starty);

				cudaStreamSynchronize(stream[s]);
				
				curlevel++;
			}
			startSegX--;
			startSegY++;
			segIdx++;
		}
		//this synchronization is might removable
		cudaDeviceSynchronize();
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

	for (int s=0; s<numStream; s++)
		cudaStreamDestroy[stream[s]];
	
	cudaFree(dev_arr1);
	cudaFree(dev_arr2);
	cudaFree(dev_table);

//	delete[] table;

	return lcslength;
}

