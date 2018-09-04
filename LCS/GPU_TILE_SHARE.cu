#include<iostream>
#include<cstdlib>
#include<fstream>
#include<string>
#include<sys/time.h>

typedef unsigned long long int UINT;

using namespace std;


__global__ void GPU(const int tilesize, const int paddsize,  
			int *dev_table, const int rowsize, const int maxlevel, int tileX, int lenY, int *dev_arr1, int *dev_arr2){
	//This code has to ensure n2 size is the multiple of 128. And n2 is no smaller than n1, where n2 is row array size, n1 is colum array size
	//on K40, tile size is max to 48K, which is 128*96; on pascal and volta, tile size is max to 64K which is 128*128
	//For running on K40, we reserve the shared memory space for a 64*64 tile. Because of the dependency, the actual shared memory size is 96 * 96. 96 is picked for ensuring memory coalscing.
	//This code, length of x axis cannot be larger than y axis for each tile.
	
	__shared__ int table[12288];

	int thread = threadIdx.x;
	int idx = thread;
	int address = thread;
	int curjobs = 0;
	int curlevel = 1;
	int startIdx, startx, starty;
	int tableX = tileX + paddsize;
	int x,y;
//	printf("thread: %d, idx: %d, address: %d, tableX: %d\n", thread, idx, address, tableX);
	if(thread < tileX+paddsize){
		while(idx < tilesize){
//		printf("thread: %d, idx: %d, address: %d\n", thread, idx, address);
			table[idx] = dev_table[address];
			address += rowsize;
			idx += (tileX+paddsize);
		}
	}

		while (curlevel <= maxlevel){
			if (curlevel <= lenY){
				curjobs++;
//			printf("curlevel: %d, lenY: %d, curjobs: %d, thread: %d\n", curlevel, lenY, curjobs, thread);
			}

			startx = paddsize + curlevel - 1;
			starty = paddsize;

			if (curlevel > tileX){
				curjobs--;	
				startx = tableX -1;
				starty = paddsize + curlevel - tileX;
			}
	
			if (thread < curjobs){
				startx -= thread;
				starty += thread;
				startIdx = startx + starty * tableX;
				table[startIdx] = max(table[startIdx - 1], table[startIdx - tableX]);
				x = startx - paddsize;
				y = starty - paddsize;
				if (dev_arr1[x] == dev_arr2[y])
					table[startIdx] = table[startIdx - tableX - 1] + 1;				
	//			printf("curlevel: %d, curjobs: %d, thread: %d, idx: %d, startx: %d, starty: %d, startIdx: %d, table[startIdx]: %d\n", curlevel, curjobs, thread, idx, startx, starty, startIdx, table[startIdx]);
			}

			curlevel++;
		
			__syncthreads();	
		}

	idx = thread;
	address = thread;
	if (thread < tileX+paddsize){
		while (idx < tilesize){
			dev_table[address] = table[idx];
			address += rowsize;
			idx += (tileX+paddsize);
		}
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
	int paddsize = 1;

	//tileY must be larger than tileX
	int tileX = 64;
	int tileY = 128;
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

	int maxlevel;
	int threadPerBlock = tileY + paddsize;
	int blockPerGrid = 1;
	int numStream = 32;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	int xseg = (n1+tileX-1) / tileX;
	int yseg = (n2+tileY-1) / tileY;
	int maxSegThreads = min(xseg, yseg);		//max number of segs at either of the seg levels.
	int maxSegLevel = xseg + yseg - 1;
	int curSegLevel = 1;
//	int curSegJobs = 1;
	int startSegX, startSegY;	

	cudaStream_t stream[numStream];
	for (int s=0; s<numStream; s++)
		cudaStreamCreate(&stream[s]);

	while(curSegLevel <= maxSegLevel){
		int segIdx = 0;
		if (curSegLevel <= xseg){
			//curSegJobs = curSegLevel;
			startSegX = curSegLevel - 1;
			startSegY = 0;
		}	
		else{
			//startSegIdx = xseg - 1 + xseg * (curSegLevel - xseg);
			//curSegJobs = 2 * xseg - curSegLevel;
			startSegX = xseg - 1;
			startSegY = curSegLevel - xseg;
		}
		
//		cout << "curSegLevel: " << curSegLevel << ", maxSegLevel: " << maxSegLevel << endl;	
		
		while ( startSegX >= 0 && startSegY <= yseg - 1){
			//suppose n2 is the row size and the longer array
			//int i = paddsize + startSegX * tileX;
			//int j = paddsize + startSegY * tileY;
			int i = startSegX * tileX;
			int j = startSegY * tileY;
			int startSegAdd = j * rowsize + i;
			int s = segIdx % numStream;
			//resY is used to determine the rest size of Y. This is used to check if the rest size of Y is smaller than tileY.
			int resY = n1 - startSegY * tileY;
			int lenY = min(resY, tileY);
			maxlevel = tileX + lenY - 1;
			int tilesize = (tileX+paddsize) * (lenY+paddsize);

			GPU<<<blockPerGrid, threadPerBlock, 0, stream[s]>>>(tilesize, paddsize,  &dev_table[startSegAdd], rowsize, 
										maxlevel, tileX, lenY, &dev_arr1[i], &dev_arr2[j]);
		
//			cout << "startSegX: " << startSegX << ", startSegY: " << startSegY << ", segIdx: " << segIdx << endl;
			startSegX--;
			startSegY++;
			segIdx++;
		}
		//this synchronization is might removable
		cudaDeviceSynchronize();

		curSegLevel++;
	}
	
	cudaMemcpy(&lcslength, &dev_table[tablesize-1], sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(table, dev_table, (n1+paddsize)*rowsize*sizeof(int), cudaMemcpyDeviceToHost);
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

	for (int s=0; s<numStream; s++)
		cudaStreamDestroy(stream[s]);
	
	cudaFree(dev_arr1);
	cudaFree(dev_arr2);
	cudaFree(dev_table);

//	delete[] table;

	return lcslength;
}

