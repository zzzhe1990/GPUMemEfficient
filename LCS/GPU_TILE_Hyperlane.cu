#include<iostream>
#include<cstdlib>
#include<fstream>
#include<string>
#include<sys/time.h>

typedef unsigned long long int UINT;

using namespace std;

__device__ void moveToShare(int *table, int *dev_table, int tileAddress, int thread, int tileX, int rowsize, int hightY, int padd){
	//potential bank conflict for accessing the data of each anti-diagonal
	//to avoid bank conflict, have to adjust the memory layout.
	//We also need to convert the dependency calculation because of the change of the memory layout.
	//For shared memory implementation, tileX is no larger than tileY
	int pos = tileAddress + thread;
	int idx = thread * (hightY + padd);
	if (thread < (tileX + padd)){
		for (int i=0; i<hightY+padd; i++){
			table[idx+i] = dev_table[pos];
			pos += (rowsize + padd - i - 1);
		}	
	}
}

__device__ void moveToGlobal(int *table, int *dev_table, int tileAddress, int thread, int tileX, int rowsize, int hightY, int padd){
	int pos = tileAddress + thread;
	int idx = thread * (hightY + padd);
	if (thread < (tileX + padd)){
		for (int i=0; i<hightY+padd; i++){
			dev_table[pos] = table[idx+i];
			pos += (rowsize + padd - i - 1);
		}	
	}
}

__device__ void moveToShareRec(int *table, int *dev_table, int tileAddress, int thread, int tileX, int rowsize, int hightY, int segLengthX, int segLengthY){
	//This function is designed for the first and the last tiles, which can be treate as rectangular but not hyperlane.
	//Rectangular tile does not have bank conflict issue.
	int pos = tileAddress + thread;
	int idx = thread;
	if (thread < segLengthX){
		for (int i=0; i<segLengthY; i++){
			table[idx] = dev_table[pos];
			pos += rowsize;
			idx += segLengthX;
		}	
	}
}

__device__ void moveToGlobalRec(int *table, int *dev_table, int tileAddress, int thread, int tileX, int rowsize, int hightY, int segLengthX, int segLengthY){
	int pos = tileAddress + thread;
	int idx = thread;
	for (int i=0; i<segLengthY; i++){
		if (thread <= i){
			dev_table[pos] = table[idx];
			pos += rowsize;
			idx += segLengthX;
		}
	}	
}

__device__ void flagRead(int curBatch, volatile int *dev_lock, int thread, int idx, int YoverX, int xseg){
	if (thread == 0){
		int limit = min(idx+YoverX, xseg);
	 	while(dev_lock[curBatch] < limit){
		}
	}
	__syncthreads();
}

__device__ void flagWrite(int curBatch, volatile int *dev_lock, int thread){
	if (thread == 0){
		dev_lock[curBatch+1] += 1;
	}
	__syncthreads();
}

__global__ void GPU(int *dev_table, int *dev_arr1, int *dev_arr2, volatile int *dev_lock, int curBatch, int curStartAddress, int rowtiles, int hightY,
			int resX, int tileX, int tileY, int padd, int rowStartOffset, int rowsize, int xseg, int yseg, int YoverX, int n1, int n2) 
	//We assume row size n2 is the multiple of 32 and can be completely divided by tileX.
	//on K40, tile size is max to 48K, which is 128*96; on pascal and volta, tile size is max to 64K which is 128*128
	//This code, length of x axis cannot be larger than y axis for each tile.
	//For each row, the first tile and the last tile are computed separately from the other tiles.
	//No padding added, thus the first tile of each row and the first row requires statement check to set dependency to 0 for the edge elements.
	//size of the shared memory is determined by the GPU architecture.
	
	__shared__ int table[12288];

	int thread = threadIdx.x;
	int tilCount = 0, tileAddress;
	int tileStartOffset, lvlStartAddress; 
	int glbStartX;
	int segLengthX = tileX + padd;
	int segLengthY = tileY + padd;
	int arrX = 0;
	int arrY = curBatch * tileY;
	int idxx, idxy, idx;
	int tile = 1;
//***********************************************************************************************************************************
	//processing the first tile of the row
	flagRead(curBatch, dev_lock, thread, tile, YoverX, xseg);
	glbStartX = curStartAddress;
	
	moveToShareRec(&table[0], dev_table, glbStartX, thread, tileX, rowsize, hightY, segLengthX, segLengthY);				
	__syncthreads();
	__threadfence_block();

	//first tile is irregular, concurrency is changed from 1 to hightY
	//the x length and y length of the first tile and the last tile are equal.
	tileStartOffset = padd * segLengthX + padd;
	for (int i=0; i<hightY; i++){
		lvlStartAddress = tileStartOffset + i;
		
		if(thread <= i){
			idx = lvlStartAddress + segLengthX * thread - thread;
			idxx = arrX + (i - thread);
			idxy = arrY + thread;
			table[idx] = max(table[idx-1], table[idx-segLengthX]);
			if(dev_arr1[idxx] == dev_arr2[idxy]){
				table[idx] = table[idx-segLengthX-1] + 1;	
			}							
		}
		__syncthreads();
	}
	
	moveToShareRec(&table[0], dev_table, glbStartX, thread, tileX, rowsize, hightY, segLengthX, segLengthY);				
	__syncthreads();	
	
	//update the tile beginning pos for the next tile.
	glbStartX += tileY;
	arrX = tileY;
	flagWrite(curBatch, dev_lock, thread);
//***********************************************************************************************************************************
	//hyperlane tiles, assume all hyperlane tiles have segLengthx == tileX
	tileStartOffset = padd * segLengthY + padd;
	for (tile = 2; tile < rowTiles; tile++){
		flagRead(curBatch, dev_lock, thread, tile, YoverX, xseg);
		moveToShare(table, dev_table, glbStartX, thread, tileX, rowsize, hightY, padd);
		__syncthreads();
		__threadfence_block();
		
		lvlStartAddress = tileStartOffset;
		if (thread < tileY){
			for (int i=0; i<tileX; i++){
				idx = lvlStartAddress + thread;
				idxx = arrX + (i - thread);
				idxy = arrY + thread;
				table[idx] = max(table[idx-segLengthY], table[idx-segLengthY-1]);
				if(dev_arr1[idxx] == dev_arr2[idxy]){
					table[idx] = table[idx-segLengthY-segLengthY-1] + 1;
				}	
 			
				lvlStartAddress += segLengthY;
				__syncthreads();
			}
			//need modification, only copy the new updated elements back to the global memory. Also modify moveToGlobalRec
			moveToGlobal(table, dev_table, glbStartX, thread, tileX, rowsize, hightY, padd);
		}
		__syncthreads();
		//update the tile beginning pos for the next tile.
		glbStartX += tileX;
		arrX+=tileX;

		flagWrite(curBatch, dev_lock, thread);
	}

//************************************************************************************************************************************
	//the last tile, which is a half of the rectangular
	flagRead(curBatch, dev_lock, thread, rowTiles, YoverX, xseg);
	
	moveToShareRec(&table[0], dev_table, glbStartX, thread, tileX, rowsize, hightY, segLengthX, segLengthY);				
	__syncthreads();
	__threadfence_block();

	//last tile is irregular, concurrency is changed from hightY-1 to 1
	//the x length and y length of the first tile and the last tile are equal.
	tileStartOffset = (padd + 1) * segLengthX + padd;
	for (int i=hightY-1; i>0; i--){
		lvlStartAddress = tileStartOffset + i;
		
		if(thread <= i){
			idx = lvlStartAddress + segLengthX * thread - thread;
			idxx = n1 - thread;
			idxy = (arrY + 1) + thread;
			table[idx] = max(table[idx-1], table[idx-segLengthX]);
			if(dev_arr1[idxx] == dev_arr2[idxy]){
				table[idx] = table[idx-segLengthX-1] + 1;	
			}							
		}
		tileStartOffset += segLengthX;
		__syncthreads();
	}
	
	moveToShareRec(&table[0], dev_table, glbStartX, thread, tileX, rowsize, hightY, segLengthX, segLengthY);				
	__syncthreads();	
	
	flagWrite(curBatch, dev_lock, thread);
}

void checkGPUError(cudaError err){
	if (cudaSuccess != err){
		printf("CUDA error in file %s, in line %i: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

int LCS(int n1, int n2, int *arr1, int *arr2){
	int lcslength;
	int padding = 2;

	//tileY must be larger than tileX
	int tileX = 64;
	int tileY = 96;
	int rowsize = poolsize + n2;
	int colsize = poolsize + n1;

	int *dev_table, *dev_arr1, *dev_arr2;
	volatile int *dev_lock;

//	int *table;

//	table = new int[(n1+poolsize) * rowsize];

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
	int threadPerBlock = tileY + padding;
	int blockPerGrid = 1;
	int numStream = 15;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	//For hyperlane tiles, if tileX!=tileY, the X length of the first tile and the last tile are equal to tileY.
//	int xseg = (n1+tileX-1) / tileX;
	int xseg = ((n1-tileY) + tileX - 1) / tileX + 1;
	int yseg = (n2+tileY-1) / tileY;
	int startSegX, startSegY;	

	cudaMalloc(&dev_lock, (yseg + 1) * sizeof(int));
	cudaMemset(dev_lock, 0, (yseg + 1) * sizeof(int));
	cudaMemset(dev_lock, xseg+1, sizeof(int));
	cudaStream_t stream[numStream];
	for (int s=0; s<numStream; s++)
		cudaStreamCreate(&stream[s]);
	
	//instead of calling kernels along anti-diagonals, we now schedule kernels for each row.
	//We assume X axis is completly divided by tileX
	for(int curBatch = 0; curBatch < yseg; curBatch++){
		int curSMStream = curBatch % yset;
		int resY = n1 - curBatch * tileY;
		int hightY = min(tileY, resY);i
		int resX = (n2 - tileY) % tileX;
		int curStartAddress = curBatch * tileY * rowsize;
		int rowStartOffset = padding * rowsize + padding;
		int rowtiles = xseg + 1;
	
		GPU<<<blockPerGrid, threadPerBlock, 0, stream[curSMStream]>>>(&dev_table[curStartAddress], dev_arr1, dev_arr2, dev_lock, curBatch, curStartAddress, rowtiles, hightY, resX, tileX, tileY,  padding, rowStartOffset, rowsize, xseg, yseg, tileY/tileX, n1, n2);			
	}
	
	cudaMemcpy(&lcslength, &dev_table[tablesize-1], sizeof(int), cudaMemcpyDeviceToHost);
//	cudaMemcpy(table, dev_table, (n1+poolsize)*rowsize*sizeof(int), cudaMemcpyDeviceToHost);

	//display table
/*	cout << "full table: " << endl;
	for (int i=0; i<n1+poolsize; i++){
		for (int j=0; j<n2+poolsize; j++){
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
	cudaFree(dev_lock);
//	delete[] table;

	return lcslength;
}

