#include<iostream>
#include<cstdlib>
#include<fstream>
#include<string>
#include<sys/time.h>

typedef unsigned long long int UINT;

using namespace std;
__device__ int row = 1;

__device__ void moveToShare(int *table, volatile int *dev_table, int tileAddress, int thread, int tileX, int rowsize, int hightY, int segLengthX, int segLengthY){
	//potential bank conflict for accessing the data of each anti-diagonal
	//to avoid bank conflict, have to adjust the memory layout by introducing dummy elements.
	//padding elements can be used as the dummy elements, but the number of padding of each dimension has to be an odd number.
	int pos = tileAddress + thread;
	int idx = thread;
	if (thread < segLengthX){
		for (int i=0; i<segLengthY; i++){
//printf("thread: %d, pos: %d, idx: %d\n", thread, pos, idx);			
			
			table[idx] = dev_table[pos];
			pos += (rowsize - 1);
			idx += segLengthX;
		}	
	}
}

__device__ void moveToGlobal(int *table, volatile int *dev_table, int tileAddress, int thread, int tileX, int rowsize, int hightY, int segLengthX, int segLengthY){
	int pos = tileAddress + thread;
	int idx = thread;
	//If y dimension cannot be completely divided by tileY, this code causes errors.
	if (thread < segLengthX){
		for (int i=0; i<segLengthY; i++){

//printf("thread: %d, segLengthX: %d, pos: %d, idx: %d\n", thread, segLengthX, pos, idx);			
			dev_table[pos] = table[idx];
			pos += (rowsize - 1);
			idx += segLengthX;
		}	
	}
}

__device__ void moveToShareRec(int *table, volatile int *dev_table, int tileAddress, int thread, int tileX, int rowsize, int hightY, int segLengthX, int segLengthY){
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

__device__ void moveToGlobalRec(int *table, volatile int *dev_table, int tileAddress, int thread, int tileX, int rowsize, int hightY, int segLengthX, int segLengthY){
	int pos = tileAddress + thread;
	int idx = thread;
	if (thread < segLengthX){
		for (int i=0; i<segLengthY; i++){
			dev_table[pos] = table[idx];
			pos += rowsize;
			idx += segLengthX;
		}
	}	
}

__device__ void moveToShareLast(int *table, volatile int *dev_table, int tileAddress, int thread, int tileX, int rowsize, int hightY, int segLengthX, int segLengthY){
	//This function is designed for the last tiles, which can be treate as rectangular but not hyperlane.
	//Rectangular tile does not have bank conflict issue.
	int pos = tileAddress + thread;
	int idx = thread;
	if (thread < hightY){
		for (int i=0; i<hightY; i++){
			table[idx] = dev_table[pos];
			pos += rowsize;
			idx += hightY;
		}	
	}
}

__device__ void moveToGlobalLast(int *table, volatile int *dev_table, int tileAddress, int thread, int tileX, int rowsize, int hightY, int segLengthX, int segLengthY){
	int pos = tileAddress + thread;
	int idx = thread;
	if (thread < hightY){
		for (int i=0; i<hightY; i++){
			dev_table[pos] = table[idx];
			pos += rowsize;
			idx += hightY;
		}
	}
}

__device__ void flagRead(int curBatch, volatile int *dev_lock, int thread, int idx, int YoverX, int xseg){
	if (thread == 0){
		int limit = min(idx+YoverX, xseg);
		
		printf("curBatch: %d, tile: %d, limit: %d, dev_lock[curBatch]: %d\n", curBatch, idx, limit, dev_lock[curBatch]);

	 	while(dev_lock[curBatch] < limit){
		}

		printf("curBatch: %d, tile: %d, is permit to proceed, dev_lock[curBatch]: %d\n", curBatch, idx, dev_lock[curBatch]);
	}
	__syncthreads();
}

__device__ void flagWrite(int curBatch, volatile int *dev_lock, int thread){
	if (thread == 0){
		dev_lock[curBatch+1] += 1;
	}
	__syncthreads();
}

__global__ void GPU(volatile int *dev_table, int *dev_arr1, int *dev_arr2, volatile int *dev_lock, int curBatch, int curStartAddress, int rowtiles, int hightY, int resX, int tileX, int tileY, int paddX, int paddY, int rowStartOffset, int rowsize, int colsize, int xseg, int yseg, int YoverX, int n1, int n2){ 
	//We assume row size n2 is the multiple of 32 and can be completely divided by tileX.
	//on K40, tile size is max to 48K, which is 128*96; on pascal and volta, tile size is max to 64K which is 128*128
	//This code, length of x axis cannot be larger than y axis for each tile.
	//For each row, the first tile and the last tile are computed separately from the other tiles.
	//No padding added, thus the first tile of each row and the first row requires statement check to set dependency to 0 for the edge elements.
	//size of the shared memory is determined by the GPU architecture.
	
	__shared__ int table[12288];

	int thread = threadIdx.x;
	int tileStartOffset, lvlStartAddress; 
	int glbStartX;
	int segLengthX = tileX + paddX;
	int segLengthY = tileY + paddY;
	int arrX = 0;
	int arrY = curBatch * tileY;
	int idxx, idxy, idx;
	int tile = 1;

//***********************************************************************************************************************************
	//processing the first tile of the row
	glbStartX = curStartAddress;
	flagRead(curBatch, dev_lock, thread, tile, YoverX, xseg);
	
	if (thread == 0 && curBatch == row){
		printf("Before computation. curBatch: %d, tile: %d, glbStartX: %d\n", curBatch, tile, glbStartX);
		for(int i=0; i<colsize; i++){
			for(int j=0; j<rowsize; j++){	
				printf("%d ", dev_table[i*rowsize+j]);
			}
			printf("\n");
		}
		printf("\n");
	}


	moveToShareRec(&table[0], dev_table, glbStartX, thread, tileX, rowsize, hightY, segLengthX, segLengthY);				
	__syncthreads();
	__threadfence_block();
	
	if (thread == 0 && curBatch == row){	
		for (int i=0; i<segLengthY; i++){
			for (int j = 0; j<segLengthX; j++){
				printf("%d ", table[i*segLengthX+j]);
			}
			printf("\n");
		}
		printf("\n");
	}

	//first tile is irregular, concurrency is changed from 1 to hightY
	//the x length and y length of the first tile and the last tile are equal.
	tileStartOffset = paddY * segLengthX + paddX;
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
//			printf("curBatch: %d, tile: %d, thread: %d, idx: %d, idxx: %d, x[idxx]: %d, idxy: %d, y[idxy]: %d, table[idx]: %d\n", curBatch, tile, thread, idx, idxx, dev_arr1[idxx], idxy, dev_arr2[idxy], table[idx]);						
		}
		__syncthreads();
	}

	
	if (thread == 0 && curBatch==row){	
		printf("After computation, in shared memory\n");
		for (int i=0; i<segLengthY; i++){
			for (int j = 0; j<segLengthX; j++){
				printf("%d ", table[i*segLengthX+j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	
	moveToGlobalRec(&table[0], dev_table, glbStartX, thread, tileX, rowsize, hightY, segLengthX, segLengthY);				
	__threadfence();
	__syncthreads();

	if (thread == 0 && curBatch == row){
		printf("After computation. curBatch: %d, tile: %d, glbStartX: %d\n", curBatch, tile, glbStartX);
		for(int i=0; i<colsize; i++){
			for(int j=0; j<rowsize; j++){	
				printf("%d ", dev_table[i*rowsize+j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	
	//update the tile beginning pos for the next tile.
	glbStartX += segLengthY;
	arrX = tileY;
	flagWrite(curBatch, dev_lock, thread);
//***********************************************************************************************************************************
	//hyperlane tiles, assume all hyperlane tiles have segLengthx == tileX
	tileStartOffset = paddY * segLengthX + paddX;
	for (tile = 2; tile < xseg; tile++){
		flagRead(curBatch, dev_lock, thread, tile, YoverX, xseg);
	
//		printf("curBatch: %d, tile: %d, thread: %d is permit to read data from global.\n", curBatch, tile, thread);
		
		if (thread == 0 && curBatch == row){
			printf("Before computation. curBatch: %d, tile: %d, xseg: %d, glbStartX: %d\n", curBatch, tile, xseg, glbStartX);
			for(int i=0; i<colsize; i++){
				for(int j=0; j<rowsize; j++){	
					printf("%d ", dev_table[i*rowsize+j]);
				}
				printf("\n");
			}
			printf("\n");
		}
	
	
		moveToShare(table, dev_table, glbStartX, thread, tileX, rowsize, hightY, segLengthX, segLengthY);
		__syncthreads();
		__threadfence_block();


		if (thread == 0 && curBatch == row){	
			for (int i=0; i<segLengthY; i++){
				for (int j = 0; j<segLengthX; j++){
					printf("%d ", table[i*segLengthX+j]);
				}
				printf("\n");
			}
			printf("\n");
		}
		
		lvlStartAddress = tileStartOffset;
		if (thread < tileY){
			for (int i=0; i<tileX; i++){
				idx = lvlStartAddress + thread * segLengthX;
				idxx = arrX + (i - thread);
				idxy = arrY + thread;
				table[idx] = max(table[idx-1], table[idx-segLengthX-1]);
				if(dev_arr1[idxx] == dev_arr2[idxy]){
					table[idx] = table[idx-segLengthY-2] + 1;
				}	
 			
//			printf("curBatch: %d, thread: %d, idx: %d, idxx: %d, idxy: %d, table[idx]: %d\n", curBatch, thread, idx, idxx, idxy, table[idx]);						
				lvlStartAddress += 1;
				__syncthreads();
			}
		}
		//need modification, only copy the new updated elements back to the global memory. Also modify moveToGlobalRec
		moveToGlobal(table, dev_table, glbStartX, thread, tileX, rowsize, hightY, segLengthX, segLengthY);
		
		__threadfence();
		__syncthreads();
		//update the tile beginning pos for the next tile.
		glbStartX += tileX;
		arrX+=tileX;

	if (thread == 0 && curBatch == row){	
		printf("After computation, in shared memory\n");
		for (int i=0; i<segLengthY; i++){
			for (int j = 0; j<segLengthX; j++){
				printf("%d ", table[i*segLengthX+j]);
			}
			printf("\n");
		}
		printf("\n");
	}

		if (thread == 0 && curBatch == row){
			printf("After computation. curBatch: %d, tile: %d, glbStartX: %d\n", curBatch, tile, glbStartX);
			for(int i=0; i<colsize; i++){
				for(int j=0; j<rowsize; j++){	
					printf("%d ", dev_table[i*rowsize+j]);
				}
				printf("\n");
			}
			printf("\n");
		}

		flagWrite(curBatch, dev_lock, thread);
	}

//************************************************************************************************************************************
	//the last tile, which is a half of the rectangular
	flagRead(curBatch, dev_lock, thread, xseg, YoverX, xseg);
	
	if (thread == 0 && curBatch == row){
		printf("Before computation. curBatch: %d, tile: %d, glbStartX: %d\n", curBatch, tile, glbStartX);
		for(int i=0; i<colsize; i++){
			for(int j=0; j<rowsize; j++){	
				printf("%d ", dev_table[i*rowsize+j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	glbStartX += (rowsize + tileX - hightY);

	moveToShareLast(&table[0], dev_table, glbStartX, thread, tileX, rowsize, hightY, segLengthX, segLengthY);				
	__syncthreads();
	__threadfence_block();

		if (thread == 0 && curBatch == row){
			printf("last tile glbStartX: %d, rowsize: %d, hightY: %d\n", glbStartX, rowsize, hightY);	
			for (int i=0; i<hightY; i++){
				for (int j = 0; j<hightY; j++){
					printf("%d ", table[i*hightY+j]);
				}
				printf("\n");
			}
			printf("\n");
		}

	//last tile is irregular, concurrency is changed from hightY-1 to 1
	//the x length and y length of the first tile and the last tile are equal.
	tileStartOffset = hightY;
	for (int i=hightY-1; i>0; i--){
		lvlStartAddress = tileStartOffset + i;
		
		if(thread < i){
			idx = lvlStartAddress + segLengthX * thread - thread;
			idxx = n1 - thread;
			idxy = (arrY + 1) + thread;
			table[idx] = max(table[idx-1], table[idx-segLengthX]);
			if(dev_arr1[idxx] == dev_arr2[idxy]){
				table[idx] = table[idx-segLengthX-1] + 1;	
			}							
		}
		tileStartOffset += hightY;
		__syncthreads();
	}
	
	moveToGlobalLast(&table[0], dev_table, glbStartX, thread, tileX, rowsize, hightY, segLengthX, segLengthY);				
	__syncthreads();	
	
	if (thread == 0 && curBatch == row){	
		printf("After computation, in shared memory\n");
		for (int i=0; i<hightY; i++){
			for (int j = 0; j<hightY; j++){
				printf("%d ", table[i*hightY+j]);
			}
			printf("\n");
		}
		printf("\n");
	}

		if (thread == 0 && curBatch == row){
			printf("After computation. curBatch: %d, tile: %d, glbStartX: %d\n", curBatch, tile, glbStartX);
			for(int i=0; i<colsize; i++){
				for(int j=0; j<rowsize; j++){	
					printf("%d ", dev_table[i*rowsize+j]);
				}
				printf("\n");
			}
			printf("\n");
		}

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
	int paddX = 3, paddY = 1;

	//tileY must be larger than tileX
	int tileX = 2;
	int tileY = 2;
	int rowsize = paddX + n2;
	int colsize = paddY + n1;

	int *dev_arr1, *dev_arr2;
	int *dev_table, *dev_lock;

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

	int threadPerBlock = max(tileY + paddY, tileX + paddX);
	int blockPerGrid = 1;
	int numStream = 15;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	//For hyperlane tiles, if tileX!=tileY, the X length of the first tile and the last tile are equal to tileY.
//	int xseg = (n1+tileX-1) / tileX;
	int xseg = ((n1-tileY) + tileX - 1) / tileX + 2;
	int yseg = (n2+tileY-1) / tileY;

	cudaMalloc(&dev_lock, (yseg+1) * sizeof(int));	
	cudaMemset(dev_lock, 0, (yseg + 1) * sizeof(int));
	cudaMemset(dev_lock, xseg+1, sizeof(int));
	cudaStream_t stream[numStream];
	for (int s=0; s<numStream; s++)
		cudaStreamCreate(&stream[s]);
	
	//instead of calling kernels along anti-diagonals, we now schedule kernels for each row.
	//We assume X axis is completly divided by tileX
	for(int curBatch = 0; curBatch < yseg; curBatch++){
		int curSMStream = curBatch % yseg;
		int resY = n1 - curBatch * tileY;
		int hightY = min(tileY, resY);
		int resX = (n2 - tileY) % tileX;
		int curStartAddress = curBatch * tileY * rowsize;
		int rowStartOffset = paddY * rowsize + paddX;
		int rowtiles = xseg + 1;
	
		GPU<<<blockPerGrid, threadPerBlock, 0, stream[curSMStream]>>>(dev_table, dev_arr1, dev_arr2, dev_lock, curBatch, curStartAddress, rowtiles, hightY, resX, tileX, tileY,  paddX, paddY, rowStartOffset, rowsize, colsize, xseg, yseg, tileY/tileX, n1, n2);			
	}
	
	cudaMemcpy(&lcslength,&dev_table[tablesize-1], sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(table, dev_table, colsize*rowsize*sizeof(int), cudaMemcpyDeviceToHost);
/*
	//display table
	cout << "full table: " << endl;
	for (int i=0; i<colsize; i++){
		for (int j=0; j<rowsize; j++){
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
	delete[] table;

	return lcslength;
}

