#include<iostream>
#include<cstdlib>
#include<fstream>
#include<string>
#include<sys/time.h>

//#define ALL
//#define DEBUG
//#define DEBUG1
//#define DEBUG2
//#define DEBUG3

using namespace std;
__device__ int row = 0;

__device__ void moveToSharePrev(volatile int *table, volatile int *dev_table, int tileAddress, int warp, int thread, int idx, int tileX, int tileY, int rowsize, int segLengthX, int segLengthY, int warpbatch, int warpsize){
	//potential bank conflict for accessing the data of each anti-diagonal
	//to avoid bank conflict, have to adjust the memory layout by introducing dummy elements.
	//padding elements can be used as the dummy elements, but the number of padding of each dimension has to be an odd number.
/*
	int pos = tileAddress + thread;
	int idx = thread;
	if (thread < segLengthX){
		for (int i=0; i<segLengthY; i++){
			table[idx] = dev_table[pos];
			pos += (rowsize - 1);
			idx += segLengthX;
		}	
	}
*/
	int glbpos = tileAddress + (rowsize - 1) + warp * (rowsize - 1) + idx;
	int shrpos = warp * warpsize + idx;
	if (thread < tileX)
		upperbuff[thread] = dev_table[tileAddress + paddX - 1 + thread];
	for (; warp < segLengthY; warp+=warpbatch){
		table[shrpos] = dev_table[glbpos];
		shrpos += (warpbatch * warpsize);
		glbpos += (warpbatch * (rowsize - 1) );
	}

}

__device__ void moveToShareUpper(volatile int *upperbuff, volatile int *dev_table, int tileAddress, int idx, int tileX, int offset){
		upperbuff[idx] = dev_table[tileAddress + paddX - 1 + offset + (rowsize - 1) * warp * warpsize + idx];
}

__device__ void moveToGlobal(volatile int *upperbuff, volatile int *dev_table, int tileAddress, int thread, int tileX, int tileY, int rowsize, int paddX, int segLengthX, int segLengthY, int warpbatch){
/*
	int idx = thread + segLengthX + paddX;
	int pos = tileAddress + (rowsize-1) + paddX + thread;
	//If y dimension cannot be completely divided by tileY, this code causes errors.
	if (thread < tileX){
		for (int i=0; i<tileY; i++){
			dev_table[pos] = table[idx];
			pos += (rowsize - 1);
			idx += segLengthX;
		}	
	}
*/
	int idx = thread % 32;
	int warpidx = thread / 32;
	int glbpos = tileAddress + (rowsize - 1) + paddX + warpidx * (rowsize - 1);
	int shrpos = segLengthX + paddX + warpidx * segLengthX;

	for (; warpidx < tileY; warpidx += warpbatch){
		for (int i = idx; i < tileX; i += 32){
			dev_table[glbpos + i] = table[shrpos + i];	
		}
		shrpos += (warpbatch * segLengthX);
		glbpos += (warpbatch * (rowsize - 1) );
	}

}

__device__ void moveToShareRec(volatile int *table, volatile int *dev_table, int tileAddress, int thread, int tileX, int tileY, int rowsize, int segLengthX, int segLengthY, int warpbatch){
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


__device__ void moveToGlobalRec(volatile int *table, volatile int *dev_table, int tileAddress, int thread, int tileX, int tileY, int rowsize, int segLengthX, int segLengthY, int paddX){
	int pos = tileAddress + rowsize + paddX + thread;
	int idx = segLengthX + paddX + thread;
	if (thread < tileX){
		for (int i=0; i<tileY; i++){
			dev_table[pos] = table[idx];
			pos += rowsize;
			idx += segLengthX;
		}
	}	
}


__device__ void flagReadGlobal(int curBatch, volatile int *dev_lock, int thread, int idx, int YoverX, int xseg){
	if (thread == 0){
		int limit = min(idx+YoverX, xseg);
/*
		printf("curBatch: %d, tile: %d, limit: %d, dev_lock[curBatch]: %d\n", curBatch, idx, limit, dev_lock[curBatch]);
*/
	 	while(dev_lock[curBatch] < limit){
		}
/*		printf("curBatch: %d, tile: %d, is permit to proceed, dev_lock[curBatch]: %d\n", curBatch, idx, dev_lock[curBatch]);
*/
	}
	__syncthreads();
}

__device__ void flagWriteGlobal(int curBatch, volatile int *dev_lock, int thread){
	if (thread == 0){
		dev_lock[curBatch+1] += 1;
	}
	__syncthreads();
}

__global__ void GPU(volatile int *dev_table, int *dev_arr1, int *dev_arr2, volatile int *dev_lock, int curBatch, int curStartAddress, int rowtiles, int resX, int paddX, int paddY, int rowStartOffset, int rowsize, int colsize, int xseg, int yseg, int YoverX, int n1, int n2, int warpbatch){ 
	//We assume row size n2 is the multiple of 32 and can be completely divided by tileX.
	//on K40, tile size is max to 48K, which is 128*96; on pascal and volta, tile size is max to 64K which is 128*128
	//This code, length of x axis cannot be larger than y axis for each tile.
	//For each row, the first tile and the last tile are computed separately from the other tiles.
	//No padding added, thus the first tile of each row and the first row requires statement check to set dependency to 0 for the edge elements.
	//size of the shared memory is determined by the GPU architecture.
	
#ifdef DEBUG
	if (threadIdx.x == 0){
		printf("This is curBatch: %d, curStartAddress: %d\n", curBatch, curStartAddress);
	}
	__syncthreads();
#endif

	int warpsize = 32;
	int tileX = 64;
	int tileY = 128;
	volatile __shared__ int warplock[4];
	volatile __shared__ int	table[(tileY+1) * warpsize]; 
	__shared__ int arr1_seg[tileX * 2];
	__shared__ int arr2_seg[tileY];
	int thread = blockDim.x * blockIdx.x + threadIdx.x;
	int warp = thread / warpsize;
	int idx = thread % warpsize;
	int tileStartOffset, lvlStartAddress; 
	int glbStartX;
	int segLengthX = tileX + paddX;
	int segLengthY = tileY + paddY;
	volatile __shared__ int upperbuf[5 * tileX];
	int arrX = 0;
	int arrY = curBatch * tileY;
	int idxx, idxy, idx;
	int tile = 1;
	int prevx, prevy, prevxy, curr, iniUpp;
	
//***********************************************************************************************************************************
	//processing the first tile of the row
	glbStartX = curStartAddress;
	flagReadGlobal(curBatch, dev_lock, thread, tile, YoverX, xseg);
	
	int highY = tileY;
	int piece = tileY / tileX;

	for (int p = 0; p < piece; p++){
#ifdef DEBUG1
#ifdef ALL
		if (thread == 32 && curBatch == row ){	
#endif
#ifndef ALL
		if (thread == 32){
#endif
			printf("Before move data share memory. curBatch: %d, tile: %d, p: %d, xseg: %d, glbStartX: %d, segLengthX: %d, segLengthY: %d\n", curBatch, tile, p, xseg, glbStartX, segLengthX, segLengthY);
			//for (int i=0; i<segLengthY; i++){
			{	
				for (int j = 0; j<segLengthX; j++){
					printf("%d ", table[segLengthX+j]);
				}
				printf("\n");
				for (int j = 0; j<segLengthX; j++){
					printf("%d ", table[highY *segLengthX+j]);
				}
				printf("\n");
				//for (int j = 0; j<segLengthX; j++){
				//	printf("%d ", table[i * (segLengthY-1) *segLengthX+j]);
				//}
				//printf("\n");
			}
			printf("\n");
		}
		__syncthreads();
#endif	
	
		moveToShareRec(&table[0], dev_table, glbStartX, thread, tileX, tileY, rowsize, segLengthX, segLengthY, warpbatch);				
		__syncthreads();
//		__threadfence_system();

#ifdef DEBUG1
#ifdef ALL	
		if (thread == 32 && curBatch == row){	
#endif
#ifndef ALL
		if (thread == 32){
#endif	
			printf("Before computation, share memory. curBatch: %d, tile: %d, p: %d, xseg: %d, glbStartX: %d\n", curBatch, tile, p, xseg, glbStartX);
			//for (int i=0; i<segLengthY; i++){

			{	
				for (int j = 0; j<segLengthX; j++){
					printf("%d ", table[segLengthX+j]);
				}
				printf("\n");
				for (int j = 0; j<segLengthX; j++){
					printf("%d ", table[highY *segLengthX+j]);
				}
				printf("\n");
				//for (int j = 0; j<segLengthX; j++){
				//	printf("%d ", table[i * (segLengthY-1) *segLengthX+j]);
				//}
				//printf("\n");
			}
			printf("\n");
		}
		__syncthreads();
#endif

		//first tile is irregular, concurrency is changed from 1 to hightY
		//the x length and y length of the first tile and the last tile are equal.
		tileStartOffset = paddY * segLengthX + paddX;
		//length Y > length X, diagonal first element starts from Y axis instead of X axis for calculating the address.
		int concurrency;
		for (int i=0; i<highY; i++){
			lvlStartAddress = tileStartOffset + i * segLengthX;
			concurrency = min(tileX, i);
			if(thread <= concurrency){
				idx = lvlStartAddress - segLengthX * thread + thread;
				idxx = arrX + thread;
				idxy = arrY + i - thread;
				table[idx] = max(table[idx-1], table[idx-segLengthX]);
				if(dev_arr1[idxx] == dev_arr2[idxy]){
					table[idx] = table[idx-segLengthX-1] + 1;	
				}	
	//			printf("curBatch: %d, tile: %d, thread: %d, idx: %d, idxx: %d, x[idxx]: %d, idxy: %d, y[idxy]: %d, table[idx]: %d\n", curBatch, tile, thread, idx, idxx, dev_arr1[idxx], idxy, dev_arr2[idxy], table[idx]);						
			}
			__syncthreads();
		}

#ifdef DEBUG1
#ifdef ALL	
		if (thread == 32 && curBatch==row){	
#endif
#ifndef ALL
		if (thread == 32){
#endif
			printf("After computation, in shared memory\n");
			//for (int i=0; i<segLengthY; i++){
				{	
					for (int j = 0; j<segLengthX; j++){
						printf("%d ", table[segLengthX+j]);
					}
					printf("\n");
					for (int j = 0; j<segLengthX; j++){
						printf("%d ", table[highY *segLengthX+j]);
					}
					printf("\n");
					//for (int j = 0; j<segLengthX; j++){
					//	printf("%d ", table[i * (segLengthY-1) *segLengthX+j]);
					//}
					//printf("\n");
				}
			printf("\n");
		}
		__syncthreads();
#endif

		moveToGlobalRec(&table[0], dev_table, glbStartX, thread, tileX, tileY, rowsize, segLengthX, segLengthY, paddX);				
//		__threadfence_system();
//		__syncthreads();


#ifdef DEBUG1
#ifdef ALL
		if (thread == 32 && curBatch == row){
#endif
#ifndef ALL
		if (thread == 32){
#endif
			printf("After computation in global memory. curBatch: %d, tile: %d, p: %d, glbStartX: %d\n", curBatch, tile, p, glbStartX);
			//for(int i=0; i<segLengthY; i++){
				for(int j=0; j<segLengthX; j++){	
					printf("%d ", dev_table[glbStartX + rowsize+j]);
				}
				printf("\n");
				for(int j=0; j<segLengthX; j++){	
					printf("%d ", dev_table[glbStartX + highY *rowsize+j]);
				}
				printf("\n");
				//for(int j=0; j<segLengthX; j++){	
				//	printf("%d ", dev_table[glbStartX + i * (segLengthY-1) *rowsize+j]);
				//}
				//printf("\n");
//			}
			printf("\n");
		}
		__syncthreads();
#endif
		arrX += tileX;		
		glbStartX += tileX;
		highY -= tileX;
//		__syncthreads();
	}
	//update the tile beginning pos for the next tile.
//	glbStartX += (tileY + 1);
	glbStartX += 1;
	arrX = tileY;
	flagWriteGlobal(curBatch, dev_lock, thread);
//***********************************************************************************************************************************
	//hyperlane tiles
	arr2_seg[thread] = dev_arr2[curBatch * tileY + thread];
	moveToSharePrev(&table[0], dev_table, glbStartX, warp, thread, idx, tileX, tileY, rowsize, segLengthX, segLengthY, warpbatch, warpsize);
	__syncthreads();

	prevx = table[(thread+1) * warpsize + paddX - 1];
	prevy = table[thread * warpsize + paddX - 2];
	prevxy = table[thread * warpsize + paddX - 3];

	for (tile = 2; tile < xseg; tile++){
		if (thread < tileX * 2)
			arr1_seg[thread] = dev_arr1[arrx - tileX + thread];
		flagReadGlobal(curBatch, dev_lock, thread, tile, YoverX, xseg);

		if (warp == 0)
			moveToShareUpper(&upperbuff[0], dev_table, glbStartX, thread, tileX, rowsize);
		__syncthreads();
		//inner loop for executing multiple 32*32 tiles in a row.
		for (int i=0; i< tileX; i+=warpsize){
			flagReadBlock(warp, &warplock[0], thread);
			for (int j=0; j<warpsize; j++){
				iniUpp = upperbuff[thread];
				
				if (thread == 0)
					prevy = iniUpp;
				curr = max(prevx, prevy);
				idxx = tileX + i - thread + j;
				idxy = thread;
				if (arr1_seg[idxx] == arr2_seg[idxy])
					curr = prevxy + 1;
				
				iniUpp = __shfl_down_sync(thread, iniUpp, 1, warpsize);
				prevx = curr;
				prevxy = prevy;
				prevy = __shfl_up_sync(thread, curr, 1, warpsize);
				
				if (thread == 31)
					upperbuf[(warp+1)*warpsize + j] = curr;   
			}
			flagWriteBlock();
		}
/*
#ifdef DEBUG2
#ifdef ALL
	if (thread == 0 && curBatch == row && tile<=3){	
#endif
#ifndef ALL
	if (thread == 0){
#endif
		printf("After computation, in shared memory.\n");
		//for (int i=0; i<segLengthY; i++){
			{	
				for (int j = 0; j<segLengthX; j++){
					printf("%d ", table[segLengthX+j]);
				}
				printf("\n");
				for (int j = 0; j<segLengthX; j++){
					printf("%d ", table[(segLengthY-tileX) * segLengthX+j]);
				}
				printf("\n");
				for (int j = 0; j<segLengthX; j++){
					printf("%d ", table[(segLengthY-1) *segLengthX+j]);
				}
				printf("\n");
				//for (int j = 0; j<segLengthX; j++){
				//	printf("%d ", table[i * (segLengthY-1) *segLengthX+j]);
				//}
				//printf("\n");
			}
		printf("\n");
	}
	__syncthreads();
#endif
*/
		//need modification, only copy the new updated elements back to the global memory. Also modify moveToGlobalRec
		moveToGlobal(&upperbuff[0], dev_table, glbStartX, thread, tileX, tileY, rowsize, paddX, segLengthX, segLengthY, warpbatch);
//		moveToGlobal(&table[paddX], dev_table, glbStartX + paddX, thread, tileX, rowsize, segLengthX, segLengthY);
		
//		__threadfence_system();
//		__syncthreads();

#ifdef DEBUG2
#ifdef ALL
		if (thread == 0 && curBatch == row && tile <= 3){
#endif
#ifndef ALL
		if (thread == 0){
#endif	
			printf("After computation, global memory. curBatch: %d, tile: %d, glbStartX: %d\n", curBatch, tile, glbStartX);
			//for(int i=0; i<segLengthY; i++){
				for(int j=0; j<segLengthX; j++){	
					printf("%d ", dev_table[glbStartX + rowsize+j - 1]);
				}
				printf("\n");
				for(int j=0; j<segLengthX; j++){	
					printf("%d ", dev_table[glbStartX + (segLengthY-tileX) *rowsize+j - (segLengthY-tileX-1)]);
				}
				printf("\n");
				for(int j=0; j<segLengthX; j++){	
					printf("%d ", dev_table[glbStartX + (segLengthY-1) *rowsize+j - (segLengthY-1)]);
				}
				printf("\n");
				//for(int j=0; j<segLengthX; j++){	
				//	printf("%d ", dev_table[glbStartX + i * (segLengthY-1) *rowsize+j - i * (segLengthY-1)]);
				//}
				//printf("\n");
		//	}
			printf("\n");
		}
//		__syncthreads();
#endif

		//update the tile beginning pos for the next tile.
		glbStartX += tileX;
		arrX+=tileX;
		flagWriteGlobal(curBatch, dev_lock, thread);
	}

//************************************************************************************************************************************
	//the last tile, which is a half of the rectangular
	flagReadGlobal(curBatch, dev_lock, thread, xseg, YoverX, xseg);
	glbStartX = curStartAddress + rowsize - tileY - paddX;
	
	piece = tileY / tileX;
	highY = tileX;
	
	for (int p=0; p<piece; p++){
#ifdef DEBUG3
#ifdef ALL	
		if (thread == 0 && curBatch == row){
#endif
#ifndef ALL
		if (thread == 0){
#endif
			printf("Before computation global memory. curBatch: %d, tile: %d, p: %d, glbStartX: %d\n", curBatch, tile, p, glbStartX);
			//for(int i=0; i<segLengthY; i++){
				for(int j=0; j<segLengthX; j++){	
					printf("%d ", dev_table[glbStartX + (segLengthY - highY - 1) * rowsize+j]);
				}
				printf("\n");
				for(int j=0; j<segLengthX; j++){	
					printf("%d ", dev_table[glbStartX + (segLengthY - highY ) * rowsize+j]);
				}
				printf("\n");
				for(int j=0; j<segLengthX; j++){	
					printf("%d ", dev_table[glbStartX + (segLengthY - highY + 1) * rowsize+j]);
				}
				printf("\n");
				for(int j=0; j<segLengthX; j++){	
					printf("%d ", dev_table[glbStartX + (segLengthY-1) *rowsize+j]);
				}
				printf("\n");
				//for(int j=0; j<segLengthX; j++){	
				//	printf("%d ", dev_table[glbStartX + i* (segLengthY-1) *rowsize+j]);
				//}
				//printf("\n");
			//}
			printf("\n");
		}
		__syncthreads();
#endif	
		moveToShareRec(&table[0], dev_table, glbStartX, thread, tileX, tileY, rowsize, segLengthX, segLengthY, warpbatch);				
		__syncthreads();
//		__threadfence_block();

#ifdef DEBUG3
#ifdef ALL
			if (thread == 0 && curBatch == row){
#endif
#ifndef ALL
			if (thread == 0){
#endif
				printf("last tile share memory before computation, glbStartX: %d, p: %d, rowsize: %d, segLengthY: %d\n", glbStartX, p, rowsize, segLengthY);	
			//	for (int i=0; i<segLengthY; i++){
				{	
					for (int j = 0; j<segLengthX; j++){
						printf("%d ", table[(segLengthY-highY - 1) * segLengthX+j]);
					}
					printf("\n");
					for (int j = 0; j<segLengthX; j++){
						printf("%d ", table[(segLengthY- highY) *segLengthX+j]);
					}
					printf("\n");
					for (int j = 0; j<segLengthX; j++){
						printf("%d ", table[(segLengthY- highY + 1) *segLengthX+j]);
					}
					printf("\n");
					for (int j = 0; j<segLengthX; j++){
						printf("%d ", table[(segLengthY-1) *segLengthX+j]);
					}
					printf("\n");
					//for (int j = 0; j<segLengthX; j++){
					//	printf("%d ", table[i * (segLengthY-1) *segLengthX+j]);
					//}
					//printf("\n");
				}
				printf("\n");
			}
			__syncthreads();
#endif

		//last tile is irregular, concurrency is changed from hightY-1 to 1
		//the x length and y length of the first tile and the last tile are equal.
		int concurrency;
		tileStartOffset = segLengthX * (segLengthY - highY);
		for (int i=highY; i>0; i--){
			lvlStartAddress = tileStartOffset + segLengthX - 1;
			concurrency = min(tileX, i);	
			if(thread < concurrency){
				idx = lvlStartAddress + segLengthX * thread - thread;
				idxx = (n1 - tileY + highY - 1) - thread;
				//idxy = (n2 - i) + thread;
				idxy = (arrY + tileY - i) + thread;
				table[idx] = max(table[idx-1], table[idx-segLengthX]);
				if(dev_arr1[idxx] == dev_arr2[idxy]){
					table[idx] = table[idx-segLengthX-1] + 1;	
				}							
			}
			tileStartOffset += segLengthX;
			__syncthreads();
		}
	
		moveToGlobalRec(&table[0], dev_table, glbStartX, thread, tileX, tileY, rowsize, segLengthX, segLengthY, paddX);				

#ifdef DEBUG3
#ifdef ALL	
		if (thread == 0 && curBatch == row){	
#endif
#ifndef ALL
		if (thread == 0){
#endif	
			__syncthreads();
			printf("After computation, in shared memory\n");
			//for (int i=0; i<segLengthY; i++){
				{	
					for (int j = 0; j<segLengthX; j++){
						printf("%d ", table[(segLengthY-highY - 1) * segLengthX+j]);
					}
					printf("\n");
					for (int j = 0; j<segLengthX; j++){
						printf("%d ", table[(segLengthY- highY) *segLengthX+j]);
					}
					printf("\n");
					for (int j = 0; j<segLengthX; j++){
						printf("%d ", table[(segLengthY- highY + 1) *segLengthX+j]);
					}
					printf("\n");
					for (int j = 0; j<segLengthX; j++){
						printf("%d ", table[(segLengthY-1) *segLengthX+j]);
					}
					printf("\n");
					//for (int j = 0; j<segLengthX; j++){
					//	printf("%d ", table[i * (segLengthY-1) *segLengthX+j]);
					//}
					//printf("\n");
				}
			printf("\n");
		}
		__syncthreads();
#endif


#ifdef DEBUG3
#ifdef ALL
		if (thread == 0 && curBatch == row){
#endif
#ifndef ALL
		if (thread == 0){
#endif	
			printf("After computation. global memory. curBatch: %d, tile: %d, p: %d, glbStartX: %d\n", curBatch, tile, p, glbStartX);
			//for(int i=0; i<segLengthY; i++){
				for(int j=0; j<segLengthX; j++){	
					printf("%d ", dev_table[glbStartX + rowsize+j]);
				}
				printf("\n");
				for(int j=0; j<segLengthX; j++){	
					printf("%d ", dev_table[glbStartX + 2 * rowsize+j]);
				}
				printf("\n");
				for(int j=0; j<segLengthX; j++){	
					printf("%d ", dev_table[glbStartX + (segLengthY - highY) * rowsize+j]);
				}
				printf("\n");
				for(int j=0; j<segLengthX; j++){	
					printf("%d ", dev_table[glbStartX + (segLengthY-1) *rowsize+j]);
				}
				printf("\n");
				//for(int j=0; j<segLengthX; j++){	
				//	printf("%d ", dev_table[glbStartX + i * (segLengthY-1) *rowsize+j]);
				//}
				//printf("\n");
			//}
			printf("\n");
		}
		__syncthreads();
#endif
		glbStartX += tileX;
		highY += tileX;
	}
	
	flagWriteGlobal(curBatch, dev_lock, thread);
}

void checkGPUError(cudaError err){
	if (cudaSuccess != err){
		printf("CUDA error in file %s, in line %i: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

int LCS(int n1, int n2, int *arr1, int *arr2, int paddX, int paddY, int *table){
	cudaSetDevice(0);	
	int lcslength;

	//tileY must be larger than tileX
	int tileX = 64;
	int tileY = 128;
	int rowsize = paddX + n2;
	int colsize = paddY + n1;

	int *dev_arr1, *dev_arr2;
	volatile int *dev_table, *dev_lock;
	int *lock;
	size_t freeMem, totalMem;
	
	
	cudaMemGetInfo(&freeMem, &totalMem);
	int tablesize = colsize * rowsize;
	cout << "current GPU memory info FREE: " << freeMem << " Bytes, Total: " << totalMem << " Bytes.";
	cout << "colsize: " << colsize << ", rowsize: " << rowsize << ", allocates: " << tablesize * sizeof(int)<< " Bytes." << endl;
	cudaError err = cudaMalloc(&dev_table, tablesize * sizeof(int));
	checkGPUError(err);
	
	cudaMalloc(&dev_arr1, n1*sizeof(int));
	cudaMalloc(&dev_arr2, n2*sizeof(int));

	cudaMemset((void*)dev_table, 0, tablesize * sizeof(int));
	cudaMemcpy(dev_arr1, arr1, n1*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_arr2, arr2, n2*sizeof(int), cudaMemcpyHostToDevice);

//	int threadPerBlock = max(tileY + 32, tileX + 32);
	int threadPerBlock = 1024;
	int blockPerGrid = 1;
	int numStream = 28;
	int warpbatch = threadPerBlock / 32;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	//For hyperlane tiles, if tileX!=tileY, the X length of the first tile and the last tile are equal to tileY.
//	int xseg = (n1+tileX-1) / tileX;
	int xseg = ((n1-tileY) + tileX - 1) / tileX + 2;
	int yseg = (n2+tileY-1) / tileY;

	lock = new int[yseg+1];
	lock[0] = xseg+1;
	for (int i=1; i<yseg+1; i++)
		lock[i] = 0;
	cudaMalloc(&dev_lock, (yseg+1) * sizeof(int));	
//	cudaMemset((void*)dev_lock, 0, (yseg + 1) * sizeof(int));
//	cudaMemset((void*)dev_lock, xseg+1, sizeof(int));
	cudaMemcpy((void*)dev_lock, lock, (yseg+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaStream_t stream[numStream];
	for (int s=0; s<numStream; s++)
		cudaStreamCreate(&stream[s]);
	
	//instead of calling kernels along anti-diagonals, we now schedule kernels for each row.
	//We assume X axis is completly divided by tileX
	for(int curBatch = 0; curBatch < yseg; curBatch++){
		int curSMStream = curBatch % numStream;
		//int resY = n1 - curBatch * tileY;
		int resX = (n2 - tileY) % tileX;
		int curStartAddress = curBatch * tileY * rowsize;
		int rowStartOffset = paddY * rowsize + paddX;
		int rowtiles = xseg + 1;
//		cout << endl << "curBatch: " << curBatch << ", yseg: " << yseg << endl;	
		GPU<<<blockPerGrid, threadPerBlock, 0, stream[curSMStream]>>>(dev_table, dev_arr1, dev_arr2, dev_lock, curBatch, curStartAddress, rowtiles, resX, tileX, tileY,  paddX, paddY, rowStartOffset, rowsize, colsize, xseg, yseg, tileY/tileX, n1, n2, warpbatch);			
//		GPU<<<blockPerGrid, threadPerBlock>>>(dev_table, dev_arr1, dev_arr2, dev_lock, curBatch, curStartAddress, rowtiles, resX, tileX, tileY,  paddX, paddY, rowStartOffset, rowsize, colsize, xseg, yseg, tileY/tileX, n1, n2);			
		checkGPUError( cudaGetLastError() );
//		cudaDeviceSynchronize();
	}
	cudaMemcpy(&lcslength, (void*)&dev_table[tablesize-1], sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(table, (void*)dev_table, tablesize*sizeof(int), cudaMemcpyDeviceToHost);

#ifdef DEBUG
	//display table
	cout << "grid size: " << blockPerGrid << ", block size: " << threadPerBlock << ", full table: " << endl;
	for (int i=0; i<colsize; i++){
		for (int j=0; j<rowsize; j++){
			cout << table[i * rowsize + j] << " ";
		}
		cout << endl;
	}
	
#endif
	for (int s=0; s<numStream; s++)
		cudaStreamDestroy(stream[s]);
	
	cudaFree(dev_arr1);
	cudaFree(dev_arr2);
	cudaFree((void*)dev_table);
	cudaFree((void*)dev_lock);
	delete[] lock;

	return lcslength;
}

