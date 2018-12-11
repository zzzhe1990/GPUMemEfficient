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

__device__ void moveToShare(volatile int *table, volatile int *dev_table, int tileAddress, int thread, int tileX, int tileY, int rowsize, int segLengthX, int segLengthY, int warpbatch){
	int idx = thread % 32;
	int warpidx = thread / 32;
	int glbpos = tileAddress + (rowsize - 1) + warpidx * (rowsize - 1) + idx;
	int shrpos = segLengthX + warpidx * segLengthX + idx;
	if (thread < segLengthX)
		table[thread] = dev_table[tileAddress + thread];
	for (; warpidx < tileY; warpidx+=warpbatch){
		table[shrpos] = dev_table[glbpos];
		shrpos += (warpbatch * segLengthX);
		glbpos += (warpbatch * (rowsize - 1) );
	}

}

__device__ void moveToGlobal(volatile int *table, volatile int *dev_table, int tileAddress, int thread, int tileX, int tileY, int rowsize, int paddX, int segLengthX, int segLengthY, int warpbatch){
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


__device__ void moveToGlobalRec(volatile int *table, volatile int *dev_table, int tileAddress, int thread, int tileX, int tileY, int rowsize, int segLengthX, int segLengthY, int paddsize){
	int pos = tileAddress + rowsize + paddsize + thread;
	int idx = segLengthX + paddsize + thread;
	if (thread < tileX){
		for (int i=0; i<tileY; i++){
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

__global__ void GPU_Tile(int *dev_arr1, int *dev_arr2, volatile int *dev_lock, int curBatch, int curStartAddress, int tileX, int tileY, int padd, int stride, int rowStartOffset, int rowsize, int colsize, int xseg, int yseg, int n1, int n2, int warpbatch, int curSMStream, int preCurSMStream, int* inter_stream_dep, int inter_stream_dep_size, int tileT){ 
//We assume row size n1 is the multiple of 32 and can be completely divided by tileX.
//For each row, the first tile and the last tile are computed separately from the other tiles.
//size of the shared memory is determined by the GPU architecture.
//tileX is multiple times of 32 to maximize the cache read.		
#ifdef DEBUG
	if (threadIdx.x == 0){
		printf("This is curBatch: %d, curStartAddress: %d\n", curBatch, curStartAddress);
	}
	__syncthreads();
#endif
	//need two arrays: 1. tile raw data; 2. intra-stream dependence
	__shared__ int tile1[5120];
	__shared__ int tile2[5120];
	__shared__ int intra_dep[2047];

	int thread = blockDim.x * blockIdx.x + threadIdx.x;
	int segLengthX = tileX + stride + 1;
	int segLengthY = tileY + stride + 1;
	int tileIdx = 1;
	int tar;
	int idx1, idx2, xidx, yidx;
	int sumStencil;
	int tilePos;
	
//if this is the first batch of the current t tile, have to copy the related dependence data from global tile array into global inter-stream-dependence array.
//Challenges: when stream 0 is still working on one of the current t tiles but stream 2 already starts processing the first batch of the next t tiles. Copying the dependence data to arr[stream[0]] does not work.
//for the first and last batches, we need charactorized function to take care of the edge elements.

//***********************************************************************************************************************************
	lock_func_for_time();
//processing the first tile of each row, use the near-edge elements for the out-of-range dependence.
	//wait until it is safe to launch and execute the new batch.
	lock_func_for_tiles();
	//copy the base spatial data to shared memory for t=0.
	moveToShare(dev_arr1, &tile1[0], curBatch, tileIdx, segLengthX, segLengthY, tileX, tileY, stride);

	if (curBatch == 0){
	//for the first batch, use the near-edge elements for the out-of-range dependence.
		for (int tt=0; tt<tileT; tt++){
			copy intra_dep[tt] to shared tile;
			copy inter_dep[tt] to shared tile;
			for (; thread < tileX * tileY; thread += blockDim.x){
				sumStencil = 0;
				//because of the bias, xidx and yidx are the pos of new time elements.
				//thread % tileX and thread / tileX are pos of current cached elements.
				xidx = thread % tileX;
				yidx = thread / tileX; 
				//idx1 is the pos of the raw elements and the destination pos.
				//idx2 is the pos of the new calculated elements. The new elements shift
				//itself by "stride" along each dimension from its initial position.
				idx1 = (yidx+stride+1) * segLengthX + stride+1 + xidx;
				idx2 = (yidx+1) * segLengthX + 1 + xdix;		
	
				sumStencil = tile1[idx+1] + tile1[idx+tileX] + tile1[idx] + tile1[thread-1] + tile1[thread-tileX];
				if (xidx < 0)
					sumStencil = sumStencil + intra_dep[] + intra_dep[] + intra_dep[];
			
				if (rowx < )
					sumStencil = sumStencil - tile1[thread-] + inter_dep[];		

				tar = sumStencil / 5;

				if (rowx < stride+tt)
					tar = tile1[thread-];	
		
				tile2[thread] = tar;
				__syncthreads();
			}	
			if ()
				move from tile2 to intra_dep;
			__syncthreads();
			if ()
				move from tile2 to inter_dep;
			lock_func_for_tiles();
		}						 
	}
	else if(curBatch == yseg){
		
	}
	else{

	}



	glbStartX = curStartAddress;

	flagRead(curBatch, dev_lock, thread, tile, YoverX, xseg);
	
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
		tileStartOffset = paddsize * segLengthX + paddsize;
		//length Y > length X, diagonal first element starts from Y axis instead of X axis for calculating the address.
		int concurrency;
		for (int i=0; i<highY; i++){
			lvlStartAddress = tileStartOffset + i * segLengthX;
			concurrency = min(tileX, i);
			if(thread <= concurrency){
				idx = lvlStartAddress - segLengthX * thread + thread;
				table[idx] = (table[idx-1] + table[idx-segLengthX] + table[idx]
						+ table[idx+1] + table[idx+segLengthX]) / 5;
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

		moveToGlobalRec(&table[0], dev_table, glbStartX, thread, tileX, tileY, rowsize, segLengthX, segLengthY, paddsize);				
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
		glbStartX += tileX;
		highY -= tileX;
	}
	//update the tile beginning pos for the next tile.
	glbStartX += 1;
	flagWrite(curBatch, dev_lock, thread);
//***********************************************************************************************************************************
	//hyperlane tiles
	tileStartOffset = paddsize * segLengthX + paddsize;
	for (tile = 2; tile < xseg; tile++){
		flagRead(curBatch, dev_lock, thread, tile, YoverX, xseg);
#ifdef DEBUG2
#ifdef ALL		
		if (thread == 0 && curBatch == row){
#endif
#ifndef ALL
		if (thread == 0){
#endif	
			printf("Before computation global memory. curBatch: %d, tile: %d, xseg: %d, glbStartX: %d\n", curBatch, tile, xseg, glbStartX);
			//for(int i=0; i<segLengthY; i++){
				for(int j=0; j<segLengthX; j++){	
					printf("%d ", dev_table[glbStartX + rowsize+j - 1]);
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
//			}
			printf("\n");
		}
		__syncthreads();
#endif	
#ifdef DEBUG2
#ifdef ALL
		if (thread == 0 && curBatch == row && tile <= 3 ){	
#endif
#ifndef ALL
		if (thread == 0){
#endif
			printf("Before move data to share memory. curBatch: %d, tile: %d, xseg: %d, glbStartX: %d\n", curBatch, tile, xseg, glbStartX);
			//for (int i=0; i<segLengthY; i++){
			{	
				for (int j = 0; j<segLengthX; j++){
					printf("%d ", table[segLengthX+j]);
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
		moveToShare(&table[0], dev_table, glbStartX, thread, tileX, tileY, rowsize, segLengthX, segLengthY, warpbatch);
		__syncthreads();
	//	__threadfence_system();
#ifdef DEBUG2
#ifdef ALL
		if (thread == 0 && curBatch == row && tile <= 3 ){	
#endif
#ifndef ALL
		if (thread == 0){
#endif
			printf("Before computation share. curBatch: %d, tile: %d, xseg: %d, glbStartX: %d\n", curBatch, tile, xseg, glbStartX);
			//for (int i=0; i<segLengthY; i++){
			{	
				for (int j = 0; j<segLengthX; j++){
					printf("%d ", table[segLengthX+j]);
				}
				printf("\n");
				for (int j = 0; j<segLengthX; j++){
					printf("%d ", table[(segLengthY-tileX) *segLengthX+j]);
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
		lvlStartAddress = tileStartOffset;
		for (int i=0; i<tileX; i++){
//this is expensive especially when tileX is large. However, if we put if statement outside the loop, we face syncthreads issue.
//New feature warp level syncronize or thread group syncronize may solve the problem.
			if (thread < tileY){
				idx = lvlStartAddress + thread * segLengthX;
				table[idx] = (table[idx-1] + table[idx-segLengthX-1] + table[idx]
						+ table[idx+1] + table[idx+segLengthX+1]) / 5;
				lvlStartAddress += 1;
			}
			__syncthreads();
		}
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
		//need modification, only copy the new updated elements back to the global memory. Also modify moveToGlobalRec
		moveToGlobal(&table[0], dev_table, glbStartX, thread, tileX, tileY, rowsize, paddsize, segLengthX, segLengthY, warpbatch);
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
		flagWrite(curBatch, dev_lock, thread);
	}

//************************************************************************************************************************************
	//the last tile, which is a half of the rectangular
	flagRead(curBatch, dev_lock, thread, xseg, YoverX, xseg);
	glbStartX = curStartAddress + rowsize - paddsize - tileY - paddsize;
	
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
				table[idx] = (table[idx-1] + table[idx-segLengthX] + table[idx]
						+ table[idx+1] + table[idx+segLengthX]) / 5;
			}
			tileStartOffset += segLengthX;
			__syncthreads();
		}
	
		moveToGlobalRec(&table[0], dev_table, glbStartX, thread, tileX, tileY, rowsize, segLengthX, segLengthY, paddsize);				

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
	
	flagWrite(curBatch, dev_lock, thread);
}

void checkGPUError(cudaError err){
	if (cudaSuccess != err){
		printf("CUDA error in file %s, in line %i: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void SOR(int n1, int n2, int padd, int *arr1, int *arr2, int MAXTRIAL){
	cudaSetDevice(0);	
//stride is the longest distance between the element and its dependence along one dimension times
//For example: F(x) = T(x-1) + T(x) + T(x+1), paddsize = 1
	int stride = 1;
	int tileX = 32;
	int tileY = 32;
	int rawElmPerTile = tileX * tileY;
	int tileT = 4;

//PTilesPerTimestamp is the number of parallelgoram tiles can be scheduled at each time stamp
	int PTilesPerTimestamp = (n1/tileX) * (n2/tileY); 
//ZTilesPerTimestamp is the number of trapezoid tiles (overlaped tiles) needed to calculate the uncovered area at each time stamp.
	int ZTilesPerTimestamp = (n1/tileX) + (n2/tileY) - 1; 
	int rowsize = 2 * padd + n1; 
	int colsize = 2 * padd + n2;

	int *dev_arr1, *dev_arr2;
	volatile int *dev_lock;
	int *lock;
	size_t freeMem, totalMem;
	
	
	cudaMemGetInfo(&freeMem, &totalMem);
	int tablesize = colsize * rowsize;
	cout << "current GPU memory info FREE: " << freeMem << " Bytes, Total: " << totalMem << " Bytes.";
	cout << "colsize: " << colsize << ", rowsize: " << rowsize << ", allocates: " << tablesize * sizeof(int)<< " Bytes." << endl;
	cudaError err = cudaMalloc(&dev_arr1, tablesize * sizeof(int));
	checkGPUError(err);
	err = cudaMalloc(&dev_arr2, tablesize * sizeof(int));
	checkGPUError(err);
	
	cudaMemcpy((void*)dev_arr1, arr1, tablesize*sizeof(int), cudaMemcpyHostToDevice);

	int threadPerBlock = min(1024, rawElmPerTile);
//	int blockPerGrid = PTilesPerTimestamp;
	int blockPerGrid = 1;
	int numStream = 28;
	int warpbatch = threadPerBlock / 32;

//memory structure: stream --> tile --> time --> dependence --> tileX
	int *dev_inter_stream_dependence;
	int stream_dep_offset = tileT * ((stride+1)*tileX) * (n1+stride);
	int inter_stream_dependence = numStream * stream_dep_offset;
	err = cudaMalloc(&dev_inter_stream_dependence, inter_stream_dependence):
	checkGPUError(err);

	int xseg = n1 / tileX;
	int yseg = n2 / tileY;
	int stream_offset = n2 % numStream;
	
	lock = new int[yseg+1];
	lock[0] = xseg+1;
	for (int i=1; i<yseg+1; i++)
		lock[i] = 0;
	cudaMalloc(&dev_lock, (yseg+1) * sizeof(int));	
	cudaMemcpy((void*)dev_lock, lock, (yseg+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaStream_t stream[numStream];
	for (int s=0; s<numStream; s++)
		cudaStreamCreate(&stream[s]);
	
	for(int t = 0; t <= MAXTRIAL; t+= tileT){
//GPU_ZTile() is the kernel function to calculate the update result, unconvered by Parallelgoram tiling.
//These data are calculated with trapezoid tiling, thus they can be launched concurrently.
// ZTile and cudaDeviceSynchronize() will stop theparallelism along the temporal dimension and force
//the beginning of the new t tiles has to wait the completion of the previous t tiles.
//		GPU_ZTile<<<>>>();
//		cudaDeviceSynchronize();		
		for(int curBatch = 0; curBatch <= yseg; curBatch++){
//Have to change the stream Index so that the stream for next time tile can start without waiting for the 
//completion of the previous time tile. 
//Example: stream 0, 1, 2 are scheduled to the last three batches in one time tile, since the execution on
//the next time tile also starts from stream 0, this new execution in stream 0 has to wait for the previous
			int logicSMStream = curBatch % numStream;
			int curSMStream = (logicSMStream +  stream_offset * t) % numStream;
			int curStartAddress = curBatch * tileY * rowsize;
			int rowStartOffset = padd * rowsize + padd;
			int preCurSMStream = (curSMStream - 1) % numStream;
//			cout << endl << "curBatch: " << curBatch << ", yseg: " << yseg << endl;	
			GPU_Tile<<<blockPerGrid, threadPerBlock, 0, stream[curSMStream]>>>(dev_arr1, dev_arr2, dev_lock, curBatch, curStartAddress, tileX, tileY,  padd, stride, rowStartOffset, rowsize, colsize, xseg, yseg, n1, n2, warpbatch, curSMStream, preCurSMStream, dev_inter_stream_dependence, tileT);	
//			GPU<<<blockPerGrid, threadPerBlock>>>(dev_table, dev_arr1, dev_arr2, dev_lock, curBatch, curStartAddress, rowtiles, resX, tileX, tileY,  paddX, paddY, rowStartOffset, rowsize, colsize, xseg, yseg, tileY/tileX, n1, n2);			
			checkGPUError( cudaGetLastError() );
//		cudaDeviceSynchronize();
		}
		cudaDeviceSynchronize();
	}	
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
	
	cudaFree((void*)dev_table);
	cudaFree((void*)dev_lock);
	delete[] lock;

}

