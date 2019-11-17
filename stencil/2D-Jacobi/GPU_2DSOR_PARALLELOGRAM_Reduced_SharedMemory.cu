#include<iostream>
#include<cstdlib>
#include<fstream>
#include<string>
#include<sys/time.h>

//#define ALL
//#define DEBUG
//#define PRINT_FIRST_BATCH
//#define PRINT_MID_BATCH
//#define PRINT_LAST_BATCH
#define PRINT_FINAL_RESULT
#define RTX_2080
#define FIRST_BATCH
#define LAST_BATCH
#define MID_BATCH
#define TIME_LOCK
#define SYNC

using namespace std;
__device__ int row = 0;

__device__ void _jacobi_square(volatile int* tile1, volatile int* tile2, int newtilePos, int tilePos, int* stride, int* segLengthX){
	int total = 0;
	for (int row = -stride[0]; row <= stride[0]; row++){
		for (int col = -stride[0]; col <= stride[0]; col++){
			total += tile1[tilePos + row * segLengthX[0] + col];
		}
	}
	tile2[newtilePos] = total / (stride[0] + stride[0] + 1) / (stride[0] + stride[0] + 1);
}

__device__ void _jacobi_cross(volatile int* tile1, volatile int* tile2, int newtilePos, int tilePos, int* stride, int* segLengthX){
	int total = 0;
	for (int row = -stride[0]; row < 0; row++){
		total += tile1[tilePos + row * segLengthX[0]];
	}
	for (int row = 1; row <= stride[0]; row++){
		total += tile1[tilePos + row * segLengthX[0]];
	}
	
	for (int col = -stride[0]; col <= stride[0]; col++){
		total += tile1[tilePos + col];
	}
	
	tile2[newtilePos] = total / ((stride[0] + stride[0] + 1) * 2 - 1);
}


__device__ void stencil(volatile int* tile1, volatile int* tile2, int newtilePos, int tilePos, int* stride, int* segLengthX){
	_jacobi_square(tile1, tile2, newtilePos, tilePos, stride, segLengthX);
//	_jacobi_cross(tile1, tile2, newtilePos, tilePos, stride, segLengthX);
}



__device__ void swapTile(volatile int* tile1, volatile int* tile2, int* tileX, int* tileY, int* segLengthX, int* dep_stride, int* warpbatch){
	int idx = threadIdx.x % 32;
	int warpidx = threadIdx.x / 32;
	if (warpidx < tileY[0]){
		int pos1 = (dep_stride[0] + warpidx) * segLengthX[0] + dep_stride[0];
		int pos2 = warpidx * tileX[0];
		for(; warpidx < tileY[0]; warpidx += warpbatch[0]){
			for(int i = idx; i < tileX[0]; i += 32){
				tile1[pos1 + i] = tile2[pos2 + i];
				tile2[pos2 + i] = 0;
			}
			pos1 += (warpbatch[0] * segLengthX[0]);
			pos2 += (warpbatch[0] * tileX[0]);
		}
	}
}
/*
__device__ void swapTile(volatile int* tile1, volatile int* tile2, int* segLengthX, int* segLengthY, int threadsPerBlock){
	int len = segLengthX[0] * segLengthY[0];
	for (int idx = threadIdx.x; idx < len; idx += threadsPerBlock){
		tile1[idx] = tile2[idx];
		tile2[idx] = 0;
	}
}
*/		
__device__ void printGlobal(volatile int* dev_arr, int* width, int* height, int curSMStream){
	if (threadIdx.x == 0 ){
		for (int r = 0; r < height[0]; r++){
			for (int c = 0; c < width[0]; c++){
				int pos = r * width[0] + c;
				printf("%d ", dev_arr[pos]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

__device__ void printSharedTile(volatile int* tile, int* segLengthX, int* tileY, int* dep_stride, int curSMStream){
	if (threadIdx.x == 0 ){
		for (int row = 0; row < dep_stride[0] + tileY[0]; row++){
			for (int col = 0; col < segLengthX[0]; col++){
				printf("%d ", tile[row * segLengthX[0] + col]);
			}
			printf("\n");
		}
	}
}

__device__ void moveMatrixToTile(volatile int* dev_arr, volatile int* tile, int* segLengthX, int tileX, int* tileY, int* dep_stride, int tileAddress, int* width, int* warpbatch){
	int idx = threadIdx.x % 32;
	int warpidx = threadIdx.x / 32;
	if (warpidx < tileY[0]){
		int glbpos = tileAddress + warpidx * width[0];
		int shrpos = dep_stride[0] * segLengthX[0] + warpidx * segLengthX[0] + dep_stride[0];
		for (; warpidx < tileY[0]; warpidx += warpbatch[0]){
			for (int i = idx; i < tileX; i += 32){
				tile[shrpos+i] = dev_arr[glbpos+i];
			}
			shrpos += (warpbatch[0] * segLengthX[0]);
			glbpos += (warpbatch[0] * width[0]);
		}
	}
}

//intra_dep array structure: tileT * dep_stride * tileY
__device__ void moveIntraDepToTile(int* intra_dep, volatile int* tile, int tt, int* tileY, int* segLengthX, int* dep_stride, int len){
	//at each tt, (stride+1) dependent data are required at x axis.
	//only the threads, which are within tileY are working here.
	//threadPerBlock has to be no less than tileY * dep_stride
	if (threadIdx.x < len * dep_stride[0]){
		int pos = tt * dep_stride[0] * tileY[0] + threadIdx.x;
		int tilepos = dep_stride[0] * segLengthX[0] + threadIdx.x/dep_stride[0] * segLengthX[0] + threadIdx.x % dep_stride[0];
		tile[tilepos] = intra_dep[pos];
	}
}

__device__ void moveIntraDepToTileEdge(volatile int* dev_arr, volatile int* tile, int* width, int* segLengthX, int* dep_stride, int tt, int* n1, int len, int* stride, int offset = 0){
	//copy out-of-range data to tile
	if (threadIdx.x < len * dep_stride[0]){
		int glbpos = -dep_stride[0] + offset * (n1[0] + dep_stride[0]) + threadIdx.x/dep_stride[0] * width[0] + threadIdx.x % dep_stride[0];
		int tilepos = threadIdx.x/dep_stride[0] * segLengthX[0] + threadIdx.x % dep_stride[0] + offset * (dep_stride[0] + tt * stride[0]);
		tile[tilepos] = dev_arr[glbpos];
	}
}

__device__ void moveTileToIntraDep(int* intra_dep, volatile int* tile, int tt, int* tileX, int* tileY, int* segLengthX, int* dep_stride, int* stride, int isRegular, int len){
	if (threadIdx.x < len * dep_stride[0]){
		int pos = tt * dep_stride[0] * tileY[0] + threadIdx.x;
		int tilepos = dep_stride[0] * segLengthX[0] + tileX[0] - tt * isRegular * stride[0];
	       	tilepos	+= threadIdx.x/dep_stride[0] * segLengthX[0] + threadIdx.x % dep_stride[0];
		intra_dep[pos] = tile[tilepos];
	}
}

__device__ void printIntraDep(int* intra_dep, int tt, int* tileY, int* segLengthX, int* dep_stride, int isRegular, int len){
	if (threadIdx.x == 0){
		printf("intra_dep: \n");
		for (int i = 0; i < len * dep_stride[0]; i++){
			int pos = tt * dep_stride[0] * tileY[0] + i;
			printf("%d ", intra_dep[pos]);
//			int tilepos = dep_stride * segLengthX + tileX - tt * isRegular;
//			tilepos += i / dep_stride * segLengthX + i % dep_stride;
//			printf("%d ", tile[tilepos]);
//			printf("thread: %d, pos: %d, tilepos: %d, intra: %d, tile: %d\n", i, pos, tilepos, intra_dep[pos], tile[tilepos]);
		}
		printf("\n");
	}
}

//inter_stream_dep array structure: stream * tileT * dep_stride * (n1 + dep_stride)
__device__ void moveInterDepToTile(volatile int* inter_stream_dep, volatile int* tile, int tt, int* tileX, int* dep_stride, int stream, int tileT, int* n1, int* segLengthX, int tileIdx, int len){
	int startAddress = (stream * tileT + tt) * dep_stride[0] * (n1[0] + dep_stride[0]);
	if (tileIdx > 0){       
//		startAddress = ( (tileIdx-1) * tileX + tileX-tt );
		startAddress -= tt * (dep_stride[0] / 2);
	}
	startAddress += ( tileIdx * tileX[0]);
	//variable len specifies the eligible elements should be moved. This is caused by the irregular tile.
	if (threadIdx.x < len + dep_stride[0]){
		int pos = startAddress + threadIdx.x;
		int tilepos = threadIdx.x;
		for (int i=0; i<dep_stride[0]; i++){
	 		tile[tilepos] = inter_stream_dep[pos];
			pos += (n1[0] + dep_stride[0]);
			tilepos += segLengthX[0];
		}
	}	
}

__device__ void moveInterDepToTileEdge(volatile int* dev_arr, volatile int* tile, int* tileX, int* dep_stride, int* stride, int* n2, int* segLengthX, int* width, int tileIdx, int tt, int len, int offset){
//	int glbpos = (padd - dep_stride) * width + offset * (dep_stride + n2) * width + padd - dep_stride + threadIdx.x;
	//"-dep_stride * width" is added to locate the address of the row that out-of-range inter-dep entries locate at;
	//"tileIdx * tileX" is added to locate the address of the first element of the out-of-range inter-dep entries;
	int glbpos = offset * dep_stride[0] * width[0] - dep_stride[0] * width[0] + tileIdx * tileX[0] - dep_stride[0] + threadIdx.x;
	if (tileIdx > 0){
//		glbpos += ((tileIdx-1) * tileX + tileX-tt);
		glbpos -= tt * stride[0];
	}
	if (threadIdx.x < len + dep_stride[0]){
		int tilepos = offset * (dep_stride[0] + tt * stride[0]) * segLengthX[0] + threadIdx.x;
		for (int i=0; i<dep_stride[0]; i++){
			tile[tilepos] = dev_arr[glbpos];
			tilepos += segLengthX[0];
			glbpos += width[0];
		}
	}
}

__device__ void moveTileToInterDep(volatile int* inter_stream_dep, volatile int* tile, int tt, int* tileX, int* tileY, int* dep_stride, int* stride, int nextSMStream, int tileT, int* n1, int* segLengthX, int tileIdx, int len, int isRegular){
	int startAddress = dep_stride[0] + (nextSMStream * tileT + tt) * dep_stride[0] * (n1[0] + dep_stride[0]);
	//for the edge tiles, the size is irregular so that the start position of some tt timestamp are not times of tileX.
	if (tileIdx > 0)       
		startAddress += ( (tileIdx - 1) * tileX[0] + tileX[0] - tt * stride[0]);
	//variable len specifies the eligible elements should be moved. This is caused by the irregular tile.
	if (threadIdx.x < len){
		int pos = startAddress + threadIdx.x;
		//tt * isRegular ? (tt + 1) * isRegular
		int tilepos = dep_stride[0] + (tileY[0] - tt * stride[0] * isRegular) * segLengthX[0] + threadIdx.x;
		for (int i=0; i<dep_stride[0]; i++){
	 		inter_stream_dep[pos] = tile[tilepos];
			pos += (n1[0] + dep_stride[0]);
			tilepos += segLengthX[0];
		}
	}
}

__device__ void printInterDep(volatile int* inter_stream_dep, int tt, int* tileX, int* dep_stride, int nextSMStream, int tileT, int* n1, int tileIdx, int len, int isRegular){
	if (threadIdx.x == 0){
		int startAddress = dep_stride[0] + (nextSMStream * tileT + tt) * dep_stride[0] * (n1[0] + dep_stride[0]);
		//for the edge tiles, the size is irregular so that the start position of some tt timestamp are not times of tileX.
		if (tileIdx > 0)       
			startAddress += ( (tileIdx - 1) * tileX[0] + tileX[0] - tt * (dep_stride[0] / 2));
		printf("nextSMStream: %d, tileIdx: %d, tt: %d, startAddress: %d, isRegular: %d\n", nextSMStream, tileIdx, tt, startAddress, isRegular);
		printf("inter_stream_dep: ");
		for (int i = 0; i < len; i++){
			int pos = startAddress + i;
			for (int j=0; j<dep_stride[0]; j++){
		 		printf("%d ",inter_stream_dep[pos]);
				pos += (n1[0] + dep_stride[0]);
			}			
		}
		printf("\n");
	}
}

__device__ void moveTileToInterDepEdge(volatile int* tile, volatile int* inter_stream_dep, int tt, int* tileY, int tileT, int nextSMStream, int* dep_stride, int* stride, int* segLengthX, int* n1, int isRegular){
	int startAddress = (nextSMStream * tileT + tt) * dep_stride[0] * (n1[0] + dep_stride[0]);
//	int glbpos = padd * width + curBatch * tileY * width + (padd - dep_stride) + (tileY - dep_stride) * width;
	//(tileY - tt) because the tile is shifted up and left in each timing iteration. Do not need to consider left shift here because the entries, shifted outside of the left boundary, are discarded and the out-of-range entries are constant.
	int shrpos = (tileY[0] - tt * stride[0] * isRegular) * segLengthX[0];  
	if (threadIdx.x < dep_stride[0]){
		int interpos = startAddress + threadIdx.x;
		int pos = shrpos + threadIdx.x;
		for (int i=0; i<dep_stride[0]; i++){
	 		inter_stream_dep[interpos] = tile[pos];
			pos += segLengthX[0];
			interpos += (n1[0] + dep_stride[0]);
		}
	}
}

__device__ void printInterDepEdge(volatile int* inter_stream_dep, int tt, int tileT, int nextSMStream, int* dep_stride, int* segLengthX, int* n1){
	if (threadIdx.x == 0){
		int startAddress = (nextSMStream * tileT + tt) * dep_stride[0] * (n1[0] + dep_stride[0]);
		printf("nextSMStream: %d, tt: %d, startAddress: %d, segLengthX: %d\n", nextSMStream, tt, startAddress, segLengthX[0]);
		printf("inter_stream_dep_edge: ");
		for (int i = 0; i < dep_stride[0]; i++){
			int interpos = startAddress + i;
			for (int j = 0; j < dep_stride[0]; j++){
				printf("%d ", inter_stream_dep[interpos]);
				interpos += (n1[0] + dep_stride[0]);
			}		
		}
		printf("\n");
	}
}

__device__ void moveShareToGlobalEdge(volatile int* tile, volatile int* dev_arr, int startPos, int ignLenX, int ignLenY, int* tileX, int* tileY, int* dep_stride, int* width, int* segLengthX){
	int row, col, glbPos, tilePos;
	for (int tid = threadIdx.x; tid < tileX[0] * tileY[0]; tid += blockDim.x){
		col = tid % tileX[0];
		row = tid / tileX[0];
		if (col < tileX[0] - ignLenX && row < tileY[0] - ignLenY){
			glbPos = startPos + row * width[0] + col;
			tilePos = (dep_stride[0] + row) * segLengthX[0] + dep_stride[0] + col;
			dev_arr[glbPos] = tile[tilePos];
		}
	}	
}	

__device__ void moveShareToGlobal(volatile int* tile, volatile int* dev_arr, int startPos, int* tileX, int* tileY, int* dep_stride, int* width, int* segLengthX){
	int xidx, yidx, glbPos, tilePos;
	for (int tid = threadIdx.x; tid < tileX[0] * tileY[0]; tid += blockDim.x){
		xidx = tid % tileX[0];
		yidx = tid / tileX[0];
		glbPos = startPos + yidx * width[0] + xidx;
		tilePos = (dep_stride[0] + yidx) * segLengthX[0] + dep_stride[0] + xidx;
		dev_arr[glbPos] = tile[tilePos];
	}	
}	
	

/*
//need a global array which has size of the number of batches in each t. 
//Each stream check the corresponding element in this array to see if it is true; it is true only when the batch beneath it and in the 
//previous t is already completed.
//If it is true, change it to false and start the computation. At the end, change it back to true when computation is finished.
__device__ void read_batch_lock_for_time(int* dev_time_lock, int curBatch){
	if (threadIdx.x == 0){
		while(dev_time_lock[curBatch] != 1){
		}
		dev_time_lock[curBatch] = 0;
	}
	__syncthreads();
}

__device__ void write_batch_lock_for_time(int* dev_time_lock, int curBatch){
	if (threadIdx.x == 0){
		dev_time_lock[curBatch] = 1;
	}
	__synchthreads();
}
*/

//Similar to the lock array in nested loop study; create a 1-d array for the size of number of total rows. 
//A counter value is used for each row.
//Besides, we need to create such an array for each time stamp.
__device__ void read_tile_lock_for_batch(volatile int* dev_row_lock, int curBatch, int tileIdx, int* YoverX, int* xseg, int* yseg, int timepiece){
	if (threadIdx.x == 0){
		int limit = min(tileIdx + YoverX[0], xseg[0]);
		while(dev_row_lock[timepiece * yseg[0] + curBatch] < limit){
		}
#ifdef DEBUG_LOCK
		printf("curBatch: %d, tileIdx: %d, timepiece: %d, value: %d, limit: %d\n", curBatch, tileIdx, timepiece, dev_row_lock[timepiece * yseg[0] + curBatch], limit);
#endif
	}
	__threadfence();
	__syncthreads();
}

__device__ void write_tile_lock_for_batch(volatile int* dev_row_lock, int curBatch, int* yseg, int timepiece){
	if (threadIdx.x == 0){
		dev_row_lock[timepiece * yseg[0] + curBatch + 1] += 1;
#ifdef DEBUG_LOCK
		printf("curBatch: %d, timepiece: %d, update to lock at: %d, value: %d\n", curBatch, timepiece, timepiece * yseg[0] + curBatch + 1, dev_row_lock[timepiece * yseg[0] + curBatch + 1]);
#endif
	}
	__threadfence();
	__syncthreads();
}

//dev_time_lock has "numStream" lock elements which are mapped to the "numStream" cuda streams. It is used to check if it is safe for one stream to perform "write" operation to the "inter-dep array" of the next stream.
__device__ void read_time_lock_for_stream(volatile int* dev_time_lock, int curSMStream, int nextSMStream, int* xseg, int curBatch){
	if (threadIdx.x == 0){
#ifdef DEBUG_LOCK
		printf("curBatch: %d, curSMStream: %d, xseg: %d, nextSMStream: %d, lock val: %d\n", curBatch, curSMStream, xseg[0], nextSMStream, dev_time_lock[nextSMStream]);
#endif
		while(dev_time_lock[nextSMStream] < xseg[0]){
#ifdef DEBUG_LOCK
			printf("curBatch: %d, curSMStream: %d, nextSMStream: %d, lock val: %d\n", curBatch, curSMStream, nextSMStream, dev_time_lock[nextSMStream]);
#endif
		}
	}
	__threadfence();
	__syncthreads();
}

__device__ void write_time_lock_for_stream(volatile int* dev_time_lock, int curSMStream, int* xseg, int curBatch){
	if (threadIdx.x == 0){
		dev_time_lock[curSMStream] = xseg[0];
//		atomicCAS(dev_time_lock + curSMStream, 0, xseg);
	}
	__threadfence();
	__syncthreads();
#ifdef DEBUG_LOCK
	if (threadIdx.x == 0){
		printf("curBatch: %d, curSMStream: %d, update lock val batck to: %d\n", curBatch, curSMStream, dev_time_lock[curSMStream]);
	}
	__syncthreads();
#endif
}

__device__ void clear_time_lock_for_stream(volatile int* dev_time_lock, int curSMStream, int curBatch){
	if (threadIdx.x == 0){
#ifdef DEBUG_LOCK
		printf("curBatch: %d, curSMStream: %d, val: %d, to be cleared.\n", curBatch, curSMStream, dev_time_lock[curSMStream]);
#endif
		dev_time_lock[curSMStream] = 0;
	}
	__threadfence();
	__syncthreads();
}

//__global__ void GPU_Tile(volatile int* dev_arr, int curBatch, int tx, int tileY, int padd, int stride, int height, int width, int xseg, int yseg, int n1, int n2, int warpbatch, int curSMStream, int nextSMStream, volatile int* inter_stream_dep, int inter_stream_dep_size, int tileT, int timepiece, int batchStartAddress, volatile int* dev_row_lock, volatile int* dev_time_lock){ 
__global__ void GPU_Tile(volatile int* dev_arr, const int curBatch, int* dev_var, const int curSMStream, const int nextSMStream, volatile int* inter_stream_dep, const int inter_stream_dep_size, const int tileT, const int timepiece, int batchStartAddress, volatile int* dev_row_lock, volatile int* dev_time_lock, const int threadsPerBlock){ 
//We assume row size n1 is the multiple of 32 and can be completely divided by tileX.
//For each row, the first tile and the last tile are computed separately from the other tiles.
//size of the shared memory is determined by the GPU architecture.
//tileX is multiple times of 32 to maximize the cache read.		
#ifdef DEBUG
	if (threadIdx.x == 0){
		printf("This is curBatch: %d, timepiece: %d, curStream: %d, nextSMStream: %d\n", curBatch, timepiece, curSMStream, nextSMStream);
	}
	__syncthreads();
#endif
	//need two arrays: 1. tile raw data; 2. intra-stream dependence
	//intra_dep size is restricted by the "dep_stride", "tileY", and "tileT"

#ifndef RTX_2080
//*************** GTX 1080 Ti *********************
	//tileX: 64; tileY: 64, stride: 1-3
//	volatile __shared__ int tile1[4900];
//	volatile __shared__ int tile2[4900];
//	__shared__ int intra_dep[2470];
	
	//tileX: 32; tileY: 32, stride: 1-10
	volatile __shared__ int tile1[2704];
	volatile __shared__ int tile2[2704];
	__shared__ int intra_dep[6800];

	//tileX: 32; tileY: 64, stride: 1-8	
//	volatile __shared__ int tile1[3840];
//	volatile __shared__ int tile2[3840];
//	__shared__ int intra_dep[4352];
	
#else
//*************** RTX 2080 Ti *********************
	//64KB per block
	//tileX: 64; tileY: 64, stride: 1-3
//	volatile __shared__ int tile1[4900];
//	volatile __shared__ int tile2[4900];
//	__shared__ int intra_dep[2470];
	
	//tileX: 32; tileY: 32, stride: 1-10
//	volatile __shared__ int tile1[2704];
//	volatile __shared__ int tile2[2704];
//	__shared__ int intra_dep[6800];

	//tileX: 32; tileY: 64, stride: 1-8	
	volatile __shared__ int tile1[3840];
	volatile __shared__ int tile2[2048];
	__shared__ int intra_dep[6380];

	//32KB per block to fully utilize 64KB
	//tileX: 32; tileY: 32, stride: 1-8
///	volatile __shared__ int tile1[2704];
//	volatile __shared__ int tile2[2704];
//	__shared__ int intra_dep[2760];

	//tileX: 32; tileY: 64, stride: 1-4	
//	volatile __shared__ int tile1[2880];
//	volatile __shared__ int tile2[2880];
//	__shared__ int intra_dep[3310];
#endif	
	
	__shared__ int YoverX[1];
	__shared__ int dep_stride[1];
	__shared__ int segLengthX[1];
	__shared__ int segLengthY[1];
	__shared__ int tileX[1], tileY[1], stride[1], height[1], width[1];
	__shared__ int xseg[1], yseg[1], n1[1], n2[1], warpbatch[1];
	if (threadIdx.x == 0){
		tileX[0] = dev_var[0]; tileY[0] = dev_var[1]; stride[0] = dev_var[3];
		height[0] = dev_var[4]; width[0] = dev_var[5]; xseg[0] = dev_var[6]; yseg[0] = dev_var[7];
		n1[0] = dev_var[8]; n2[0] = dev_var[9]; warpbatch[0] = dev_var[10];
		YoverX[0] = tileY[0]/tileX[0];
		dep_stride[0] = stride[0] + stride[0];
		segLengthX[0] = tileX[0] + dep_stride[0];
		segLengthY[0] = tileY[0] + dep_stride[0];
	}
	__threadfence();
	__syncthreads();
	//version 1.0: dep_stride == stride + 1 is not true when stride is larger than 1.
//	int dep_stride = stride + 1;
//	int segLengthX = tileX + dep_stride[0];
//	int segLengthY = tileY + dep_stride[0];
	int tileIdx = 0;
	int xidx, yidx;
	const int xidx2 = threadIdx.x % tileX[0];
        const int yidx2 = threadIdx.x / tileX[0];
	int tilePos, newtilePos, glbPos;
	int tileAddress;
//	int YoverX = tileY/tileX;	
//if this is the first batch of the current t tile, have to copy the related dependence data from global tile array into global inter-stream-dependence array.
//Challenges: when stream 0 is still working on one of the current t tiles but stream 2 already starts processing the first batch of the next t tiles. Copying the dependence data to arr[stream[0]] does not work.
//for the first and last batches, we need charactorized function to take care of the edge elements.

//***********************************************************************************************************************************
//	read_batch_lock_for_time(timepiece, curBatch);
//processing the first tile of each row, use the near-edge elements for the out-of-range dependence.
	//wait until it is safe to launch and execute the new batch.

	if (curBatch == 0){
		//for the first batch, use the near-edge elements for the out-of-range dependence.
		//when tile = 0, the calculated data which are outside the range are not copied to tile2, tile size is shrinking 
		//along T dimension. Out-of-range elements are used for dependent data.
		tileAddress = batchStartAddress + tileIdx * tileX[0];
		read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
#ifdef TIME_LOCK
		//Before starting the process on one stream, we must first ensure that it is safe for this stream writing data to inter-dep array of the next stream.
		read_time_lock_for_stream(dev_time_lock, curSMStream, nextSMStream, xseg, curBatch);
		//if it is safe to start the work now, clear the time_lock for the current stream to 0.
		clear_time_lock_for_stream(dev_time_lock, curSMStream, curBatch);
#endif
#ifdef FIRST_BATCH
		moveMatrixToTile(dev_arr, &tile1[0], segLengthX, tileX[0], tileY, dep_stride, tileAddress, width, warpbatch);
#ifdef SYNC
		__threadfence();
		__syncthreads();
#endif
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("This is curBatch: %d, timepiece: %d, curStream: %d, nextSMStream: %d, dep_stride: %d\n", curBatch, timepiece, curSMStream, nextSMStream, dep_stride[0]);
	printf("move data matrix to tile %d: \n", tileIdx);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif
		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTileEdge(&dev_arr[batchStartAddress], &tile1[dep_stride[0] * segLengthX[0]], width, segLengthX, dep_stride, tt, n1, tileY[0], stride, 0);
			//parameter offset == 0
			moveInterDepToTileEdge(&dev_arr[batchStartAddress], &tile1[0], tileX, dep_stride, stride, n2, segLengthX, width, tileIdx, tt, tileX[0], 0);
			__threadfence();
			__syncthreads();
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("curSMStream: %d, curBatch: %d, tileId: %d, timepiece: %d\n", curSMStream, curBatch, tileIdx, timepiece);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif	
			for (int tid = threadIdx.x; tid < tileX[0] * tileY[0]; tid += blockDim.x){
				//out-of-range results should be ignored
				//because of the bias, xidx and yidx are the pos of new time elements.
				//thread % tileX and thread / tileX are pos of current cached elements.
				xidx = tid % tileX[0];
				yidx = tid / tileX[0];
			        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by "dep_stride / 2".
				tilePos = stride[0] * segLengthX[0] + stride[0] + yidx * segLengthX[0] + xidx;	
				//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array.
				newtilePos = yidx2 * tileX[0] + xidx2;
				//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
				//Because the edge elements use only the out-of-range elements as dependent data, we need specific manipulation.
				if (xidx > 0 && xidx < tileX[0] - tt * stride[0] && yidx > 0 && yidx < tileY[0] - tt * stride[0]){
					stencil(tile1, tile2, newtilePos, tilePos, stride, segLengthX);
				}
			}
			__threadfence();
			__syncthreads();
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("tile2\n");
	printSharedTile(&tile2[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif				
			//Since the tile size is reduced along the calculation, the intraDep elements (in last two column of the valid tile) is also shifted to left.
			//Set variable isRegular == 1, when there is a size reduction. 
			moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, stride, 1, tileY[0]);
#ifdef SYNC
			__threadfence();
			__syncthreads();
#endif

#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("intra_dep: \n");
	printIntraDep(&intra_dep[0], tt, tileY, segLengthX, dep_stride, 1, tileY[0]);
}
__threadfence();
__syncthreads();
#endif
			//first tile has to copy the out-of-range elements, which are on the left-hand side, to next stream's inter_stream_dep array
			moveTileToInterDepEdge(tile1, inter_stream_dep, tt, tileY, tileT, nextSMStream, dep_stride, stride, segLengthX, n1, 1);
#ifdef SYNC
			__threadfence();
			__syncthreads();
#endif
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("inter_dep_edge: \n");
	printInterDepEdge(inter_stream_dep, tt, tileT, nextSMStream, dep_stride, segLengthX, n1);
}
__threadfence();
__syncthreads();
#endif			
			//variable isRegular == 1, because one row is shifted out-side-of the upper boundary
			//variable len == tileX-tt because this tile is not in a regular size.
			moveTileToInterDep(&inter_stream_dep[0], tile1, tt, tileX, tileY, dep_stride, stride, nextSMStream, tileT, n1, segLengthX, tileIdx, tileX[0] - tt * stride[0], 1);
#ifdef SYNC
			__threadfence();
			__syncthreads();
#endif
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("inter_dep: \n");
	printInterDep(&inter_stream_dep[0], tt, tileX, dep_stride, nextSMStream, tileT, n1, tileIdx, tileX[0] - tt * stride[0], 1);
}
__threadfence();
__syncthreads();
#endif
			//swap tile2 with tile1;
			swapTile(&tile1[0], &tile2[0], tileX, tileY, segLengthX, dep_stride, warpbatch);
//			swapTile(tile1, tile2, segLengthX, segLengthY, threadsPerBlock);
			__threadfence();
			__syncthreads();
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("tile1 after calculation: \n");
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif	
		}
		//glbPos is the index where the calculated elements should be stored at in the global matrix array.
		//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
		//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
		//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
		//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
		//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
		//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
		//ignLenX == tileT because tileX-tileT elements are copied at each row, ignLenY == tileT because tileY-tileT elements are copied at each column.
		glbPos = tileAddress;	
		moveShareToGlobalEdge(tile1, dev_arr, glbPos, tileT * stride[0], tileT * stride[0], tileX, tileY, dep_stride, width, segLengthX);	
#ifdef SYNC
		__threadfence();
		__syncthreads();
#endif
#ifdef PRINT_FIRST_BATCH
		printGlobal(dev_arr, width, height, curSMStream);
		__threadfence();
		__syncthreads();
#endif
//endif FIRST_BATCH
#endif
		write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);

		//tile = 1 to xseg-1; regular size tiles, with index shifting.
		for (tileIdx = 1; tileIdx < xseg[0]-1; tileIdx++){
			tileAddress = batchStartAddress + tileIdx * tileX[0];
			read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
			//copy the base spatial data to shared memory for t=0.
#ifdef FIRST_BATCH
			moveMatrixToTile(dev_arr, tile1, segLengthX, tileX[0], tileY, dep_stride, tileAddress, width, warpbatch);
#ifdef SYNC
			__threadfence();
			__syncthreads();
#endif
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("move data matrix to tile %d: \n", tileIdx);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif
			for (int tt=0; tt<tileT; tt++){
				moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, tileY[0]);
				moveInterDepToTileEdge(&dev_arr[batchStartAddress], &tile1[0], tileX, dep_stride, stride, n2, segLengthX, width, tileIdx, tt, tileX[0], 0);
				__threadfence();
				__syncthreads();
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("curSMStream: %d, curBatch: %d, tileId: %d, timepiece: %d\n", curSMStream, curBatch, tileIdx, timepiece);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif

				for (int tid = threadIdx.x; tid < tileX[0] * tileY[0]; tid += blockDim.x){
					//out-of-range results should be ignored
					//because of the bias, xidx and yidx are the pos of new time elements.
					//thread % tileX and thread / tileX are pos of current cached elements.
					xidx = tid % tileX[0];
					yidx = tid / tileX[0];
				        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by "stride"
					tilePos = stride[0] * segLengthX[0] + stride[0] + yidx * segLengthX[0] + xidx;	
					//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array
					newtilePos = yidx2 * tileX[0] + xidx2;
					//newtilePos = tilePos;
					//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
					//Because, the edge elements use only the out-of-range elements as dependent data, we need specific manipulation.
					if (yidx > 0 && yidx < tileY[0] - tt * stride[0]){
						stencil(tile1, tile2, newtilePos, tilePos, stride, segLengthX);
					}
				}	
				__threadfence();
				__syncthreads();
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("tile2\n");
	printSharedTile(&tile2[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif	
				//Set variable isRegular == 0 to disable the tile size reduction, when tile size are constant during the calculation. 
				moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, stride, 0, tileY[0]);
#ifdef SYNC
				__threadfence();
				__syncthreads();
#endif
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("intra_dep: \n");
	printIntraDep(&intra_dep[0], tt, tileY, segLengthX, dep_stride, 1, tileY[0]);
}
__threadfence();
__syncthreads();
#endif
				//variable isRegular == 1 because one row is shifted out-side-of the upper boundary.
				//variable len == tileX-tt because row is shifted out-side-of the upper boundary
				moveTileToInterDep(&inter_stream_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, stride, nextSMStream, tileT, n1, segLengthX, tileIdx, tileX[0], 1);
#ifdef SYNC
				__threadfence();
				__syncthreads();
#endif
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("inter_dep: \n");
	printInterDep(&inter_stream_dep[0], tt, tileX, dep_stride, nextSMStream, tileT, n1, tileIdx, tileX[0], 1);
}
__threadfence();
__syncthreads();
#endif
				//swap tile2 with tile1;
				swapTile(&tile1[0], &tile2[0], tileX, tileY, segLengthX, dep_stride, warpbatch);
//				swapTile(&tile1[0], &tile2[0], segLengthX, segLengthY, threadsPerBlock);
				__threadfence();
				__syncthreads();
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("tile1 after calculation: \n");
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif	
			}						 

			//glbPos is the index where the calculated elements should be stored at in the global matrix array.
			//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
			//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
			//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
			//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
			//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
			//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
			//ignLenX == 0 because all elements are copied at each row, ignLenY == tileT because tileY-tileT elements are copied at each column.
			glbPos = tileAddress - tileT * stride[0];	
			moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, 0, tileT, tileX, tileY, dep_stride, width, segLengthX);	
#ifdef SYNC
			__syncthreads();
			__threadfence();
#endif
#ifdef PRINT_FIRST_BATCH
		printGlobal(dev_arr, width, height, curSMStream);
__threadfence();
__syncthreads();
#endif

//endif FIRST_BATCH
#endif
			write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);
		}

		//when tile = xseg-1, if matrix is completely divided by the tile, no t0 elements copy to shared memory; 
		//use dependent data and out-of-range data to calculate.
		tileIdx = xseg[0] - 1;
		//unlike the other two cases that tileAddress points to the source pos of t0, here tileAddress is the destination pos of t(tileT-1).
		tileAddress = batchStartAddress + tileIdx * tileX[0] - tileT * stride[0]; //might be "timepiece" instead of "tileT"
		read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
#ifdef FIRST_BATCH
		//The last tile of each batch may not have complete tileX data entries at each row. Thus, the size of each row is not tileX.
		moveMatrixToTile(dev_arr, &tile1[0], segLengthX, n1[0] % tileX[0], tileY, dep_stride, tileAddress, width, warpbatch);
#ifdef SYNC
		__threadfence();
		__syncthreads();
#endif
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("move data matrix to tile %d: \n", tileIdx);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif
		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, tileY[0]);
			//set variable offset == 1 if it is the last tile of each batch to copy right-side out-of-range elements to 
			moveIntraDepToTileEdge(&dev_arr[batchStartAddress], &tile1[dep_stride[0] * segLengthX[0]], height, segLengthX, dep_stride, tt, n1, tileY[0], stride, 1);
			moveInterDepToTileEdge(&dev_arr[batchStartAddress], &tile1[0], tileX, dep_stride, stride, n2, segLengthX, width, tileIdx, tt, tt * stride[0] + dep_stride[0], 0);
			__threadfence();
			__syncthreads();
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("curSMStream: %d, curBatch: %d, tileId: %d, timepiece: %d\n", curSMStream, curBatch, tileIdx, timepiece);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
#endif
			//tileX of the last tile is changed throughout the simulation from 0 to tileT;
			//We assume that the size of the data matrix is power of 2 and larger than 32. Thus, there is no remaining data entries.
			for (int tid = threadIdx.x; tid < (tt+1) * stride[0] * tileY[0]; tid += blockDim.x){
				//out-of-range results should be ignored
				//because of the bias, xidx and yidx are the pos of new time elements.
				//thread % tileX and thread / tileX are pos of current cached elements.
				xidx = tid % ((tt + 1) * stride[0]);
				yidx = tid / ((tt + 1) * stride[0]);
			        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1.
				tilePos = stride[0] * segLengthX[0] + stride[0] + yidx * segLengthX[0] + xidx;	
				//newtilePos starts one row above the tile matrix because the next tile is shifted out-side-of the up boundary
				newtilePos = yidx2 * tileX[0] + xidx2;
				//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
				//Because, the edge elements use only the out-of-range elements as dependent data, we need specific manipulation
				if (xidx < (tt+1) * stride[0] && yidx > 0 && yidx < tileY[0] - tt * stride[0]){
					//_9pt_SQUARE_SOR(tile1, tile2, newtilePos, tilePos, stride, segLengthX);
					//_5ptSOR(tile1, tile2, newtilePos, tilePos, stride, segLengthX);
					stencil(tile1, tile2, newtilePos, tilePos, stride, segLengthX);
				}
			}	
			__threadfence();
			__syncthreads();
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("tile2\n");
	printSharedTile(&tile2[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif	
			
			//variable isRegular == 1 because one row is shifted out-side-of the upper boundary.
			//len = n1 % tileX + dep_stride + tt, variable len specifies the lenth of eligible elements should be moved to inter_stream_dep[].
			int len = n1[0] % tileX[0] + tt * stride[0];
			moveTileToInterDep(&inter_stream_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, stride, nextSMStream, tileT, n1, segLengthX, tileIdx, len, 1);
#ifdef SYNC
			__threadfence();
			__syncthreads();
#endif
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("inter_dep: \n");
	printInterDep(&inter_stream_dep[0], tt, tileX, dep_stride, nextSMStream, tileT, n1, tileIdx, len, 1);
}
__threadfence();
__syncthreads();
#endif
			//swap tile2 with tile1;
			swapTile(&tile1[0], &tile2[0], tileX, tileY, segLengthX, dep_stride, warpbatch);
//			swapTile(&tile1[0], &tile2[0], segLengthX, segLengthY, threadsPerBlock);
			__threadfence();
			__syncthreads();
#ifdef PRINT_FIRST_BATCH
if (threadIdx.x == 0){
	printf("tile1 after calculation: \n");
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif	
		}
		//glbPos is the index where the calculated elements should be stored at in the global matrix array.
		//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
		//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
		//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
		//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
		//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
		//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
		//ignLenX == tileX-tileT because tileT elements are copied at each row, ignLenY == tileT because tileY-tileT elements are copied at each column.
		glbPos = tileAddress;	
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileX[0] - tileT * stride[0], tileT * stride[0], tileX, tileY, dep_stride, width, segLengthX);	
#ifdef SYNC
		__threadfence();
		__syncthreads();
#endif
#ifdef PRINT_FIRST_BATCH
		printGlobal(dev_arr, width, height, curSMStream);
__threadfence();
__syncthreads();
#endif
//endif FIRST_BATCH
#endif
		write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);
#ifdef TIME_LOCK
		//at the end of the batch, we need to update the dev_time_lock of the current stream back to "xseg"
		write_time_lock_for_stream(dev_time_lock, curSMStream, xseg, curBatch);
#endif
	}
	else if(curBatch == yseg[0] - 1){
	//version 1.0: we suppose that the data block is evenly divided by the blocks. Thus, the last batch has no dependence on original 
	// 	       data elements when tt == 0.
	//for the last batch, all the tiles are irregular
		//when tile = 0, the calculated data which are outside the range are not copied to tile2, tile size is shrinking 
		//along T dimension. Out-of-range elements are used for dependent data.
		read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
#ifdef TIME_LOCK	
		clear_time_lock_for_stream(dev_time_lock, curSMStream, curBatch);
#endif
#ifdef LAST_BATCH
#ifdef PRINT_LAST_BATCH
if (threadIdx.x == 0){
	printf("This is curBatch: %d, timepiece: %d, curStream: %d, nextSMStream: %d\n", curBatch, timepiece, curSMStream, nextSMStream);
	printf("move data matrix to tile %d: \n", tileIdx);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif
		for (int tt=0; tt<tileT; tt++){
			//for the first tile, intra_dep entries are from the elements which are in the data block but out-of-range.
			//version 1.0: Here we assume that the total length of the shifting does not exceed the size of a single tile.
			//version 1.0: batchStartAddress points to the out-of-range elements for the last batch. Thus, real start position is prior
			//to the batchStartAddress by (tt + stride) rows. Here (tt + stride) might not be correct if stride is larger than 1.
			//the first tile is not in regular size, so variable len = tileX-tt
			moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, dep_stride, curSMStream, tileT, n1, segLengthX, tileIdx, tileX[0] - tt * stride[0]);
			//move out-of-range elements which are beanth the bottom boundary to the tile
			//variable offset == 1, used to locate the bottom out-of-boundary elements.
			moveInterDepToTileEdge(&dev_arr[batchStartAddress], &tile1[0], tileX, dep_stride, stride, n2, segLengthX, width, tileIdx, tt, tileX[0], 1);
			moveIntraDepToTileEdge(&dev_arr[batchStartAddress - (tt * stride[0] + dep_stride[0]) * width[0]], &tile1[0], width, segLengthX, dep_stride, tt, n1, tt * stride[0] + dep_stride[0], stride, 0);
			__threadfence();	
			__syncthreads();
#ifdef PRINT_LAST_BATCH
if (threadIdx.x == 0){
	printf("curSMStream: %d, curBatch: %d, tileId: %d, timepiece: %d\n", curSMStream, curBatch, tileIdx, timepiece);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif	
			for (int tid = threadIdx.x; tid < tileX[0] * (tt + 1) * stride[0]; tid += blockDim.x){
				//out-of-range results should be ignored
				//because of the bias, xidx and yidx are the pos of new time elements.
				//thread % tileX and thread / tileX are pos of current cached elements.
				xidx = tid % tileX[0];
				yidx = tid / tileX[0];
			        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1.
				tilePos = stride[0] * segLengthX[0] + stride[0] + yidx * segLengthX[0] + xidx;	
				//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array.
				//left column shift out-side-of the boundary, so retain all rows but discard the left-most column.
				newtilePos = yidx2 * tileX[0] + xidx2;
				//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
				//Because the edge elements use only the out-of-range elements as dependent data, we need specific manipulation.
				if (xidx>0 && xidx < tileX[0] - tt * stride[0] && yidx < (tt + 1) * stride[0]){
					stencil(tile1, tile2, newtilePos, tilePos, stride, segLengthX);
				}
			}	
			__threadfence();
			__syncthreads();
			
			//Since the tile size is reduced along the calculation, the intraDep elements (in last two column of the valid tile) is also shifted to left.
			//Set variable isRegular == 1, when there is a size reduction. 
			moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, stride, 1, tt * stride[0]);
#ifdef SYNC
			__threadfence();
			__syncthreads();
#endif
#ifdef PRINT_LAST_BATCH
if (threadIdx.x == 0){
	printf("intra_dep: \n");
	printIntraDep(&intra_dep[0], tt, tileY, segLengthX, dep_stride, 1, tt * stride[0]);
}
__threadfence();
__syncthreads();
#endif
			//swap tile2 with tile1;
			swapTile(&tile1[0], &tile2[0], tileX, tileY, segLengthX, dep_stride, warpbatch);
//			swapTile(&tile1[0], &tile2[0], segLengthX, segLengthY, threadsPerBlock);
			__threadfence();
			__syncthreads();
#ifdef PRINT_LAST_BATCH
if (threadIdx.x == 0){
	printf("tile1 after calculation: \n");
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif	
		}
		//glbPos is the index where the calculated elements should be stored at in the global matrix array.
		//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
		//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
		//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
		//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
		//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
		//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
		glbPos = batchStartAddress - tileT * stride[0] * width[0] + tileIdx * tileX[0];
		//ignLenX == tileT because tileX-tileT elements are copied at each row, ignLenY == tileY-tileT because tileT elements are copied at each column.
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileT, tileY[0] - tileT * stride[0], tileX, tileY, dep_stride, width, segLengthX);	
#ifdef SYNC
		__threadfence();
		__syncthreads();
#endif
#ifdef PRINT_LAST_BATCH
		printGlobal(dev_arr, width, height, curSMStream);
__threadfence();
__syncthreads();
#endif
//endif LAST_BATCH
#endif
		write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);

		//tile = 1 to xseg-1; regular size tiles, with index shifting.
		for (tileIdx = 1; tileIdx < xseg[0]-1; tileIdx++){
			read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
#ifdef LAST_BATCH
#ifdef PRINT_LAST_BATCH
if (threadIdx.x == 0){
	printf("move data matrix to tile %d: \n", tileIdx);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif
			for (int tt=0; tt<tileT; tt++){
				//when tileT is larger than 1, the offset should be considered in the shared tile.
				moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, dep_stride, curSMStream, tileT, n1, segLengthX, tileIdx, tileX[0]);
				//move out-of-range elements which are beanth the bottom boundary to the tile
				//variable offset == 1, used to locate the bottom out-of-boundary elements.
				moveInterDepToTileEdge(&dev_arr[batchStartAddress], &tile1[0], tileX, dep_stride, stride, n2, segLengthX, width, tileIdx, tt, tileX[0], 1);
				moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, tt * stride[0]);
				__threadfence();
				__syncthreads();
#ifdef PRINT_LAST_BATCH
if (threadIdx.x == 0){
	printf("curSMStream: %d, curBatch: %d, tileId: %d, timepiece: %d\n", curSMStream, curBatch, tileIdx, timepiece);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif	
				for (int tid = threadIdx.x; tid < tileX[0] * (tt + 1) * stride[0]; tid += blockDim.x){
					//out-of-range results should be ignored
					//because of the bias, xidx and yidx are the pos of new time elements.
					//thread % tileX and thread / tileX are pos of current cached elements.
					yidx = tid % tileX[0];
					xidx = tid / tileX[0];
				        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1
					tilePos = stride[0] * segLengthX[0] + stride[0] + xidx * segLengthX[0] + yidx;	
					//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array
					newtilePos = yidx2 * tileX[0] + xidx2;
					//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
					//Because, the edge elements use only the out-of-range elements as dependent data, we need specific manipulation.
					if (xidx < (tt + 1) * stride[0]){
						stencil(tile1, tile2, newtilePos, tilePos, stride, segLengthX);
					}
				}	
				__threadfence();
				__syncthreads();
				//isRegular == 0 to disable the tile size reduction, when tile size are constant during the calculation. 
				moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, stride, 0, tt * stride[0]);
#ifdef SYNC
				__threadfence();
				__syncthreads();
#endif
#ifdef PRINT_LAST_BATCH
if (threadIdx.x == 0){
	printf("intra_dep: \n");
	printIntraDep(&intra_dep[0], tt, tileY, segLengthX, dep_stride, 1, tt * stride[0]);
}
__threadfence();
__syncthreads();
#endif
				//swap tile2 with tile1;
				swapTile(&tile1[0], &tile2[0], tileX, tileY, segLengthX, dep_stride, warpbatch);
//				swapTile(&tile1[0], &tile2[0], segLengthX, segLengthY, threadsPerBlock);
				__threadfence();
				__syncthreads();
#ifdef PRINT_LAST_BATCH
if (threadIdx.x == 0){
	printf("tile1 after calculation: \n");
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif	
			}						 
			//glbPos is the index where the calculated elements should be stored at in the global matrix array.
			//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
			//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
			//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
			//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
			//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
			//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
			glbPos = batchStartAddress - tileT * stride[0] * width[0] + tileIdx * tileX[0] - tileT * stride[0];
			//ignLenX == 0 because all elements are copied at each row, ignLenY == tileY-tileT because tileT elements are copied at each column.
			moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, 0, tileY[0] - tileT * stride[0], tileX, tileY, dep_stride, width, segLengthX);	
#ifdef SYNC
			__threadfence();
			__syncthreads();
#endif
#ifdef PRINT_LAST_BATCH
		printGlobal(dev_arr, width, height, curSMStream);
__threadfence();
__syncthreads();
#endif
//endif LAST_BATCH
#endif	
			write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);
		}

		//when tile = xseg-1, if matrix is completely divided by the tile, no t0 elements copy to shared memory; 
		//use dependent data and out-of-range data to calculate.
		tileIdx = xseg[0] - 1;
		read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
#ifdef LAST_BATCH
		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, tt * stride[0]);
			//set variable offset == 1 if it is the last tile of each batch to copy right-side out-of-range elements to 
			moveIntraDepToTileEdge(&dev_arr[batchStartAddress - (dep_stride[0] + tt * stride[0]) * width[0]], &tile1[0], height, segLengthX, dep_stride, tt, n1, dep_stride[0] + tt * stride[0], stride, 1);
			
			//1. inter_stream_dep elements from previous tile (on top of intra_dep elements); total size == len + dev_stride, where len == tt, which is 0 at t0
			//2. out-of-range elements
			//copy edge elements first to cover the out-of-range elements, then copy the inter_stream_dep of previous stream and cover a part of the out-of-range elements.
			//variable len == tt + dev_stride, which covers the size of the elements, calculated in previous stream, and the out-of-range elements.
			moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, dep_stride, curSMStream, tileT, n1, segLengthX, tileIdx, tt * stride[0]);
			//move out-of-range elements which are beanth the bottom boundary to the tile
			//variable offset == 1, used to locate the bottom out-of-boundary elements.
			moveInterDepToTileEdge(&dev_arr[batchStartAddress], &tile1[0], tileX, dep_stride, stride, n2, segLengthX, width, tileIdx, tt, tt * stride[0] + dep_stride[0], 1);
			__threadfence();
			__syncthreads();
#ifdef PRINT_LAST_BATCH
if (threadIdx.x == 0){
	printf("curSMStream: %d, curBatch: %d, tileId: %d, timepiece: %d\n", curSMStream, curBatch, tileIdx, timepiece);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif	
			//tileX of the last tile is changed throughout the simulation from 0 to tileT;
			for (int tid = threadIdx.x; tid < (tt+1) * stride[0] * tileY[0]; tid += blockDim.x){
				//out-of-range results should be ignored
				//because of the bias, xidx and yidx are the pos of new time elements.
				//thread % tileX and thread / tileX are pos of current cached elements.
				yidx = tid % tileX[0];
				xidx = tid / tileX[0];
			        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1.
				tilePos = stride[0] * segLengthX[0] + stride[0] + xidx * segLengthX[0] + yidx;	
				//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array.
				newtilePos = yidx2 * tileX[0] + xidx2;
				//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
				//Because, the edge elements use only the out-of-range elements as dependent data, we need specific manipulation
				if (xidx < (tt + 1) * stride[0] && yidx < (tt + 1) * stride[0]){
					stencil(tile1, tile2, newtilePos, tilePos, stride, segLengthX);
				}
			}	
			__threadfence();
			__syncthreads();
			
			//swap tile2 with tile1;
			swapTile(&tile1[0], &tile2[0], tileX, tileY, segLengthX, dep_stride, warpbatch);
			//swapTile(&tile1[0], &tile2[0], segLengthX, segLengthY, threadsPerBlock);
			__threadfence();
			__syncthreads();
#ifdef PRINT_LAST_BATCH
if (threadIdx.x == 0){
	printf("tile1 after calculation: \n");
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif	
		}
		//glbPos is the index where the calculated elements should be stored at in the global matrix array.
		//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
		//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
		//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
		//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
		//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
		//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
		//unlike the other two cases that glbPos points to the source pos of t0, here tileAddress is the destination pos of t(tileT-1).
		glbPos = batchStartAddress - tileT * stride[0] * width[0] + tileIdx * tileX[0] - tileT * stride[0];	
		//ignLenX == tileX-tileT because tileT elements are copied at each row, ignLenY == tileY-tileT because tileT elements are copied at each column.
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileX[0] - tileT * stride[0], tileY[0] - tileT * stride[0], tileX, tileY, dep_stride, width, segLengthX);	
#ifdef SYNC
		__threadfence();
		__syncthreads();
#endif
#ifdef PRINT_LAST_BATCH
		printGlobal(dev_arr, width, height, curSMStream);
__threadfence();
__syncthreads();
#endif
//endif LAST_BATCH
#endif	
		write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);
	
#ifdef TIME_LOCK
		//at the end of the batch, we need to update the dev_time_lock of the current stream back to "xseg"
		write_time_lock_for_stream(dev_time_lock, curSMStream, xseg, curBatch);
#endif
	}
	else{
	//for the regular batch, use the near-edge elements for the out-of-range dependence of first and last tile only.
		//when tile = 0, the calculated data which are outside the range are not copied to tile2, tile size is shrinking 
		//along T dimension. Out-of-range elements are used for dependent data.
		tileAddress = batchStartAddress + tileIdx * tileX[0];
		read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
#ifdef TIME_LOCK	
		read_time_lock_for_stream(dev_time_lock, curSMStream, nextSMStream, xseg, curBatch);
		clear_time_lock_for_stream(dev_time_lock, curSMStream, curBatch);
#endif
#ifdef MID_BATCH
		moveMatrixToTile(dev_arr, &tile1[0], segLengthX, tileX[0], tileY, dep_stride, tileAddress, width, warpbatch);
#ifdef SYNC
		__threadfence();
		__syncthreads();
#endif
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("This is curBatch: %d, timepiece: %d, curStream: %d, nextSMStream: %d\n", curBatch, timepiece, curSMStream, nextSMStream);
	printf("current global data entries.\n");
	printGlobal(dev_arr, width, height, curSMStream);
	printf("move data matrix to tile %d: \n", tileIdx);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif
		for (int tt = 0; tt < tileT; tt++){
			//moveIntraDepToTileEdge() should include two parts: 
			//1. the elements in prior to the tile entries; 
			moveIntraDepToTileEdge(&dev_arr[batchStartAddress - tt * stride[0] * width[0]], &tile1[dep_stride[0] * segLengthX[0]], width, segLengthX, dep_stride, tt, n1, tileY[0], stride, 0);
			//the first tile is not in regular size, so variable len = tileX-tt
			moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, dep_stride, curSMStream, tileT, n1, segLengthX, tileIdx, tileX[0] - tt * stride[0]);
			//2. the elements in prior to the inter-dep entries
			moveIntraDepToTileEdge(&dev_arr[batchStartAddress - (dep_stride[0] + tt * stride[0]) * width[0]], &tile1[0], width, segLengthX, dep_stride, tt, n1, tileY[0], stride, 0);
			__threadfence();
			__syncthreads();
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("curSMStream: %d, curBatch: %d, tileId: %d, timepiece: %d\n", curSMStream, curBatch, tileIdx, timepiece);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif
			for (int tid = threadIdx.x; tid < tileX[0] * tileY[0]; tid += blockDim.x){
				//out-of-range results should be ignored
				//because of the bias, xidx and yidx are the pos of new time elements.
				//thread % tileX and thread / tileX are pos of current cached elements.
				xidx = tid % tileX[0];
				yidx = tid / tileX[0];
			        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1.
				tilePos = stride[0] * segLengthX[0] + stride[0] + yidx * segLengthX[0] + xidx;	
				//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array.
				//left column shift out-side-of the boundary, so retain all rows but discard the left-most column.
				newtilePos = yidx2 * tileX[0] + xidx2;
				//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
				//Because the edge elements use only the out-of-range elements as dependent data, we need specific manipulation.
				if (xidx > 0 && xidx < tileX[0] - tt * stride[0]){
					stencil(tile1, tile2, newtilePos, tilePos, stride, segLengthX);
				}
			}	
			__threadfence();
			__syncthreads();
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("tile2\n");
	printSharedTile(&tile2[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif				
			
			//Since the tile size is reduced along the calculation, the intraDep elements (in last two column of the valid tile) is also shifted to left.
			//Set variable isRegular == 1, when there is a size reduction. 
			moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, stride, 1, tileY[0]);
#ifdef SYNC
			__threadfence();
			__syncthreads();
#endif
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("intra_dep: \n");
	printIntraDep(&intra_dep[0], tt, tileY, segLengthX, dep_stride, 1, tileY[0]);
}
__threadfence();
__syncthreads();
#endif
			//first tile has to copy the out-of-range elements, which are on the left-hand side, to next stream's inter_stream_dep array
			moveTileToInterDepEdge(&tile1[0], inter_stream_dep, tt, tileY, tileT, nextSMStream, dep_stride, stride, segLengthX, n1, 0);
#ifdef SYNC
			__threadfence();
			__syncthreads();
#endif
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("inter_dep_edge: \n");
	printInterDepEdge(inter_stream_dep, tt, tileT, nextSMStream, dep_stride, segLengthX, n1);
}
__threadfence();
__syncthreads();
#endif			
			//variable len == tileX-tt because the tile size is reduced during calculation.
			//isRegular == 0 because there is no row move out-side-of upper boundary
			moveTileToInterDep(&inter_stream_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, stride, nextSMStream, tileT, n1, segLengthX, tileIdx, tileX[0] - tt * stride[0], 0);
#ifdef SYNC
			__threadfence();
			__syncthreads();
#endif
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("inter_dep: \n");
	printInterDep(&inter_stream_dep[0], tt, tileX, dep_stride, nextSMStream, tileT, n1, tileIdx, tileX[0] - tt * stride[0], 0);
}
__threadfence();
__syncthreads();
#endif
			//swap tile2 with tile1;
			swapTile(&tile1[0], &tile2[0], tileX, tileY, segLengthX, dep_stride, warpbatch);
			//swapTile(&tile1[0], &tile2[0], segLengthX, segLengthY, threadsPerBlock);
			__threadfence();
			__syncthreads();
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("tile1 after calculation: \n");
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif	
		}
		//glbPos is the index where the calculated elements should be stored at in the global matrix array. Shifting should be considered.
		//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
		//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
		//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
		//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
		//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
		//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
		//ignLenX == tileT because tileX-tileT elements are copied at each row, ignLenY == 0 because no size reduction along Y dim.
		glbPos = tileAddress - tileT * stride[0] * width[0];	
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileT, 0, tileX, tileY, dep_stride, width, segLengthX);	
#ifdef SYNC
		__threadfence();
		__syncthreads();
#endif
#ifdef PRINT_MID_BATCH
		printGlobal(dev_arr, width, height, curSMStream);
__threadfence();
__syncthreads();
#endif
//endif MID_BATCH
#endif
		write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);

		//tile = 1 to xseg-1; regular size tiles, with index shifting.
		for (tileIdx = 1; tileIdx < xseg[0]-1; tileIdx++){
			tileAddress = batchStartAddress + tileIdx * tileX[0];
			read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
			//copy the base spatial data to shared memory for t=0.
#ifdef MID_BATCH
			moveMatrixToTile(dev_arr, &tile1[0], segLengthX, tileX[0], tileY, dep_stride, tileAddress, width, warpbatch);
#ifdef SYNC
			__threadfence();
			__syncthreads();
#endif
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("move data matrix to tile %d: \n", tileIdx);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif
			for (int tt=0; tt<tileT; tt++){
				//isRegular == 0 because this is a regular tile.
				moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, tileY[0]);
				moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, dep_stride, curSMStream, tileT, n1, segLengthX, tileIdx, tileX[0]);
				__threadfence();
				__syncthreads();
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("curSMStream: %d, curBatch: %d, tileId: %d, timepiece: %d\n", curSMStream, curBatch, tileIdx, timepiece);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif
				for (int tid = threadIdx.x; tid < tileX[0] * tileY[0]; tid += blockDim.x){
					//out-of-range results should be ignored
					//because of the bias, xidx and yidx are the pos of new time elements.
					//thread % tileX and thread / tileX are pos of current cached elements.
					xidx = tid % tileX[0];
					yidx = tid / tileX[0];
				        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1
					tilePos = stride[0] * segLengthX[0] + stride[0] + yidx * segLengthX[0] + xidx;	
					//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array
					newtilePos = yidx2 * tileX[0] + xidx2;
					stencil(tile1, tile2, newtilePos, tilePos, stride, segLengthX);
				}	
				__threadfence();
				__syncthreads();
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("tile2\n");
	printSharedTile(&tile2[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif				
				//isRegular == 0 to disable the tile size reduction, when tile size are constant during the calculation. 
				moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, stride, 0, tileY[0]);
#ifdef SYNC
				__threadfence();
				__syncthreads();
#endif
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("intra_dep: \n");
	printIntraDep(&intra_dep[0], tt, tileY, segLengthX, dep_stride, 0, tileY[0]);
}
__threadfence();
__syncthreads();
#endif
				//variable len == tileX because the tile size is constant.
				moveTileToInterDep(&inter_stream_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, stride, nextSMStream, tileT, n1, segLengthX, tileIdx, tileX[0], 0);
#ifdef SYNC
				__threadfence();
				__syncthreads();
#endif
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("inter_dep: \n");
	printInterDep(&inter_stream_dep[0], tt, tileX, dep_stride, nextSMStream, tileT, n1, tileIdx, tileX[0], 0);
}
__threadfence();
__syncthreads();
#endif
				//swap tile2 with tile1;
				swapTile(&tile1[0], &tile2[0], tileX, tileY, segLengthX, dep_stride, warpbatch);
//				swapTile(&tile1[0], &tile2[0], segLengthX, segLengthY, threadsPerBlock);
				__threadfence();
				__syncthreads();
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("tile1 after calculation: \n");
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif	
			}						 
			//glbPos is the index where the calculated elements should be stored at in the global matrix array.
			//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
			//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
			//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
			//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
			//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
			//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
			glbPos = tileAddress - tileT * stride[0] * width[0] - tileT * stride[0];	
			moveShareToGlobal(&tile1[0], dev_arr, glbPos, tileX, tileY, dep_stride, width, segLengthX);	
#ifdef SYNC
			__threadfence();
			__syncthreads();
#endif
#ifdef PRINT_MID_BATCH
		printGlobal(dev_arr, width, height, curSMStream);
__threadfence();
__syncthreads();
#endif
			write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);
//endif MID_BATCH
#endif
		}

		//when tile = xseg-1, if matrix is completely divided by the tile, no t0 elements copy to shared memory; 
		//use dependent data and out-of-range data to calculate.
		tileIdx = xseg[0] - 1;
		//unlike the other two cases that tileAddress points to the source pos of t0, here tileAddress is the destination pos of t(tileT-1).
		tileAddress = batchStartAddress + tileIdx * tileX[0] - tileT * stride[0];
		read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
#ifdef MID_BATCH
		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, tileY[0]);
			//set variable offset == 1 if it is the last tile of each batch to copy right-side out-of-range elements to 
			moveIntraDepToTileEdge(&dev_arr[batchStartAddress - tt * stride[0] * width[0]], &tile1[dep_stride[0] * segLengthX[0]], width, segLengthX, dep_stride, tt, n1, tileY[0], stride, 1);
			//1. inter_stream_dep elements from previous tile (on top of intra_dep elements); total size == len + dev_stride, where len == tt, which is 0 at t0
			//2. out-of-range elements
			//copy edge elements first to cover the out-of-range elements, then copy the inter_stream_dep of previous stream and cover a part of the out-of-range elements.
			//variable len == tt + dev_stride, which covers the size of the elements, calculated in previous stream, and the out-of-range elements.
			moveInterDepToTileEdge(&dev_arr[batchStartAddress - tt * stride[0] * width[0]], &tile1[0], tileX, dep_stride, stride, n2, segLengthX, width, tileIdx, tt, tt * stride[0] + dep_stride[0], 0);
			moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, dep_stride, curSMStream, tileT, n1, segLengthX, tileIdx, tt * stride[0]);
			__threadfence();
			__syncthreads();
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("curSMStream: %d, curBatch: %d, tileId: %d, timepiece: %d\n", curSMStream, curBatch, tileIdx, timepiece);
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif
			//tileX of the last tile is changed throughout the simulation from 0 to tileT;
			for (int tid = threadIdx.x; tid < (tt+1) * stride[0] * tileY[0]; tid += blockDim.x){
				//out-of-range results should be ignored
				//because of the bias, xidx and yidx are the pos of new time elements.
				//thread % tileX and thread / tileX are pos of current cached elements.
				xidx = tid % ((tt + 1) * stride[0]);
				yidx = tid / ((tt + 1) * stride[0]);
			        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1.
				tilePos = stride[0] * segLengthX[0] + stride[0] + yidx * segLengthX[0] + xidx;	
				//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array.
				newtilePos = yidx2 * tileX[0] + xidx2;
				//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
				//Because, the edge elements use only the out-of-range elements as dependent data, we need specific manipulation
				if (xidx < (tt + 1) * stride[0]){
					stencil(tile1, tile2, newtilePos, tilePos, stride, segLengthX);
				}
			}	
			__threadfence();
			__syncthreads();
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("tile2\n");
	printSharedTile(&tile2[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif				
			//variable isRegular == 0 because one row is shifted out-side-of the upper boundary.
			//len = tileX-1-tt, variable len specifies the lenth of eligible elements should be moved to inter_stream_dep[].
			int len = n1[0] % tileX[0] + dep_stride[0] + tt * stride[0];
			moveTileToInterDep(&inter_stream_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, stride, nextSMStream, tileT, n1, segLengthX, tileIdx, len, 0);
#ifdef SYNC
			__threadfence();
			__syncthreads();
#endif
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("inter_dep: \n");
	printInterDep(&inter_stream_dep[0], tt, tileX, dep_stride, nextSMStream, tileT, n1, tileIdx, len, 0);
}
__threadfence();
__syncthreads();
#endif
			//swap tile2 with tile1;
			swapTile(&tile1[0], &tile2[0], tileX, tileY, segLengthX, dep_stride, warpbatch);
			//swapTile(&tile1[0], &tile2[0], segLengthX, segLengthY, threadsPerBlock);
			__threadfence();
			__syncthreads();
#ifdef PRINT_MID_BATCH
if (threadIdx.x == 0){
	printf("tile1 after calculation: \n");
	printSharedTile(&tile1[0], segLengthX, tileY, dep_stride, curSMStream);
}
__threadfence();
__syncthreads();
#endif	
		}
		//glbPos is the index where the calculated elements should be stored at in the global matrix array.
		//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
		//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
		//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
		//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
		//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
		//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
		//ignLenX == tileX-tileT because only tileT elements are copied in each row, ignLenY == 0 because no size reduction along Y dim.
		glbPos = tileAddress - tileT * stride[0] * width[0];	
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileX[0] - tileT * stride[0], 0, tileX, tileY, dep_stride, width, segLengthX);	
#ifdef SYNC
		__threadfence();
		__syncthreads();
#endif
#ifdef PRINT_MID_BATCH
		printGlobal(dev_arr, width, height, curSMStream);
__threadfence();
__syncthreads();
#endif
//endif MID_BATCH
#endif
		write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);
#ifdef TIME_LOCK
		//at the end of the batch, we need to update the dev_time_lock of the current stream back to "xseg"
		write_time_lock_for_stream(dev_time_lock, curSMStream, xseg, curBatch);
#endif
	}

//	write_batch_lock_for_time(timepiece, curBatch);
}


void checkGPUError(cudaError err){
	if (cudaSuccess != err){
		printf("CUDA error in file %s, in line %i: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void SOR(int n1, int n2, int stride, int padd, int *arr, int MAXTRIAL){
	cudaSetDevice(0);	
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
#ifndef RTX_2080
	const int MAX_THREADS_PER_BLOCK = 1024;
#else
	const int MAX_THREADS_PER_BLOCK = 1024;
#endif
//stride is the longest distance between the element and its dependence along one dimension times
//For example: F(x) = T(x-1) + T(x) + T(x+1), stride = 1
//when we change stride, we also need to update parameter "paddsize" in data generator file.
	//padd = 2 * stride
	int dep_stride = padd;
	int tileX = 32;
	int tileY = 64;
	//the shared memory available for intra_dep array.
#ifndef RTX_2080
	int intra_size = 48 / (int)sizeof(int) * 1024 - (tileX + dep_stride) * (tileY + dep_stride) * 2;
#else
	int intra_size = 48 / (int)sizeof(int) * 1024 - (tileX + dep_stride) * (tileY + dep_stride) * 2;
#endif
	//tileT is restriced by "tileY" and "intra_dep" shared array size.
	int tileT = min(16, min(tileX, tileY));
	tileT = tileT > MAXTRIAL ? MAXTRIAL : tileT;
	//compare required shared memory to available shared memory for intra_dep array.
	while (tileT * dep_stride * tileY > intra_size){
		tileT /= 2;
	}
	//because of the shift during calculation, the total shift distance for tileT iteration must be smaller than min(tileX, tileY).
	//total shift distance = tileT * stride;
	tileT = (tileT * stride) > min(tileX, tileY) ? min(tileX, tileY) / stride - 1 : tileT;
	while (MAXTRIAL % tileT != 0){
		tileT--;
	}
	if (tileT < 2){
		cout << "time tile size is smaller than 2, please reduce tileX and tileY size to reserve larger shared memory capacity for intra_dep array." << endl;
		exit(-1);
	}
	cout << "tileT: " << tileT << endl;

	int xseg = n1 / tileX + 1;
	int yseg = n2 / tileY + 1;
	int tseg = (MAXTRIAL + tileT - 1) / tileT;
#ifndef RTX_2080
	int numStream = min(28, yseg);
#else
	int numStream = min(68, yseg);
#endif
	int stream_offset = yseg % numStream;

#ifdef PRINT_FINAL_RESULT
	cout << "Final Matrix after a total of " << MAXTRIAL << " time stamps, which are completed as a batch for every " << tileT << endl;
#endif
	int width = 2 * padd + n1; 
	int height = 2 * padd + n2;

	volatile int *dev_arr;
	int *lock;
	volatile int *dev_time_lock, *dev_row_lock;	
	int *dev_var, *var;

	int tablesize = width * height;
#ifdef DEBUG
	size_t freeMem, totalMem;
	cudaMemGetInfo(&freeMem, &totalMem);
	cout << "current GPU memory info FREE: " << freeMem << " Bytes, Total: " << totalMem << " Bytes." << endl;
	cout << "width: " << width << ", height: " << height << ", allocates: " << tablesize * sizeof(int)<< " Bytes." << endl;
#endif
	cudaError err = cudaMalloc(&dev_arr, tablesize * sizeof(int));
	checkGPUError(err);
	
	err = cudaMemcpy((void*)dev_arr, arr, tablesize*sizeof(int), cudaMemcpyHostToDevice);
	checkGPUError(err);

	int threadPerBlock = MAX_THREADS_PER_BLOCK;
	int blockPerGrid = 1;
	int warpbatch = MAX_THREADS_PER_BLOCK / 32;

//memory structure: stream --> tile --> time --> dependence --> tileX
	int *dev_inter_stream_dependence;
	int stream_dep_offset = tileT * (n1 + dep_stride) * dep_stride;
	int inter_stream_dependence = numStream * stream_dep_offset;
	err = cudaMalloc(&dev_inter_stream_dependence, inter_stream_dependence * sizeof(int));
	checkGPUError(err);

	lock = new int[tseg * yseg];
	for (int i = 0; i < tseg; i++){
		int idx = i * yseg;
		lock[idx] = xseg;
		for (int j=1; j<yseg; j++)
			lock[idx+j] = 0;
	}

	err = cudaMalloc(&dev_row_lock, tseg * yseg * sizeof(int));
	checkGPUError(err);
	err = cudaMemcpy((void*)dev_row_lock, lock, tseg * yseg *sizeof(int), cudaMemcpyHostToDevice);
	checkGPUError(err);
//	cudaMalloc(&dev_time_lock, n2/tileY * sizeof(int));
//	cudaMemset((void*)dev_time_lock, 1, n2/tileY * sizeof(int));
	int* time_lock = new int[numStream];
	for (int i = 0; i < numStream; i++){
		time_lock[i] = xseg;
	}
	err = cudaMalloc(&dev_time_lock, numStream * sizeof(int));
	checkGPUError(err);
	err = cudaMemcpy((void*)dev_time_lock, time_lock, numStream * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaStream_t stream[numStream];
	for (int s=0; s<numStream; s++)
		cudaStreamCreate(&stream[s]);

	var = new int[11];
	var[0] = tileX; var[1] = tileY; var[2] = padd; var[3] = stride; var[4] = height; var[5] = width;
	var[6] = xseg; var[7] = yseg; var[8] = n1; var[9] = n2; var[10] = warpbatch;
	err = cudaMalloc(&dev_var, 11 * sizeof(int));
	checkGPUError(err);
	err = cudaMemcpy(dev_var, var, 11 * sizeof(int), cudaMemcpyHostToDevice);

	struct timeval tbegin, tend;
	gettimeofday(&tbegin, NULL);

//t < MAXTRIAL? or t <= MAXTRIAL	
	for(int t = 0; t < MAXTRIAL; t+= tileT){
		for(int curBatch = 0; curBatch < yseg; curBatch++){
//Have to change the stream Index so that the stream for next time tile can start without waiting for the 
//completion of the previous time tile. 
//Example: stream 0, 1, 2 are scheduled to the last three batches in one time tile, since the execution on
//the next time tile also starts from stream 0, this new execution in stream 0 has to wait for the previous
			int timepiece = t / tileT;
			int logicSMStream = curBatch % numStream;
			int curSMStream = (logicSMStream +  stream_offset * timepiece) % numStream;
			int curStartAddress = curBatch * tileY * width;
			int rowStartOffset = padd * width + padd;
			int batchStartAddress = rowStartOffset + curStartAddress;
			int nextSMStream = (curSMStream + 1) % numStream;
//			cout << "curBatch: " << curBatch << ", stride: " << stride << ", tileX: " << tileX << ", tileY: " << tileY << ", t: " << t << ", xseg: " << xseg << ", yseg: " << yseg << ", logicStream: " << logicSMStream << ", curStream: " << curSMStream  << endl;	
//			GPU_Tile<<<blockPerGrid, threadPerBlock, 0, stream[curSMStream]>>>(dev_arr, curBatch, tileX, tileY, padd, stride, height, width, xseg, yseg, n1, n2, warpbatch, curSMStream, nextSMStream, dev_inter_stream_dependence, inter_stream_dependence, tileT, timepiece, batchStartAddress, dev_row_lock, dev_time_lock);	
			GPU_Tile<<<blockPerGrid, threadPerBlock, 0, stream[curSMStream]>>>(dev_arr, curBatch, dev_var, curSMStream, nextSMStream, dev_inter_stream_dependence, inter_stream_dependence, tileT, timepiece, batchStartAddress, dev_row_lock, dev_time_lock, threadPerBlock);	
			checkGPUError( cudaGetLastError() );
		}
		//this global synchronization enforces the sequential computation along t dimension.
//		cudaDeviceSynchronize();
	}
	cudaDeviceSynchronize();
	gettimeofday(&tend, NULL);

	err = cudaMemcpy(arr, (void*)dev_arr, tablesize*sizeof(int), cudaMemcpyDeviceToHost);
	checkGPUError(err);
#ifdef PRINT_FINAL_RESULT
       	for (int r = 0; r < height; r++){
		for (int c = 0; c < width; c++){
			cout << arr[r * width + c] << " ";
		}
		cout << endl;
	}
	cout << endl;
#endif
	double s = (double)(tend.tv_sec - tbegin.tv_sec) + (double)(tend.tv_usec - tbegin.tv_usec)/1000000.0;
	cout << "execution time: " << s << " second." << endl;

	for (int s=0; s<numStream; s++)
		cudaStreamDestroy(stream[s]);
	
	cudaFree((void*)dev_arr);
	cudaFree((void*)dev_row_lock);
	cudaFree((void*)dev_inter_stream_dependence);
	delete[] lock;
	delete[] time_lock;
}

