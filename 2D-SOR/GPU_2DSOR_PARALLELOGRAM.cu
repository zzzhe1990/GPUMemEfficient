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
#define PRINT_MATRIX

const int MAX_THREADS_PER_BLOCK = 512;
using namespace std;
__device__ int row = 0;

__device__ void printSharedTile(int* tile, int segLengthX, int tileX, int tileY, int dep_stride, int curSMStream){
	if (threadIdx.x == 0 && curSMStream == 0){
		for (int row = 0; row < dep_stride + tileY; row++){
			for (int col = 0; col < segLengthX; col++){
				printf("%d ", tile[row * segLengthX + col]);
			}
			printf("\n");
		}
	}
}

__device__ void printGlobal(volatile int* dev_arr, int width, int height, int curSMStream){
	if (threadIdx.x == 0 && curSMStream == 0){
		for (int r = 0; r < height; r++){
			for (int c = 0; c < width; c++){
				int pos = r * width + c;
				printf("%d ", dev_arr[pos]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

__device__ void moveMatrixToTile(volatile int* dev_arr, int* tile, int segLengthX, int tileX, int tileY, int dep_stride, int tileAddress, int width, int warpbatch){
	int idx = threadIdx.x % 32;
	int warpidx = threadIdx.x / 32;
	if (warpidx < tileY){
		int glbpos = tileAddress + warpidx * width;
		int shrpos = dep_stride * segLengthX + warpidx * segLengthX + dep_stride;
//	if (threadIdx.x < segLengthX)
//		table[threadIdx.x] = dev_table[tileAddress + threadIdx.x];
		for (; warpidx < tileY; warpidx += warpbatch){
			for (int i = idx; i < tileX; i += 32){
				tile[shrpos+i] = dev_arr[glbpos+i];
			}
			shrpos += (warpbatch * segLengthX);
			glbpos += (warpbatch * width);
		}
	}
	__syncthreads();
}

//intra_dep array structure: tileT * dep_stride * tileY
__device__ void moveIntraDepToTile(int* intra_dep, int* tile, int tt, int tileY, int segLengthX, int dep_stride, int len){
	//at each tt, (stride+1) dependent data are required at x axis.
	//only the threads, which are within tileY are working here.
	//threadPerBlock has to be no less than tileY * dep_stride
	if (threadIdx.x < len * dep_stride){
		int pos = tt * dep_stride * tileY + threadIdx.x;
		int tilepos = dep_stride * segLengthX + threadIdx.x/dep_stride * segLengthX + threadIdx.x % dep_stride;
		tile[tilepos] = intra_dep[pos];
	}
}

__device__ void moveIntraDepToTileEdge(volatile int* dev_arr, int* tile, int stride, int height, int segLengthX, int dep_stride, int tt, int padd, int n1, int len, int offset = 0){
	//copy out-of-range data to tile
	if (threadIdx.x < len * dep_stride){
		int glbpos = padd * height + (padd - dep_stride) + offset * (n1 + dep_stride) + threadIdx.x/dep_stride * height + threadIdx.x % dep_stride;
		int tilepos = dep_stride * segLengthX + threadIdx.x/dep_stride * segLengthX + threadIdx.x % dep_stride + offset * (dep_stride + tt);
		tile[tilepos] = dev_arr[glbpos];
	}
}

__device__ void moveTileToIntraDep(int* intra_dep, int* tile, int tt, int tileX, int tileY, int segLengthX, int dep_stride, int isRegular, int len){
	if (threadIdx.x < len * dep_stride){
		int pos = tt * dep_stride * tileY + threadIdx.x;
		int tilepos = dep_stride * segLengthX + tileX - tt * isRegular;
	       	tilepos	+= threadIdx.x/dep_stride * segLengthX + threadIdx.x % dep_stride;
		intra_dep[pos] = tile[tilepos];
	}
#ifdef PRINT_MATRIX
	if (threadIdx.x == 0){
		for (int i = 0; i < len * dep_stride; i++){
			int pos = tt * dep_stride * tileY + i;
			printf("%d ", intra_dep[pos]);
		}
		printf("\n");
	}
#endif
}

//inter_stream_dep array structure: stream * tileT * dep_stride * (n1 + dep_stride)
__device__ void moveInterDepToTile(int* inter_stream_dep, int* tile, int tt, int tileX, int tileY, int dep_stride, int stream, int tileT, int n1, int segLengthX, int tileIdx, int len){
	int startAddress = (stream * tileT + tt) * dep_stride * (n1 + dep_stride);
	if (tileIdx > 0)       
		startAddress += ( (tileIdx-1) * tileX + tileX-tt );
	startAddress += ( tileIdx * tileX);
	//variable len specifies the eligible elements should be moved. This is caused by the irregular tile.
	if (threadIdx.x < len + dep_stride){
		int pos = startAddress + threadIdx.x;
		int tilepos = threadIdx.x;
		for (int i=0; i<dep_stride; i++){
	 		tile[tilepos] = inter_stream_dep[pos];
			pos += (n1 + dep_stride);
			tilepos += segLengthX;
		}
	}	
}

__device__ void moveInterDepToTileEdge(volatile int* dev_arr, int* tile, int tileX, int tileY, int dep_stride, int n2, int segLengthX, int padd, int width, int tileIdx, int tt, int len, int offset){
	int glbpos = (padd - dep_stride) * width + offset * (dep_stride + n2) * width + padd - dep_stride + threadIdx.x;
	if (tileIdx > 0)
		glbpos += ((tileIdx-1) * tileX + tileX-tt);
	if (threadIdx.x < len + dep_stride){
		int tilepos = offset * (dep_stride + len) * segLengthX + threadIdx.x;
		for (int i=0; i<dep_stride; i++){
			tile[tilepos] = dev_arr[glbpos];
			tilepos += segLengthX;
			glbpos += width;
		}
	}
}

__device__ void moveTileToInterDep(int* inter_stream_dep, int* tile, int tt, int tileX, int tileY, int dep_stride, int nextSMStream, int tileT, int n1, int segLengthX, int tileIdx, int len, int isRegular){
	int startAddress = dep_stride + (nextSMStream * tileT + tt) * dep_stride * (n1 + dep_stride);
	//for the edge tiles, the size is irregular so that the start position of some tt timestamp are not times of tileX.
	if (tileIdx > 0)       
		startAddress += ( (tileIdx-1) * tileX + tileX-tt );
	//variable len specifies the eligible elements should be moved. This is caused by the irregular tile.
	if (threadIdx.x < len){
		int pos = startAddress + threadIdx.x;
		//tt * isRegular ? (tt + 1) * isRegular
		int tilepos = dep_stride + (tileY - tt * isRegular) * segLengthX + threadIdx.x;
		for (int i=0; i<dep_stride; i++){
	 		inter_stream_dep[pos] = tile[tilepos];
			pos += (n1 + dep_stride);
			tilepos += segLengthX;
		}
	}
#ifdef PRINT_MATRIX
	if (threadIdx.x == 0){
		printf("nextSMStream: %d, tileIdx: %d, tt: %d, startAddress: %d\n", nextSMStream, tileIdx, tt, startAddress);
		printf("isRegular: %d\n", isRegular);
		for (int i = 0; i < len; i++){
			int pos = startAddress + i;
			int tilepos = dep_stride + (tileY - tt * isRegular) * segLengthX + i;
			for (int j=0; j<dep_stride; j++){
		 		printf("%d ",inter_stream_dep[pos]);
				pos += (n1 + dep_stride);
			}			
		}
		printf("\n");
	}
#endif	
}

__device__ void moveTileToInterDepEdge(volatile int* tile, int* inter_stream_dep, int tt, int tileX, int tileY, int tileT, int nextSMStream, int dep_stride, int segLengthX, int tileIdx, int n1){
	int startAddress = (nextSMStream * tileT + tt) * dep_stride * (n1 + dep_stride);
//	int glbpos = padd * width + curBatch * tileY * width + (padd - dep_stride) + (tileY - dep_stride) * width;
	int shrpos = dep_stride * segLengthX + (tileY - dep_stride) * segLengthX;  
	if (threadIdx.x < dep_stride){
		int interpos = startAddress + threadIdx.x;
		int pos = shrpos + threadIdx.x;
		for (int i=0; i<dep_stride; i++){
	 		inter_stream_dep[interpos] = tile[pos];
			pos += segLengthX;
			interpos += (n1 + dep_stride);
		}
	}
#ifdef PRINT_MATRIX
	if (threadIdx.x == 0){
		printf("nextSMStream: %d, tt: %d, startAddress: %d\n", nextSMStream, tt, startAddress);
		printf("segLengthX: %d, shrpos: %d\n", segLengthX, shrpos);
		for (int i = 0; i < dep_stride; i++){
			int interpos = startAddress + i;
			for (int j = 0; j < dep_stride; j++){
				printf("%d ", inter_stream_dep[interpos]);
				interpos += (n1 + dep_stride);
			}		
		}
		printf("\n");
	}	
#endif	
}

__device__ void moveShareToGlobalEdge(int* tile, volatile int* dev_arr, int startPos, int ignLenX, int ignLenY, int tileX, int tileY, int dep_stride, int height, int segLengthX){
	int xidx, yidx, glbPos, tilePos;
	for (int tid = threadIdx.x; tid < tileX * tileY; tid += blockDim.x){
		xidx = tid % tileX;
		yidx = tid / tileX;
		if (xidx < tileX - ignLenX && yidx < tileY - ignLenY){
			glbPos = startPos + yidx * height + xidx;
			tilePos = (dep_stride + yidx) * segLengthX + dep_stride + xidx;
			dev_arr[glbPos] = tile[tilePos];
		}
	}	
}	

__device__ void moveShareToGlobal(int* tile, volatile int* dev_arr, int startPos, int tileX, int tileY, int dep_stride, int height, int segLengthX){
	int xidx, yidx, glbPos, tilePos;
	for (int tid = threadIdx.x; tid < tileX * tileY; tid += blockDim.x){
		xidx = tid % tileX;
		yidx = tid / tileX;
		glbPos = startPos + yidx * height + xidx;
		tilePos = (dep_stride + yidx) * segLengthX + dep_stride + xidx;
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
__device__ void read_tile_lock_for_batch(volatile int* dev_row_lock, int curBatch, int tileIdx, int YoverX, int xseg, int yseg, int timepiece){
	if (threadIdx.x == 0){
		int limit = min(tileIdx + YoverX, xseg);
		while(dev_row_lock[timepiece * yseg + curBatch] < limit){
		}
#ifdef DEBUG
		printf("curBatch: %d, tileIdx: %d, timepiece: %d, value: %d, limit: %d\n", curBatch, tileIdx, timepiece, dev_row_lock[timepiece*yseg+curBatch], limit);
#endif
	}
	__syncthreads();
}

__device__ void write_tile_lock_for_batch(volatile int* dev_row_lock, int curBatch, int yseg, int timepiece){
	if (threadIdx.x == 0){
		dev_row_lock[timepiece * yseg + curBatch + 1] += 1;
#ifdef DEBUG
		printf("curBatch: %d, timepiece: %d, update to lock at: %d, value: %d\n", curBatch, timepiece, timepiece*yseg+curBatch+1, dev_row_lock[timepiece*yseg+curBatch+1]);
#endif
	}
	__syncthreads();
}

//__global__ void GPU_Tile(int stride, int tileX, int tileY, int curBatch, int batchStartAddress, int* dev_row_lock, int timepiece, int xseg, int yseg, int tileT){
__global__ void GPU_Tile(volatile int* dev_arr, int curBatch, int curStartAddress, int tileX, int tileY, int padd, int stride, int rowStartOffset, int height, int width, int xseg, int yseg, int n1, int n2, int warpbatch, int curSMStream, int nextSMStream, int* inter_stream_dep, int inter_stream_dep_size, int tileT, int timepiece, int batchStartAddress, volatile int* dev_row_lock){ 
//We assume row size n1 is the multiple of 32 and can be completely divided by tileX.
//For each row, the first tile and the last tile are computed separately from the other tiles.
//size of the shared memory is determined by the GPU architecture.
//tileX is multiple times of 32 to maximize the cache read.		
#ifdef DEBUG
	if (threadIdx.x == 0){
		printf("This is curBatch: %d, timepiece: %d, curStream: %d\n", curBatch, timepiece, curSMStream);
	}
	__syncthreads();
#endif
	//need two arrays: 1. tile raw data; 2. intra-stream dependence
	//tile size stored in the shared array could be as large as 64 * 64
	__shared__ int tile1[5120];
	__shared__ int tile2[5120];
	__shared__ int intra_dep[2047];

	int dep_stride = stride + 1;
	int segLengthX = tileX + dep_stride;
	int segLengthY = tileY + dep_stride;
	int tileIdx = 0;
	int xidx, yidx;
	int tilePos, newtilePos, glbPos;
	int tileAddress;
	int YoverX = tileY/tileX;	
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
		tileAddress = batchStartAddress + tileIdx * tileX;
		read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
		moveMatrixToTile(dev_arr, &tile1[0], segLengthX, tileX, tileY, dep_stride, tileAddress, width, warpbatch);
		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTileEdge(dev_arr, &tile1[0], stride, height, segLengthX, dep_stride, tt, padd, n1, tileY, 0);
			//parameter offset == 0
			moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, n2, segLengthX, padd, height, tileIdx, tt, tileX, 0);
#ifdef PRINT_MATRIX
		if (threadIdx.x == 0){
			printf("curSMStream: %d, curBatch: %d, tileId: %d, timepiece: %d\n", curSMStream, curBatch, tileIdx, timepiece);
			printSharedTile(&tile1[0], segLengthX, tileX, tileY, dep_stride, curSMStream);
		}
#endif	
			for (int tid = threadIdx.x; tid < tileX * tileY; tid += blockDim.x){
				//out-of-range results should be ignored
				//because of the bias, xidx and yidx are the pos of new time elements.
				//thread % tileX and thread / tileX are pos of current cached elements.
				xidx = tid % tileX;
				yidx = tid / tileX;
			        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1.
				tilePos = (dep_stride-1) * segLengthX + (dep_stride - 1) + yidx * segLengthX + xidx;	
				//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array.
				//NEED MODIFICATION BECAUSE newtilePos is not correct here because of the irregular tile size.
				newtilePos = tilePos;
				//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
				//Because the edge elements use only the out-of-range elements as dependent data, we need specific manipulation.
				if (xidx > 0 && xidx < tileX-tt && yidx > 0 && yidx < tileY-tt)
					tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[tilePos-segLengthX]) / 5;
#ifdef PRINT_MATRIX
		if (threadIdx.x == 0){
			printf("tile2\n");
			printSharedTile(&tile2[0], segLengthX, tileX, tileY, dep_stride, curSMStream);
		}
#endif	
			}	
			__syncthreads();
			
			//Since the tile size is reduced along the calculation, the intraDep elements (in last two column of the valid tile) is also shifted to left.
			//Set variable isRegular == 1, when there is a size reduction. 
			moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, 1, tileY);
			//first tile has to copy the out-of-range elements, which are on the left-hand side, to next stream's inter_stream_dep array
			//moveTileToInterDepEdge(dev_arr, inter_stream_dep, tt, tileX, tileY, tileT, nextSMStream, dep_stride, n1, tileIdx, width, curBatch, padd);
			moveTileToInterDepEdge(&tile1[0], inter_stream_dep, tt, tileX, tileY, tileT, nextSMStream, dep_stride, segLengthX, tileIdx, n1);
			//variable isRegular == 1, because one row is shifted out-side-of the upper boundary
			//variable len == tileX-tt because this tile is not in a regular size.
			moveTileToInterDep(&inter_stream_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, nextSMStream, tileT, n1, segLengthX, tileIdx, tileX-tt, 1);
			//swap tile2 with tile1;
			for (int tid = threadIdx.x; tid < 5120; tid+=blockDim.x){
				tile1[tid] = tile2[tid];
				tile2[tid] = 0;
			}
			__syncthreads();
#ifdef PRINT_MATRIX
		if (threadIdx.x == 0){
			printf("tile1 after calculation: \n");
			printSharedTile(&tile1[0], segLengthX, tileX, tileY, dep_stride, curSMStream);
		}
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
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileT, tileT, tileX, tileY, dep_stride, height, segLengthX);	
#ifdef PRINT_MATRIX
		printGlobal(dev_arr, width, height, curSMStream);
#endif
		__syncthreads();
		write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);

		//tile = 1 to xseg-1; regular size tiles, with index shifting.
		for (tileIdx = 1; tileIdx < xseg-1; tileIdx++){
			tileAddress = batchStartAddress + tileIdx * tileX;
			read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
			//copy the base spatial data to shared memory for t=0.
			moveMatrixToTile(dev_arr, &tile1[0], segLengthX, tileX, tileY, dep_stride, tileAddress, width, warpbatch);
			for (int tt=0; tt<tileT; tt++){
				moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, tileY);
				moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, n2, segLengthX, padd, width, tileIdx, tt, tileX, 0);
#ifdef PRINT_MATRIX
		if (threadIdx.x == 0){
			printf("curSMStream: %d, curBatch: %d, tileId: %d, timepiece: %d\n", curSMStream, curBatch, tileIdx, timepiece);
			printSharedTile(&tile1[0], segLengthX, tileX, tileY, dep_stride, curSMStream);
		}
#endif	
/*				for (int tid = threadIdx.x; tid < tileX * tileY; tid += blockDim.x){
					//out-of-range results should be ignored
					//because of the bias, xidx and yidx are the pos of new time elements.
					//thread % tileX and thread / tileX are pos of current cached elements.
					xidx = tid % tileX;
					yidx = tid / tileX;
				        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1
					tilePos = (dep_stride-1) * segLengthX + (dep_stride - 1) + yidx * segLengthX + xidx;	
					//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array
					//newtilePos = dep_stride * segLengthX + dep_stride + yidx * segLengthX + xidx;
					newtilePos = tilePos;
					//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
					//Because, the edge elements use only the out-of-range elements as dependent data, we need specific manipulation.
					if (yidx > 0 && yidx < tileX-tt)
						tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[tilePos-segLengthX]) / 5;
				}	
				__syncthreads();
				//Set variable isRegular == 0 to disable the tile size reduction, when tile size are constant during the calculation. 
				moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, 0, tileY);
				//variable isRegular == 1 because one row is shifted out-side-of the upper boundary.
				//variable len == tileX-tt because row is shifted out-side-of the upper boundary
				moveTileToInterDep(&inter_stream_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, nextSMStream, tileT, n1, segLengthX, tileIdx, tileX-tt, 1);
				//swap tile2 with tile1;
				for (int tid = threadIdx.x; tid < 5120; tid+=blockDim.x){
					tile1[tid] = tile2[tid];
					tile2[tid] = 0;
				}
				__syncthreads();
*/			}						 
/*			//glbPos is the index where the calculated elements should be stored at in the global matrix array.
			//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
			//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
			//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
			//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
			//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
			//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
			//ignLenX == 0 because all elements are copied at each row, ignLenY == tileT because tileY-tileT elements are copied at each column.
			glbPos = tileAddress;	
			moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, 0, tileT, tileX, tileY, dep_stride, height, segLengthX);	
			__syncthreads();
*/	
			write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);
		}

		//when tile = xseg-1, if matrix is completely divided by the tile, no t0 elements copy to shared memory; 
		//use dependent data and out-of-range data to calculate.
		tileIdx = xseg-1;
		//unlike the other two cases that tileAddress points to the source pos of t0, here tileAddress is the destination pos of t(tileT-1).
		tileAddress = batchStartAddress + tileIdx * tileX - tileT; //might be "timepiece" instead of "tileT"
		read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
/*		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, tileY);
			//set variable offset == 1 if it is the last tile of each batch to copy right-side out-of-range elements to 
			moveIntraDepToTileEdge(dev_arr, &tile1[0], stride, height, segLengthX, dep_stride, tt, padd, n1, tileY, 1);
			moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, n2, segLengthX, padd, height, tileIdx, tt, tt + dep_stride);
			//tileX of the last tile is changed throughout the simulation from 0 to tileT;
			for (int tid = threadIdx.x; tid < (tt+1) * tileY; tid += blockDim.x){
				//out-of-range results should be ignored
				//because of the bias, xidx and yidx are the pos of new time elements.
				//thread % tileX and thread / tileX are pos of current cached elements.
				xidx = tid % tileX;
				yidx = tid / tileX;
			        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1.
				tilePos = (dep_stride-1) * segLengthX + (dep_stride - 1) + yidx * segLengthX + xidx;	
				//newtilePos starts one row above the tile matrix because the next tile is shifted out-side-of the up boundary
				newtilePos = (dep_stride-1) * segLengthX + dep_stride + yidx * segLengthX + xidx;
				//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
				//Because, the edge elements use only the out-of-range elements as dependent data, we need specific manipulation
				if (xidx <= tt && yidx > 0 && yidx < tileY-tt)
					tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[tilePos-segLengthX]) / 5;
			}	
			__syncthreads();
			
			//variable isRegular == 1 because one row is shifted out-side-of the upper boundary.
			//len = tileX-1-tt, variable len specifies the lenth of eligible elements should be moved to inter_stream_dep[].
			moveTileToInterDep(&inter_stream_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, nextSMStream, tileT, n1, segLengthX, tileIdx, tileX-tt-1, 1);
			//swap tile2 with tile1;
			for (int tid = threadIdx.x; tid < 5120; tid+=blockDim.x){
				tile1[tid] = tile2[tid];
				tile2[tid] = 0;
			}
			__syncthreads();
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
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileX-tileT, tileT, tileX, tileY, dep_stride, height, segLengthX);	
		__syncthreads();
*/
		write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);

	}
	else if(curBatch == yseg - 1){
	//for the last batch, all the tiles are irregular
		//when tile = 0, the calculated data which are outside the range are not copied to tile2, tile size is shrinking 
		//along T dimension. Out-of-range elements are used for dependent data.
		read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
/*		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTileEdge(dev_arr, &tile1[0], stride, height, segLengthX, dep_stride, tt, padd, n1, tt, 1);
			//the first tile is not in regular size, so variable len = tileX-tt
			moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, tileY, dep_stride, curSMStream, tileT, n1, segLengthX, tileIdx, tileX-tt);
			//move out-of-range elements which are beanth the bottom boundary to the tile
			//variable offset == 1, used to locate the bottom out-of-boundary elements.
			moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, n2, segLengthX, padd, height, tileIdx, tt, tileX, 1);
			
			for (int tid = threadIdx.x; tid < tileX * tileY; tid += blockDim.x){
				//out-of-range results should be ignored
				//because of the bias, xidx and yidx are the pos of new time elements.
				//thread % tileX and thread / tileX are pos of current cached elements.
				xidx = tid % tileX;
				yidx = tid / tileX;
			        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1.
				tilePos = (dep_stride-1) * segLengthX + (dep_stride - 1) + yidx * segLengthX + xidx;	
				//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array.
				//left column shift out-side-of the boundary, so retain all rows but discard the left-most column.
				newtilePos = dep_stride * segLengthX + (dep_stride - 1) + yidx * segLengthX + xidx;
				//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
				//Because the edge elements use only the out-of-range elements as dependent data, we need specific manipulation.
				if (xidx>0 && xidx < tileX-tt && yidx <= tt)
					tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[tilePos-segLengthX]) / 5;
			}	
			__syncthreads();
			
			//Since the tile size is reduced along the calculation, the intraDep elements (in last two column of the valid tile) is also shifted to left.
			//Set variable isRegular == 1, when there is a size reduction. 
			moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, 1, tt);
			//swap tile2 with tile1;
			for (int tid = threadIdx.x; tid < 5120; tid+=blockDim.x){
				tile1[tid] = tile2[tid];
				tile2[tid] = 0;
			}
			__syncthreads();
		}
		//glbPos is the index where the calculated elements should be stored at in the global matrix array.
		//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
		//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
		//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
		//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
		//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
		//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
		glbPos = batchStartAddress + tileIdx * tileX;
		//ignLenX == tileT because tileX-tileT elements are copied at each row, ignLenY == tileY-tileT because tileT elements are copied at each column.
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileT, tileY-tileT, tileX, tileY, dep_stride, height, segLengthX);	
		__syncthreads();
*/
//		write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);

		//tile = 1 to xseg-1; regular size tiles, with index shifting.
		for (tileIdx = 1; tileIdx < xseg-1; tileIdx++){
			read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
/*			for (int tt=0; tt<tileT; tt++){
				//isRegular == 0 because this is a regular tile.
//				moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, 0, tt);
				moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, tt);
				moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, tileY, dep_stride, curSMStream, tileT, n1, segLengthX, tileIdx, tileX);
				//move out-of-range elements which are beanth the bottom boundary to the tile
				//variable offset == 1, used to locate the bottom out-of-boundary elements.
				moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, n2, segLengthX, padd, height, tileIdx, tt, tileX, 1);
				for (int tid = threadIdx.x; tid < tileX * tileY; tid += blockDim.x){
					//out-of-range results should be ignored
					//because of the bias, xidx and yidx are the pos of new time elements.
					//thread % tileX and thread / tileX are pos of current cached elements.
					xidx = tid % tileX;
					yidx = tid / tileX;
				        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1
					tilePos = (dep_stride-1) * segLengthX + (dep_stride - 1) + yidx * segLengthX + xidx;	
					//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array
					newtilePos = dep_stride * segLengthX + dep_stride + yidx * segLengthX + xidx;
					//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
					//Because, the edge elements use only the out-of-range elements as dependent data, we need specific manipulation.
					if (yidx <= tt)
						tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[tilePos-segLengthX]) / 5;
				}	
				__syncthreads();
				//isRegular == 0 to disable the tile size reduction, when tile size are constant during the calculation. 
				moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, 0, tt);
				//variable len == tileX because the tile size is constant.
				moveTileToInterDep(&inter_stream_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, nextSMStream, tileT, n1, segLengthX, tileIdx, tileX, 0);
				//swap tile2 with tile1;
				for (int tid = threadIdx.x; tid < 5120; tid+=blockDim.x){
					tile1[tid] = tile2[tid];
					tile2[tid] = 0;
				}
				__syncthreads();
			}						 
			//glbPos is the index where the calculated elements should be stored at in the global matrix array.
			//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
			//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
			//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
			//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
			//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
			//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
			glbPos = batchStartAddress + tileIdx * tileX;
			//ignLenX == 0 because all elements are copied at each row, ignLenY == tileY-tileT because tileT elements are copied at each column.
			moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, 0, tileY-tileT, tileX, tileY, dep_stride, height, segLengthX);	
			__syncthreads();
			
*/
//			write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);
		}

		//when tile = xseg-1, if matrix is completely divided by the tile, no t0 elements copy to shared memory; 
		//use dependent data and out-of-range data to calculate.
		tileIdx = xseg-1;
		read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
/*		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, tt);
			//set variable offset == 1 if it is the last tile of each batch to copy right-side out-of-range elements to 
			moveIntraDepToTileEdge(dev_arr, &tile1[0], stride, height, segLengthX, dep_stride, tt, padd, n1, tt, 1);
			
			//1. inter_stream_dep elements from previous tile (on top of intra_dep elements); total size == len + dev_stride, where len == tt, which is 0 at t0
			//2. out-of-range elements
			//copy edge elements first to cover the out-of-range elements, then copy the inter_stream_dep of previous stream and cover a part of the out-of-range elements.
			//variable len == tt + dev_stride, which covers the size of the elements, calculated in previous stream, and the out-of-range elements.
			moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, n2, segLengthX, padd, height, tileIdx, tt, tt + dep_stride);
			moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, tileY, dep_stride, curSMStream, tileT, n1, segLengthX, tileIdx, tt);
			//move out-of-range elements which are beanth the bottom boundary to the tile
			//variable offset == 1, used to locate the bottom out-of-boundary elements.
			moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, n2, segLengthX, padd, height, tileIdx, tt, tt + dep_stride, 1);
			//tileX of the last tile is changed throughout the simulation from 0 to tileT;
			for (int tid = threadIdx.x; tid < (tt+1) * tileY; tid += blockDim.x){
				//out-of-range results should be ignored
				//because of the bias, xidx and yidx are the pos of new time elements.
				//thread % tileX and thread / tileX are pos of current cached elements.
				xidx = tid % tileX;
				yidx = tid / tileX;
			        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1.
				tilePos = (dep_stride-1) * segLengthX + (dep_stride - 1) + yidx * segLengthX + xidx;	
				//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array.
				newtilePos = dep_stride * segLengthX + dep_stride + yidx * segLengthX + xidx;
				//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
				//Because, the edge elements use only the out-of-range elements as dependent data, we need specific manipulation
				if (xidx <= tt && yidx <= tt)
					tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[tilePos-segLengthX]) / 5;
			}	
			__syncthreads();
			
			//swap tile2 with tile1;
			for (int tid = threadIdx.x; tid < 5120; tid+=blockDim.x){
				tile1[tid] = tile2[tid];
				tile2[tid] = 0;
			}
			__syncthreads();
		}
		//glbPos is the index where the calculated elements should be stored at in the global matrix array.
		//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
		//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
		//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
		//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
		//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
		//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
		//unlike the other two cases that glbPos points to the source pos of t0, here tileAddress is the destination pos of t(tileT-1).
		glbPos = batchStartAddress + tileIdx * tileX - tileT;	
		//ignLenX == tileX-tileT because tileT elements are copied at each row, ignLenY == tileY-tileT because tileT elements are copied at each column.
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileX-tileT, tileY-tileT, tileX, tileY, dep_stride, height, segLengthX);	
		__syncthreads();
*/	
//		write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);
	
	}
	else{
	//for the regular batch, use the near-edge elements for the out-of-range dependence of first and last tile only.
		//when tile = 0, the calculated data which are outside the range are not copied to tile2, tile size is shrinking 
		//along T dimension. Out-of-range elements are used for dependent data.
		tileAddress = batchStartAddress + tileIdx * tileX;
		read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
/*		moveMatrixToTile(dev_arr, &tile1[0], segLengthX, tileX, tileY, dep_stride, tileAddress, width, warpbatch);
		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTileEdge(dev_arr, &tile1[0], stride, height, segLengthX, dep_stride, tt, padd, n1, tileY, 0);
			//the first tile is not in regular size, so variable len = tileX-tt
			moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, tileY, dep_stride, curSMStream, tileT, n1, segLengthX, tileIdx, tileX-tt);
			for (int tid = threadIdx.x; tid < tileX * tileY; tid += blockDim.x){
				//out-of-range results should be ignored
				//because of the bias, xidx and yidx are the pos of new time elements.
				//thread % tileX and thread / tileX are pos of current cached elements.
				xidx = tid % tileX;
				yidx = tid / tileX;
			        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1.
				tilePos = (dep_stride-1) * segLengthX + (dep_stride - 1) + yidx * segLengthX + xidx;	
				//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array.
				//left column shift out-side-of the boundary, so retain all rows but discard the left-most column.
				newtilePos = dep_stride * segLengthX + (dep_stride - 1) + yidx * segLengthX + xidx;
				//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
				//Because the edge elements use only the out-of-range elements as dependent data, we need specific manipulation.
				if (xidx > 0 && xidx < tileX-tt)
					tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[tilePos-segLengthX]) / 5;
			}	
			__syncthreads();
			
			//Since the tile size is reduced along the calculation, the intraDep elements (in last two column of the valid tile) is also shifted to left.
			//Set variable isRegular == 1, when there is a size reduction. 
			moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, 1, tileY);
			//first tile has to copy the out-of-range elements, which are on the left-hand side, to next stream's inter_stream_dep array
			moveTileToInterDepEdge(dev_arr, inter_stream_dep, tt, tileX, tileY, tileT, nextSMStream, dep_stride, n1, tileIdx, height, curBatch, padd);
			//variable len == tileX-tt because the tile size is reduced during calculation.
			//isRegular == 0 because there is no row move out-side-of upper boundary
			moveTileToInterDep(&inter_stream_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, nextSMStream, tileT, n1, segLengthX, tileIdx, tileX-tt, 0);
			//swap tile2 with tile1;
			for (int tid = threadIdx.x; tid < 5120; tid+=blockDim.x){
				tile1[tid] = tile2[tid];
				tile2[tid] = 0;
			}
			__syncthreads();
		}
		//glbPos is the index where the calculated elements should be stored at in the global matrix array.
		//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
		//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
		//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
		//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
		//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
		//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
		//ignLenX == tileT because tileX-tileT elements are copied at each row, ignLenY == 0 because no size reduction along Y dim.
		glbPos = tileAddress;	
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileT, 0, tileX, tileY, dep_stride, height, segLengthX);	
		__syncthreads();
*/		write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);

		//tile = 1 to xseg-1; regular size tiles, with index shifting.
		for (tileIdx = 1; tileIdx < xseg-1; tileIdx++){
			tileAddress = batchStartAddress + tileIdx * tileX;
			read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
/*			//copy the base spatial data to shared memory for t=0.
			moveMatrixToTile(dev_arr, &tile1[0], segLengthX, tileX, tileY, dep_stride, tileAddress, width, warpbatch);
			for (int tt=0; tt<tileT; tt++){
				//isRegular == 0 because this is a regular tile.
//				moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, 0, tileY);
				moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, tileY);
				moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, tileY, dep_stride, curSMStream, tileT, n1, segLengthX, tileIdx, tileX);
				for (int tid = threadIdx.x; tid < tileX * tileY; tid += blockDim.x){
					//out-of-range results should be ignored
					//because of the bias, xidx and yidx are the pos of new time elements.
					//thread % tileX and thread / tileX are pos of current cached elements.
					xidx = tid % tileX;
					yidx = tid / tileX;
				        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1
					tilePos = (dep_stride-1) * segLengthX + (dep_stride - 1) + yidx * segLengthX + xidx;	
					//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array
					newtilePos = dep_stride * segLengthX + dep_stride + yidx * segLengthX + xidx;
					tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[tilePos-segLengthX]) / 5;
				}	
				__syncthreads();
				//isRegular == 0 to disable the tile size reduction, when tile size are constant during the calculation. 
				moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, 0, tileY);
				//variable len == tileX because the tile size is constant.
				moveTileToInterDep(&inter_stream_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, nextSMStream, tileT, n1, segLengthX, tileIdx, tileX, 0);
				//swap tile2 with tile1;
				for (int tid = threadIdx.x; tid < 5120; tid+=blockDim.x){
					tile1[tid] = tile2[tid];
					tile2[tid] = 0;
				}
				__syncthreads();
			}						 
			//glbPos is the index where the calculated elements should be stored at in the global matrix array.
			//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
			//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
			//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
			//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
			//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
			//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
			glbPos = tileAddress;	
			moveShareToGlobal(&tile1[0], dev_arr, glbPos, tileX, tileY, dep_stride, height, segLengthX);	
			__syncthreads();
*/			
			write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);
		}

		//when tile = xseg-1, if matrix is completely divided by the tile, no t0 elements copy to shared memory; 
		//use dependent data and out-of-range data to calculate.
		tileIdx = xseg-1;
		//unlike the other two cases that tileAddress points to the source pos of t0, here tileAddress is the destination pos of t(tileT-1).
		tileAddress = batchStartAddress + tileIdx * tileX - tileT;
		read_tile_lock_for_batch(dev_row_lock, curBatch, tileIdx, YoverX, xseg, yseg, timepiece);
/*		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, tileY);
			//set variable offset == 1 if it is the last tile of each batch to copy right-side out-of-range elements to 
			moveIntraDepToTileEdge(dev_arr, &tile1[0], stride, height, segLengthX, dep_stride, tt, padd, n1, tileY, 1);
			
			//1. inter_stream_dep elements from previous tile (on top of intra_dep elements); total size == len + dev_stride, where len == tt, which is 0 at t0
			//2. out-of-range elements
			//copy edge elements first to cover the out-of-range elements, then copy the inter_stream_dep of previous stream and cover a part of the out-of-range elements.
			//variable len == tt + dev_stride, which covers the size of the elements, calculated in previous stream, and the out-of-range elements.
			moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, n2, segLengthX, padd, height, tileIdx, tt, tt + dep_stride);
			moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, tileY, dep_stride, curSMStream, tileT, n1, segLengthX, tileIdx, tt);
			//tileX of the last tile is changed throughout the simulation from 0 to tileT;
			for (int tid = threadIdx.x; tid < (tt+1) * tileY; tid += blockDim.x){
				//out-of-range results should be ignored
				//because of the bias, xidx and yidx are the pos of new time elements.
				//thread % tileX and thread / tileX are pos of current cached elements.
				xidx = tid % tileX;
				yidx = tid / tileX;
			        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1.
				tilePos = (dep_stride-1) * segLengthX + (dep_stride - 1) + yidx * segLengthX + xidx;	
				//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array.
				newtilePos = dep_stride * segLengthX + dep_stride + yidx * segLengthX + xidx;
				//when curBatch == 0, eligible tile size is reduced along the timestamp because of the shifting.
				//Because, the edge elements use only the out-of-range elements as dependent data, we need specific manipulation
				if (xidx <= tt)
					tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[tilePos-segLengthX]) / 5;
			}	
			__syncthreads();
			
			//variable isRegular == 0 because one row is shifted out-side-of the upper boundary.
			//len = tileX-1-tt, variable len specifies the lenth of eligible elements should be moved to inter_stream_dep[].
			moveTileToInterDep(&inter_stream_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, nextSMStream, tileT, n1, segLengthX, tileIdx, tileX-tt-1, 0);
			//swap tile2 with tile1;
			}
			__syncthreads();
		}
		//glbPos is the index where the calculated elements should be stored at in the global matrix array.
		//when curBatch == 0 && tileIdx == 0, glbPos always start from the first eligible element of the tile, which is tileAddress
		//and then ignore the out-of-range elements by using ignLenX and ignLenY variables.
		//when curBatch == 0 or tile idx == 0, the out of range elements should be ignored, ignLenX and ignLenY are set accordingly.
		//curBatch > 0 && tileIdx == 0, glbPos is shifted up by tileT unit from tileAddress.
		//curBatch == 0 && tileIdx > 0, glbPos is shifted left by tileT unit from tileAddress.
		//when curBatch > 0 && tileIdx > 0, glbPos is shifted up and left by tileT unit from tileAddress, complete tile is moved,
		//ignLenX == tileX-tileT because only tileT elements are copied in each row, ignLenY == 0 because no size reduction along Y dim.
		glbPos = tileAddress;	
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileX-tileT, 0, tileX, tileY, dep_stride, height, segLengthX);	
		__syncthreads();
*/
		write_tile_lock_for_batch(dev_row_lock, curBatch, yseg, timepiece);
	}

//	write_batch_lock_for_time(timepiece, curBatch);
}


void checkGPUError(cudaError err){
	if (cudaSuccess != err){
		printf("CUDA error in file %s, in line %i: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void SOR(int n1, int n2, int padd, int *arr, int MAXTRIAL){
	cudaSetDevice(0);	
//stride is the longest distance between the element and its dependence along one dimension times
//For example: F(x) = T(x-1) + T(x) + T(x+1), stride = 1
//when we change stride, we also need to update parameter "paddsize" in data generator file.
	int stride = 1;
	int dep_stride = stride+1;
	int tileX = 4;
	int tileY = 4;
	int rawElmPerTile = tileX * tileY;
	int tileT = 1;
	int numStream = 28;

//PTilesPerTimestamp is the number of parallelgoram tiles can be scheduled at each time stamp
//	int PTilesPerTimestamp = (n1/tileX) * (n2/tileY); 
//ZTilesPerTimestamp is the number of trapezoid tiles (overlaped tiles) needed to calculate the uncovered area at each time stamp.
//	int ZTilesPerTimestamp = (n1/tileX) + (n2/tileY) - 1; 
	int height = 2 * padd + n1; 
	int width = 2 * padd + n2;

	volatile int *dev_arr;
	int *lock;
	size_t freeMem, totalMem;
	volatile int *dev_time_lock, *dev_row_lock;	
	
	cudaMemGetInfo(&freeMem, &totalMem);
	int tablesize = width * height;
#ifdef DEBUG
	cout << "current GPU memory info FREE: " << freeMem << " Bytes, Total: " << totalMem << " Bytes." << endl;
	cout << "width: " << colsize << ", height: " << rowsize << ", allocates: " << tablesize * sizeof(int)<< " Bytes." << endl;
#endif
	cudaError err = cudaMalloc(&dev_arr, tablesize * sizeof(int));
	checkGPUError(err);
	
//	cudaMalloc(&dev_time_lock, n2/tileY * sizeof(int));
	err = cudaMemcpy((void*)dev_arr, arr, tablesize*sizeof(int), cudaMemcpyHostToDevice);
	checkGPUError(err);
//	cudaMemset((void*)dev_time_lock, 1, n2/tileY * sizeof(int));

//	int threadPerBlock = min(MAX_THREADS_PER_BLOCK, rawElmPerTile);
	int threadPerBlock = MAX_THREADS_PER_BLOCK;
//	int blockPerGrid = PTilesPerTimestamp;
	int blockPerGrid = 1;
	int warpbatch = MAX_THREADS_PER_BLOCK / 32;

//memory structure: stream --> tile --> time --> dependence --> tileX
	int *dev_inter_stream_dependence;
	int stream_dep_offset = tileT * (n1 + dep_stride) * dep_stride;
	int inter_stream_dependence = numStream * stream_dep_offset;
	err = cudaMalloc(&dev_inter_stream_dependence, inter_stream_dependence * sizeof(int));
	checkGPUError(err);

	int xseg = n1 / tileX + 1;
	int yseg = n2 / tileY + 1;
	int tseg = (MAXTRIAL + tileT - 1) / tileT;
	int stream_offset = yseg % numStream;
	
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
	cudaStream_t stream[numStream];
	for (int s=0; s<numStream; s++)
		cudaStreamCreate(&stream[s]);

//print initial matrix
#ifdef DEBUG
	for (int row = 0; row < height; row++){
		for (int col = 0; col < width; col++){
			cout << arr[row * width + col] << " ";	
		}
		cout << endl;
	}
#endif


//t < MAXTRIAL? or t <= MAXTRIAL	
	for(int t = 0; t < MAXTRIAL; t+= tileT){
//GPU_ZTile() is the kernel function to calculate the update result, unconvered by Parallelgoram tiling.
//These data are calculated with trapezoid tiling, thus they can be launched concurrently.
// ZTile and cudaDeviceSynchronize() will stop theparallelism along the temporal dimension and force
//the beginning of the new t tiles has to wait the completion of the previous t tiles.
//		GPU_ZTile<<<>>>();
//		cudaDeviceSynchronize();		
		for(int curBatch = 0; curBatch < yseg; curBatch++){
//Have to change the stream Index so that the stream for next time tile can start without waiting for the 
//completion of the previous time tile. 
//Example: stream 0, 1, 2 are scheduled to the last three batches in one time tile, since the execution on
//the next time tile also starts from stream 0, this new execution in stream 0 has to wait for the previous
			int timepiece = t / tileT;
			int logicSMStream = curBatch % numStream;
			int curSMStream = (logicSMStream +  stream_offset * timepiece) % numStream;
			int curStartAddress = curBatch * tileY * height;
			int rowStartOffset = padd * height + padd;
			int batchStartAddress = rowStartOffset + curStartAddress;
			int nextSMStream = (curSMStream + 1) % numStream;
//			cout << "curBatch: " << curBatch << ", stride: " << stride << ", tileX: " << tileX << ", tileY: " << tileY << ", t: " << t << ", xseg: " << xseg << ", yseg: " << yseg << ", logicStream: " << logicSMStream << ", curStream: " << curSMStream  << endl;	
			GPU_Tile<<<blockPerGrid, threadPerBlock, 0, stream[curSMStream]>>>(dev_arr, curBatch, curStartAddress, tileX, tileY,  padd, stride, rowStartOffset, height, width, xseg, yseg, n1, n2, warpbatch, curSMStream, nextSMStream, dev_inter_stream_dependence, inter_stream_dependence, tileT, timepiece, batchStartAddress, dev_row_lock);	
			checkGPUError( cudaGetLastError() );
		}
		//this global synchronization enforces the sequential computation along t dimension.
//		cudaDeviceSynchronize();
	}	
//cudaMemcpy(table, (void*)dev_table, tablesize*sizeof(int), cudaMemcpyDeviceToHost);

	for (int s=0; s<numStream; s++)
		cudaStreamDestroy(stream[s]);
	
	cudaFree((void*)dev_arr);
	cudaFree((void*)dev_row_lock);
	cudaFree((void*)dev_inter_stream_dependence);
	delete[] lock;

}

