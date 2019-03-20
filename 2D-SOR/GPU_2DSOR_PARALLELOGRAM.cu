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

__device__ void moveMatrixToTile(int *dev_arr, int *tile, int segLengthX, int tileX, int tileY, int dep_stride, int tileAddress, int rowsize, int warpbatch, int thread){
	int idx = thread % 32;
	int warpidx = thread / 32;
	int glbpos = tileAddress + warpidx * rowsize + idx;
	int shrpos = dep_stride * segLength + warpidx * segLengthX + dep_stride + idx;
//	if (thread < segLengthX)
//		table[thread] = dev_table[tileAddress + thread];
	for (; warpidx < tileY; warpidx += warpbatch){
		for (int i = idx; i < tileX; i += 32){
			tile[shrpos+i] = dev_arr[glbpos+i];
		}
		shrpos += (warpbatch * segLengthX);
		glbpos += (warpbatch * rowsize);
	}
}

//intra_dep array structure: tileT * dep_stride * tileY
__device__ void moveIntraDepToTile(int* intra_dep, int* tile, int tt, int tileY, int segLengthX, int dep_stride, int thread, int len){
	//at each tt, (stride+1) dependent data are required at x axis.
	//only the threads, which are within tileY are working here.
	//threadPerBlock has to be no less than tileY * dep_stride
	if (thread < len * dep_stride){
		int pos = tt * dep_stride * tileY + thread;
		int tilepos = dep_stride * segLengthX + thread/dep_stride * segLengthX + thread % dep_stride;
		tile[tilepos] = intra_dep[pos];
	}
}

__device__ void moveIntraDepToTileEdge(int* dev_arr, int* tile, int stride, int rowsize, int segLengthX, int dep_stride, int thread, int tt, int padd, int n1, int offset=0, int len){
	//copy out-of-range data to tile
	if (thread < len * dep_stride){
		int glbpos = padd * rowsize + (padd - dep_stride) + offset * (n1 + dep_stride) + thread/dep_stride * rowsize + thread % dep_stride;
		int tilepos = dep_stride * segLengthX + thread/dep_stride * segLengthX + thread % dep_stride + offset * (dep_stride + tt);
		tile[tilepos] = dev_arr[glbpos];
	}
}

__device__ void moveTileToIntraDep(int* intra_dep, int* tile, int tt, int tileX, int tileY, int segLengthX, int dep_stride, int thread, int isRegular, int len){
	if (thread < len * dep_stride){
		int pos = tt * dep_stride * tileY + thread;
		int tilepos = dep_stride * segLengthX + tileX - tt * isRegular;
	       	tilepos	+= thread/dep_stride * segLengthX + thread % dep_stride;
		intra_dep[pos] = tile[tilepos];
	}
}

//inter_dep array structure: stream * tileT * dep_stride * (n1 + dep_stride)
__device__ void moveInterDepToTile(int* inter_dep, int* tile, int tt, int tileX, int tileY, int dep_stride, int thread, int stream, int tileT, int n1, int segLengthX, int tileIdx, int len){
	int startAddress = (stream * tileT + tt) * dep_stride * (n1 + dep_stride);
	if (tileIdx > 0)       
		startAddress += ( (tileIdx-1) * tileX + tileX-tt );
	startAddress += ( tileIdx * tileX);
	//variable len specifies the eligible elements should be moved. This is caused by the irregular tile.
	if (thread < len + dep_stride){
		int pos = startAddress + thread;
		int tilepos = thread;
		for (int i=0; i<dep_stride; i++){
	 		tile[tilepos] = inter_dep[pos];
			pos += (n1 + dep_stride);
			tilepos += segLengthX;
		}
	}	
}

__device__ void moveInterDepToTileEdge(int* dev_arr, int* tile, int tileX, int tileY, int dep_stride, int thread, int n2, int segLengthX, int padd, int rowsize, int tileIdx, int tt, int len, int offset = 0){
	int glbpos = (padd - dev_stride) * rowsize + offset * (dev_stride + n2) * rowsize + padd - dep_stride + thread;
	if (tileIdx > 0)
		glbpos += ((tileIdx-1) * tileX + tileX-tt);
	if (thread < len + dep_stride){
		int tilepos = offset * (dev_stride + len) * segLengthX + thread;
		for (int i=0; i<dep_stride; i++){
			tile[tilepos] = dev_arr[glbpos];
			tilepos += segLengthX;
			pos += rowsize;
		}
	}
}

__device__ void moveTileToInterDep(int* inter_dep, int* tile, int tt, int tileX, int tileY, int dep_stride, int thread, int nextstream, int tileT, int n1, int segLengthX, int tileIdx, int len, int isRegular){
	int startAddress = dep_stride + (nextstream * tileT + tt) * dep_stride * (n1 + dep_stride);
	//for the edge tiles, the size is irregular so that the start position of some tt timestamp are not times of tileX.
	if (tileIdx > 0)       
		startAddress += ( (tileIdx-1) * tileX + tileX-tt );
	//variable len specifies the eligible elements should be moved. This is caused by the irregular tile.
	if (thread < len){
		int pos = startAddress + thread;
		int tilepos = dep_stride + (tileY - (tt+1) * isRegular) * segLengthX + thread;
		for (int i=0; i<dep_stride; i++){
	 		inter_dep[pos] = tile[tilepos];
			pos += (n1 + dep_stride);
			tilepos += segLengthX;
		}
	}	
}

__device__ void moveTileToInterDepEdge(int* dev_arr, int* inter_dep, int tt, int tileX, int tileY, int tileT, int nextstream, int dep_stride, int n1, int tileIdx, int rowsize, int curBatch, int padd){
	int startAddress = (nextstream * tileT + tt) * dep_stride * (n1 + dep_stride);
	int glbpos = padd * rowsize + curBatch * tileY * rowsize + (padd - dep_stride) + (tileY - dep_stride) * rowsize;
	if (thread < dep_stride){
		int interpos = startAddress + thread;
		int pos = glbpos + thread;
		for (int i=0; i<dep_stride; i++){
	 		inter_dep[interpos] = dev_arr[pos];
			pos += rowsize;
			interpos += (n1 + dep_stride);
		}
	}	
}

__device__ void moveShareToGlobalEdge(int* tile, int* dev_arr, int startPos, int ignLenX, int ignLenY, int tileX, int tileY, int dep_stride, int rowsize, int segLengthX, int thread){
	int xidx, yidx, glbPos, tilePos;
	for (int tid = thread; tid < tileX * tileY; tid += blockDim.x){
		xidx = tid % tileX;
		yidx = tid / tileX;
		if (xidx < tileX - ignLenX && yidx < tileY - ignLenY){
			glbPos = startPos + yidx * rowsize + xidx;
			tilePos = (dep_stride + yidx) * segLengthX + dep_stride + xidx;
			dev_arr[glbPos] = tile[tilePos];
		}
	}	
}	

__device__ void moveShareToGlobal(int* tile, int* dev_arr, int startPos, int tileX, int tileY, int dep_stride, int rowsize, int segLengthX, int thread){
	int xidx, yidx, glbPos, tilePos;
	for (int tid = thread; tid < tileX * tileY; tid += blockDim.x){
		xidx = tid % tileX;
		yidx = tid / tileX;
		glbPos = startPos + yidx * rowsize + xidx;
		tilePos = (dep_stride + yidx) * segLengthX + dep_stride + xidx;
		dev_arr[glbPos] = tile[tilePos];
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

__device__ void read_batch_lock_for_time(){

}

__device__ void read_tile_lock_for_batch(){

}

__device__ void write_batch_lock_for_time(){

}

__device__ void write_tile_lock_for_batch(){

}

__global__ void GPU_Tile(int *dev_arr, volatile int *dev_lock, int curBatch, int curStartAddress, int tileX, int tileY, int padd, int stride, int rowStartOffset, int rowsize, int colsize, int xseg, int yseg, int n1, int n2, int warpbatch, int curSMStream, int nextSMStream, int* inter_stream_dep, int inter_stream_dep_size, int tileT, int timepiece, int batchStartAddress){ 
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
	int dep_stride = stride + 1;
	int segLengthX = tileX + dep_stride;
	int segLengthY = tileY + dep_stride;
	int tileIdx = 0;
	int xidx, yidx;
	int sumStencil;
	int tilePos, newtilePos, glbPos;
	int tileAddress;
//if this is the first batch of the current t tile, have to copy the related dependence data from global tile array into global inter-stream-dependence array.
//Challenges: when stream 0 is still working on one of the current t tiles but stream 2 already starts processing the first batch of the next t tiles. Copying the dependence data to arr[stream[0]] does not work.
//for the first and last batches, we need charactorized function to take care of the edge elements.

//***********************************************************************************************************************************
	read_batch_lock_for_time(timepiece, curBatch);
//processing the first tile of each row, use the near-edge elements for the out-of-range dependence.
	//wait until it is safe to launch and execute the new batch.

	if (curBatch == 0){
	//for the first batch, use the near-edge elements for the out-of-range dependence.
		//when tile = 0, the calculated data which are outside the range are not copied to tile2, tile size is shrinking 
		//along T dimension. Out-of-range elements are used for dependent data.
		tileAddress = batchStartAddress + tileIdx * tileX;
		read_tile_lock_for_batch(timepiece, curBatch, tileIdx);
		moveMatrixToTile(dev_arr, &tile1[0], segLengthX, tileX, tileY, dep_stride, tileAddress, rowsize, warpbatch, thread);
		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTileEdge(dev_arr, &tile1[0], stride, rowsize, segLengthX, dep_stride, thread, tt, padd, n1, 0, tileY);
			moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, thread, n2, segLengthX, padd, rowsize, tileIdx, tt, tileX);
			for (int tid = thread; tid < tileX * tileY; tid += blockDim.x){
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
					tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[idx2-segLengthX]) / 5;
			}	
			__syncthreads();
			
			//Since the tile size is reduced along the calculation, the intraDep elements (in last two column of the valid tile) is also shifted to left.
			//Set variable isRegular == 1, when there is a size reduction. 
			moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, thread, 1, tileY);
			//first tile has to copy the out-of-range elements, which are on the left-hand side, to next stream's inter_dep array
			moveTileToInterDepEdge(dev_arr, inter_dep, tt, tileX, tileY, tileT, nextstream, dep_stride, n1, tileIdx, rowsize, curBatch, padd);
			//variable isRegular == 1, because one row is shifted out-side-of the upper boundary
			//variable len == tileX-tt because this tile is not in a regular size.
			moveTileToInterDep(&inter_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, thread, nextstream, tileT, n1, segLengthX, tileIdx, tileX-tt, 1);
			//swap tile2 with tile1;
			for (int tid = thread; tid < 5120; tid+=blockDim.x){
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
		//ignLenX == tileT because tileX-tileT elements are copied at each row, ignLenY == tileT because tileY-tileT elements are copied at each column.
		glbPos = tileAddress;	
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileT, tileT, tileX, tileY, dep_stride, rowsize, thread);	
		__syncthreads();
		write_tile_lock_for_batch();

		//tile = 1 to xseg-1; regular size tiles, with index shifting.
		for (tileIdx = 1; tileIdx < xseg-1; tileIdx++){
			tileAddress = batchStartAddress + tileIdx * tileX;
			read_tile_lock_for_batch(timepiece, curBatch, tileIdx);
			//copy the base spatial data to shared memory for t=0.
			moveMatrixToTile(dev_arr, &tile1[0], segLengthX, tileX, tileY, dep_stride, tileAddress, rowsize, warpbatch, thread);
			for (int tt=0; tt<tileT; tt++){
				moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, thread, tileY);
				moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, thread, n2, segLengthX, padd, rowsize, tileIdx, tt, tileX);
				for (int tid = thread; tid < tileX * tileY; tid += blockDim.x){
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
						tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[idx2-segLengthX]) / 5;
				}	
				__syncthreads();
				//Set variable isRegular == 0 to disable the tile size reduction, when tile size are constant during the calculation. 
				moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, thread, 0, tileY);
				//variable isRegular == 1 because one row is shifted out-side-of the upper boundary.
				//variable len == tileX-tt because row is shifted out-side-of the upper boundary
				moveTileToInterDep(&inter_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, thread, nextstream, tileT, n1, segLengthX, tileIdx, tileX, tileX-tt, 1);
				//swap tile2 with tile1;
				for (int tid = thread; tid < 5120; tid+=blockDim.x){
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
			//ignLenX == 0 because all elements are copied at each row, ignLenY == tileT because tileY-tileT elements are copied at each column.
			glbPos = tileAddress;	
			moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, 0, tileT, tileX, tileY, dep_stride, rowsize, thread);	
			__syncthreads();
			
			write_tile_lock_for_batch(timepiece, curBatch, tile);
		}

		//when tile = xseg-1, if matrix is completely divided by the tile, no t0 elements copy to shared memory; 
		//use dependent data and out-of-range data to calculate.
		tileIdx = xseg-1;
		//unlike the other two cases that tileAddress points to the source pos of t0, here tileAddress is the destination pos of t(tileT-1).
		tileAddress = batchStartAddress + tileIdx * tileX - tt;
		read_tile_lock_for_batch(timepiece, curBatch, tileIdx);
		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, thread, tileY);
			//set variable offset == 1 if it is the last tile of each batch to copy right-side out-of-range elements to 
			moveIntraDepToTileEdge(dev_arr, &tile1[0], stride, rowsize, segLengthX, dep_stride, thread, tt, padd, n1, 1, tileY);
			moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, thread, n2, segLengthX, padd, rowsize, tileIdx, tt, tt + dev_stride);
			//tileX of the last tile is changed throughout the simulation from 0 to tileT;
			for (int tid = thread; tid < (tt+1) * tileY; tid += blockDim.x){
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
					tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[idx2-segLengthX]) / 5;
			}	
			__syncthreads();
			
			//variable isRegular == 1 because one row is shifted out-side-of the upper boundary.
			//len = tileX-1-tt, variable len specifies the lenth of eligible elements should be moved to inter_stream_dep[].
			moveTileToInterDep(&inter_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, thread, nextstream, tileT, n1, segLengthX, tileIdx, tileX, tt, 1);
			//swap tile2 with tile1;
			for (int tid = thread; tid < 5120; tid+=blockDim.x){
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
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileX-tileT, tileT, tileX, tileY, dep_stride, rowsize, thread);	
		__syncthreads();
		write_tile_lock_for_batch();

	}
	else if(curBatch == yseg){
	//for the last batch, all the tiles are irregular
		//when tile = 0, the calculated data which are outside the range are not copied to tile2, tile size is shrinking 
		//along T dimension. Out-of-range elements are used for dependent data.
		read_tile_lock_for_batch(timepiece, curBatch, tileIdx);
		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTileEdge(dev_arr, &tile1[0], stride, rowsize, segLengthX, dep_stride, thread, tt, padd, n1, 0, tt);
			//the first tile is not in regular size, so variable len = tileX-tt
			moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, tileY, dep_stride, thread, curSMStream, tileT, n1, segLengthX, tileIdx, tileX-tt);
			//move out-of-range elements which are beanth the bottom boundary to the tile
			//variable offset == 1, used to locate the bottom out-of-boundary elements.
			moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, thread, n2, segLengthX, padd, rowsize, tileIdx, tt, tileX, 1);
			
			for (int tid = thread; tid < tileX * tileY; tid += blockDim.x){
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
					tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[idx2-segLengthX]) / 5;
			}	
			__syncthreads();
			
			//Since the tile size is reduced along the calculation, the intraDep elements (in last two column of the valid tile) is also shifted to left.
			//Set variable isRegular == 1, when there is a size reduction. 
			moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, thread, 1, tt);
			//swap tile2 with tile1;
			for (int tid = thread; tid < 5120; tid+=blockDim.x){
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
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileT, tileY-tileT, tileX, tileY, dep_stride, rowsize, thread);	
		__syncthreads();
		write_tile_lock_for_batch();

		//tile = 1 to xseg-1; regular size tiles, with index shifting.
		for (tileIdx = 1; tileIdx < xseg-1; tileIdx++){
			read_tile_lock_for_batch(timepiece, curBatch, tileIdx);
			for (int tt=0; tt<tileT; tt++){
				//isRegular == 0 because this is a regular tile.
				moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, thread, 0, tt);
				moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, tileY, dep_stride, thread, curSMStream, tileT, n1, segLengthX, tileIdx, tileX);
				//move out-of-range elements which are beanth the bottom boundary to the tile
				//variable offset == 1, used to locate the bottom out-of-boundary elements.
				moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, thread, n2, segLengthX, padd, rowsize, tileIdx, tt, tileX, 1);
				for (int tid = thread; tid < tileX * tileY; tid += blockDim.x){
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
						tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[idx2-segLengthX]) / 5;
				}	
				__syncthreads();
				//isRegular == 0 to disable the tile size reduction, when tile size are constant during the calculation. 
				moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, thread, 0, tt);
				//variable len == tileX because the tile size is constant.
				moveTileToInterDep(&inter_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, thread, nextstream, tileT, n1, segLengthX, tileIdx, tileX, 0);
				//swap tile2 with tile1;
				for (int tid = thread; tid < 5120; tid+=blockDim.x){
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
			moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, 0, tileY-tileT, tileX, tileY, dep_stride, rowsize, thread);	
			__syncthreads();
			
			write_tile_lock_for_batch(timepiece, curBatch, tile);
		}

		//when tile = xseg-1, if matrix is completely divided by the tile, no t0 elements copy to shared memory; 
		//use dependent data and out-of-range data to calculate.
		tileIdx = xseg-1;
		read_tile_lock_for_batch(timepiece, curBatch, tileIdx);
		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, thread, tt);
			//set variable offset == 1 if it is the last tile of each batch to copy right-side out-of-range elements to 
			moveIntraDepToTileEdge(dev_arr, &tile1[0], stride, rowsize, segLengthX, dep_stride, thread, tt, padd, n1, 1, tt);
			
			//1. inter_dep elements from previous tile (on top of intra_dep elements); total size == len + dev_stride, where len == tt, which is 0 at t0
			//2. out-of-range elements
			//copy edge elements first to cover the out-of-range elements, then copy the inter_dep of previous stream and cover a part of the out-of-range elements.
			//variable len == tt + dev_stride, which covers the size of the elements, calculated in previous stream, and the out-of-range elements.
			moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, thread, n2, segLengthX, padd, rowsize, tileIdx, tt, tt + dev_stride);
			moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, tileY, dep_stride, thread, curSMStream, tileT, n1, segLengthX, tileIdx, tt);
			//move out-of-range elements which are beanth the bottom boundary to the tile
			//variable offset == 1, used to locate the bottom out-of-boundary elements.
			moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, thread, n2, segLengthX, padd, rowsize, tileIdx, tt, tt + dev_stride, 1);
			//tileX of the last tile is changed throughout the simulation from 0 to tileT;
			for (int tid = thread; tid < (tt+1) * tileY; tid += blockDim.x){
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
					tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[idx2-segLengthX]) / 5;
			}	
			__syncthreads();
			
			//swap tile2 with tile1;
			for (int tid = thread; tid < 5120; tid+=blockDim.x){
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
		glbPos = batchStartAddress + tileIdx * tileX -tt;	
		//ignLenX == tileX-tileT because tileT elements are copied at each row, ignLenY == tileY-tileT because tileT elements are copied at each column.
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileX-tileT, tileY-tileT, tileX, tileY, dep_stride, rowsize, thread);	
		__syncthreads();
		write_tile_lock_for_batch();
	
	}
	else{
	//for the regular batch, use the near-edge elements for the out-of-range dependence of first and last tile only.
		//when tile = 0, the calculated data which are outside the range are not copied to tile2, tile size is shrinking 
		//along T dimension. Out-of-range elements are used for dependent data.
		tileAddress = batchStartAddress + tileIdx * tileX;
		read_tile_lock_for_batch(timepiece, curBatch, tileIdx);
		moveMatrixToTile(dev_arr, &tile1[0], segLengthX, tileX, tileY, dep_stride, tileAddress, rowsize, warpbatch, thread);
		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTileEdge(dev_arr, &tile1[0], stride, rowsize, segLengthX, dep_stride, thread, tt, padd, n1, 0, tileY);
			//the first tile is not in regular size, so variable len = tileX-tt
			moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, tileY, dep_stride, thread, curSMStream, tileT, n1, segLengthX, tileIdx, tileX-tt);
			for (int tid = thread; tid < tileX * tileY; tid += blockDim.x){
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
					tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[idx2-segLengthX]) / 5;
			}	
			__syncthreads();
			
			//Since the tile size is reduced along the calculation, the intraDep elements (in last two column of the valid tile) is also shifted to left.
			//Set variable isRegular == 1, when there is a size reduction. 
			moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, thread, 1, tileY);
			//first tile has to copy the out-of-range elements, which are on the left-hand side, to next stream's inter_dep array
			moveTileToInterDepEdge(dev_arr, inter_dep, tt, tileX, tileY, tileT, nextstream, dep_stride, n1, tileIdx, rowsize, curBatch, padd);
			//variable len == tileX-tt because the tile size is reduced during calculation.
			//isRegular == 0 because there is no row move out-side-of upper boundary
			moveTileToInterDep(&inter_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, thread, nextstream, tileT, n1, segLengthX, tileIdx, tileX-tt, 0);
			//swap tile2 with tile1;
			for (int tid = thread; tid < 5120; tid+=blockDim.x){
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
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileT, 0, tileX, tileY, dep_stride, rowsize, thread);	
		__syncthreads();
		write_tile_lock_for_batch();

		//tile = 1 to xseg-1; regular size tiles, with index shifting.
		for (tileIdx = 1; tileIdx < xseg-1; tileIdx++){
			tileAddress = batchStartAddress + tileIdx * tileX;
			read_tile_lock_for_batch(timepiece, curBatch, tileIdx);
			//copy the base spatial data to shared memory for t=0.
			moveMatrixToTile(dev_arr, &tile1[0], segLengthX, tileX, tileY, dep_stride, tileAddress, rowsize, warpbatch, thread);
			for (int tt=0; tt<tileT; tt++){
				//isRegular == 0 because this is a regular tile.
				moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, thread, 0, tileY);
				moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, tileY, dep_stride, thread, curSMStream, tileT, n1, segLengthX, tileIdx, tileX);
				for (int tid = thread; tid < tileX * tileY; tid += blockDim.x){
					//out-of-range results should be ignored
					//because of the bias, xidx and yidx are the pos of new time elements.
					//thread % tileX and thread / tileX are pos of current cached elements.
					xidx = tid % tileX;
					yidx = tid / tileX;
				        //tilePos is the index of each element, to be calculated in the next timestamp. shifted left and up by 1
					tilePos = (dep_stride-1) * segLengthX + (dep_stride - 1) + yidx * segLengthX + xidx;	
					//newtilePos is the index where the new calculated elements should be stored into the shared tile2 array
					newtilePos = dep_stride * segLengthX + dep_stride + yidx * segLengthX + xidx;
					tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[idx2-segLengthX]) / 5;
				}	
				__syncthreads();
				//isRegular == 0 to disable the tile size reduction, when tile size are constant during the calculation. 
				moveTileToIntraDep(&intra_dep[0], &tile1[0], tt, tileX, tileY, segLengthX, dep_stride, thread, 0, tileY);
				//variable len == tileX because the tile size is constant.
				moveTileToInterDep(&inter_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, thread, nextstream, tileT, n1, segLengthX, tileIdx, tileX, 0);
				//swap tile2 with tile1;
				for (int tid = thread; tid < 5120; tid+=blockDim.x){
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
			moveShareToGlobal(&tile1[0], dev_arr, glbPos, tileX, tileY, dep_stride, rowsize, segLengthX, thread);	
			__syncthreads();
			
			write_tile_lock_for_batch(timepiece, curBatch, tile);
		}

		//when tile = xseg-1, if matrix is completely divided by the tile, no t0 elements copy to shared memory; 
		//use dependent data and out-of-range data to calculate.
		tileIdx = xseg-1;
		//unlike the other two cases that tileAddress points to the source pos of t0, here tileAddress is the destination pos of t(tileT-1).
		tileAddress = batchStartAddress + tileIdx * tileX - tt;
		read_tile_lock_for_batch(timepiece, curBatch, tileIdx);
		for (int tt=0; tt<tileT; tt++){
			moveIntraDepToTile(&intra_dep[0], &tile1[0], tt, tileY, segLengthX, dep_stride, thread, tileY);
			//set variable offset == 1 if it is the last tile of each batch to copy right-side out-of-range elements to 
			moveIntraDepToTileEdge(dev_arr, &tile1[0], stride, rowsize, segLengthX, dep_stride, thread, tt, padd, n1, 1, tileY);
			
			//1. inter_dep elements from previous tile (on top of intra_dep elements); total size == len + dev_stride, where len == tt, which is 0 at t0
			//2. out-of-range elements
			//copy edge elements first to cover the out-of-range elements, then copy the inter_dep of previous stream and cover a part of the out-of-range elements.
			//variable len == tt + dev_stride, which covers the size of the elements, calculated in previous stream, and the out-of-range elements.
			moveInterDepToTileEdge(dev_arr, &tile1[0], tileX, tileY, dep_stride, thread, n2, segLengthX, padd, rowsize, tileIdx, tt, tt + dev_stride);
			moveInterDepToTile(inter_stream_dep, &tile1[0], tt, tileX, tileY, dep_stride, thread, curSMStream, tileT, n1, segLengthX, tileIdx, tt);
			//tileX of the last tile is changed throughout the simulation from 0 to tileT;
			for (int tid = thread; tid < (tt+1) * tileY; tid += blockDim.x){
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
					tile2[newtilePos] = (tile1[tilePos+stride] + tile1[tilePos+segLengthX] + tile1[tilePos] + tile1[tilePos-stride] + tile1[idx2-segLengthX]) / 5;
			}	
			__syncthreads();
			
			//variable isRegular == 0 because one row is shifted out-side-of the upper boundary.
			//len = tileX-1-tt, variable len specifies the lenth of eligible elements should be moved to inter_stream_dep[].
			moveTileToInterDep(&inter_dep[0], &tile1[0], tt, tileX, tileY, dep_stride, thread, nextstream, tileT, n1, segLengthX, tileIdx, tileX, tt, 0);
			//swap tile2 with tile1;
			for (int tid = thread; tid < 5120; tid+=blockDim.x){
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
		//ignLenX == tileX-tileT because only tileT elements are copied in each row, ignLenY == 0 because no size reduction along Y dim.
		glbPos = tileAddress;	
		moveShareToGlobalEdge(&tile1[0], dev_arr, glbPos, tileX-tileT, 0, tileX, tileY, dep_stride, rowsize, thread);	
		__syncthreads();
		write_tile_lock_for_batch();

	}

	write_batch_lock_for_time(timepiece, curBatch);
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
	int stride = 1;
	int dep_stride = stride+1;
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

	int *dev_arr;
	volatile int *dev_lock;
	int *lock;
	size_t freeMem, totalMem;
	
	
	cudaMemGetInfo(&freeMem, &totalMem);
	int tablesize = colsize * rowsize;
	cout << "current GPU memory info FREE: " << freeMem << " Bytes, Total: " << totalMem << " Bytes.";
	cout << "colsize: " << colsize << ", rowsize: " << rowsize << ", allocates: " << tablesize * sizeof(int)<< " Bytes." << endl;
	cudaError err = cudaMalloc(&dev_arr, tablesize * sizeof(int));
	checkGPUError(err);
	
	cudaMemcpy((void*)dev_arr, arr, tablesize*sizeof(int), cudaMemcpyHostToDevice);

	int threadPerBlock = min(1024, rawElmPerTile);
//	int blockPerGrid = PTilesPerTimestamp;
	int blockPerGrid = 1;
	int numStream = 28;
	int warpbatch = threadPerBlock / 32;

//memory structure: stream --> tile --> time --> dependence --> tileX
	int *dev_inter_stream_dependence;
	int stream_dep_offset = tileT * (n1 + dep_stride) * dep_stride;
	int inter_stream_dependence = numStream * stream_dep_offset;
	err = cudaMalloc(&dev_inter_stream_dependence, inter_stream_dependence * sizeof(int)):
	checkGPUError(err);

	int xseg = n1 / tileX + 1;
	int yseg = n2 / tileY + 1;
	int stream_offset = yset % numStream;
	
	lock = new int[yseg];
	lock[0] = xseg;
	for (int i=1; i<yseg; i++)
		lock[i] = 0;
	cudaMalloc(&dev_lock, (yseg) * sizeof(int));	
	cudaMemcpy((void*)dev_lock, lock, (yseg)*sizeof(int), cudaMemcpyHostToDevice);
	cudaStream_t stream[numStream];
	for (int s=0; s<numStream; s++)
		cudaStreamCreate(&stream[s]);

//t < MAXTRIAL? or t <= MAXTRIAL	
	for(int t = 0; t <= MAXTRIAL; t+= tileT){
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
			int logicSMStream = curBatch % numStream;
			int curSMStream = (logicSMStream +  stream_offset * t) % numStream;
			int curStartAddress = curBatch * tileY * rowsize;
			int rowStartOffset = padd * rowsize + padd;
			int batchStartAddress = rowStartOffset + curStartAddress;
			int nextSMStream = (curSMStream + 1) % numStream;
//			cout << endl << "curBatch: " << curBatch << ", yseg: " << yseg << endl;	
			GPU_Tile<<<blockPerGrid, threadPerBlock, 0, stream[curSMStream]>>>(dev_arr, dev_lock, curBatch, curStartAddress, tileX, tileY,  padd, stride, rowStartOffset, rowsize, colsize, xseg, yseg, n1, n2, warpbatch, curSMStream, nextSMStream, dev_inter_stream_dependence, tileT, t, batchStartAddress);	
//			GPU<<<blockPerGrid, threadPerBlock>>>(dev_table, dev_arr1, dev_arr2, dev_lock, curBatch, curStartAddress, rowtiles, resX, tileX, tileY,  paddX, paddY, rowStartOffset, rowsize, colsize, xseg, yseg, tileY/tileX, n1, n2);			
			checkGPUError( cudaGetLastError() );
		}
		//this global synchronization enforces the sequential computation along t dimension.
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
	
	cudaFree((void*)dev_arr);
	cudaFree((void*)dev_lock);
	delete[] lock;

}

