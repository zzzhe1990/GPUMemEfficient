#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<string.h>
#include<errno.h>

const int n1 = 8192, n2 = 8192;
const int nn1 = 8196, nn2 = 8196;

void SOR(int len1, int len2, int arr1[nn1][nn2], int arr2[nn1][nn2], int padd, int trial){
	struct timeval tbegin, tend;
	gettimeofday(&tbegin, NULL);
	
#pragma scop
	for (int t=0; t < trial; t += 2){
		for (int r = padd; r < len1; r++){
			for (int c = padd; c < len2; c++){
				arr2[r][c] = (arr1[r-1][c] + arr1[r][c-1] + arr1[r][c] + arr1[r][c+1] + arr1[r+1][c]) / 5;
			}	
		}
		for (int r = padd; r < len1; r++){
			for (int c = padd; c < len2; c++){
				arr1[r][c] = (arr2[r-1][c] + arr2[r][c-1] + arr2[r][c] + arr2[r][c+1] + arr2[r+1][c]) / 5;
			}	
		}
	}
#pragma endscop
	
	gettimeofday(&tend, NULL);
	double tt = (double)(tend.tv_sec - tbegin.tv_sec) + (double)(tend.tv_usec - tbegin.tv_usec) / 1000000.0;
	printf("execution time: %lf s\n", tt);
}

int main(){
	int trial = 64;
     	int padd = 2;
	static int arr1[nn1][nn2];
	static int arr2[nn1][nn2];

	for (int row = 0; row < nn1; row++){
		for (int col = 0; col < nn2; col++){
			arr1[row][col] = rand() % 100;
			arr2[row][col] = arr1[row][col];
		}
	}	


	SOR(n1 + padd, n2 + padd, arr1, arr2, padd, trial);

	return 0;
}
