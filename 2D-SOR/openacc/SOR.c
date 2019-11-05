#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<string.h>
#include<errno.h>

int main(){
	int n = 4096;
	int trial = 64;
	int s = 5;	//stride
     	int padd = s * 2;
	int nn = n + padd + padd;

//	int* arr1 = (int*)malloc(nn * nn * sizeof(int));
//	int* arr2 = (int*)malloc(nn * nn * sizeof(int));

	static int arr1[8230][8230];
	static int arr2[8230][8230];

	for (int row = 0; row < nn; row++){
		for (int col = 0; col < nn; col++){
//			arr1[row * nn + col] = rand() % 100;
			arr1[row][col] = rand() % 100;
		}
	}	
	
//#pragma acc data copy(arr1[0:nn*nn]) create(arr2[0:nn*nn])	
#pragma acc data copy(arr1[0:nn][0:nn]) create(arr2[0:nn][0:nn])	
	{
		struct timeval tbegin, tend;
		gettimeofday(&tbegin, NULL);

		for (int t=0; t < trial; t++){
	        	#pragma acc kernels
	     		{
				#pragma acc loop tile(32,8) device_type(nvidia)
     	     			for (int r = padd; r < nn - padd; r++){
//					#pragma acc loop gang(16), vector(32)
					for (int c = padd; c < nn - padd; c++){
						int total = 0;
/*						for (int x = -s; x <= s; x++)
							total += arr1[r][c + x];
						for (int y = -s; y < 0; y++)
							total += arr1[r + y][c];
						for (int y = 1; y < s; y++)
							total += arr1[r + y][c];
						arr2[r][c] = total / ((s + s + 1) * 2 - 1);
*/
						for (int y = -s; y <= s; y++){
							for (int x = -s; x <= s; x++)
								total += arr1[r + y][c + x];
						}
						arr2[r][c] = total / (s + s + 1) / (s + s + 1);

					}
				}
				#pragma acc loop tile(32,8) device_type(nvidia)
		                for (int r = padd; r < nn - padd; r++){
					for (int c = padd; c < nn - padd; c++)
//						arr1[r * nn + c] = arr2[r * nn + c];
						arr1[r][c] = arr2[r][c];
				}
			}
		}
		#pragma acc wait
		gettimeofday(&tend, NULL);
		double tt = (double)(tend.tv_sec - tbegin.tv_sec) + (double)(tend.tv_usec - tbegin.tv_usec) / 1000000.0;
		printf("execution time: %lf s\n", tt);

	}

	return 0;
}
