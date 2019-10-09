#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<sys/time.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<string.h>
#include<errno.h>
/*
void readInputData(int n1, int n2, int padd, int *arr1){
	FILE* fp = fopen("./Data/x_2_3_y_2_3_STRIDE_1.txt", "r");
	if (fp == NULL){
		printf("Error Message: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}
	char* line = NULL;
	size_t len = 0;
	
//this code would not work on Windows.
	if (getline(&line, &len, fp) == -1){
		perror("getline error");
		exit(EXIT_FAILURE);
	}

	int row = 0, col = 0;
	while(getline(&line, &len, fp) != -1){
		col = 0;
		char* word = strtok(line, " ");
		while(word != NULL){
			int tmp;
			sscanf(word, "%d", &tmp);
			int idx = row * (n1 + 2 * padd) + col;
			arr1[idx] = tmp;
			col++;
			word = strtok(NULL, " ");
		}	
		row++;
	}

	if (row != n2 + 2 * padd){
		printf("Error: read input data incorrectly\n");
		exit(EXIT_FAILURE);
	}

	if (fclose(fp) < 0){
		printf("Error Message: %s\n", strerror(errno));
		exit(EXIT_FAILURE);
	}

	free(line);
}

void displayInput(int *arr, int n1, int n2, int padd){
	printf("SOR table: \n");
	for (int j=0; j<n2+2*padd; j++){
		for (int i=0; i<n1+2*padd; i++){
			printf("%d ", arr[j*(n1+2*padd) + i]);
		}
		printf("\n");
	}
}
*/
void SOR(int tablesize, int arr1[tablesize], int arr2[tablesize], int n1, int n2, int padd){
#pragma scop
	for (int y = padd; y < n2 + padd; y++){
		for (int x = padd; x < n1 + padd; x++){
			int idx = x;
			int idy = y;
			arr2[idy * (n1 + 2 * padd) + idx] = (arr1[(idy-1)*(n1+2*padd) + idx] + arr1[idy * (n1+2*padd) + idx - 1] + arr1[idy * (n1+2*padd) + idx] + arr1[idy * (n1+2*padd) + idx + 1] + arr1[(idy+1)*(n1+2*padd) + idx]) / 5;
		}	
	}
#pragma endscop
}

int main(int argc, char **argv){
	int trial = 64;
	int stride = 1;
//	int n1 = (int)pow(2.0, (double)nn1);
//     	int n2 = (int)pow(2.0, (double)nn2);
     	int padd = 2;
//	int *arr1, *arr2;
//	arr1 = (int*)malloc((n1 + 2 * padd) * (n2 + 2 * padd) * sizeof(int));
//	arr2 = (int*)malloc((n1 + 2 * padd) * (n2 + 2 * padd) * sizeof(int));

	int n1 = 8, n2 = 8;
	int nn1 = n1 + 2 * padd;
	int nn2 = n2 + 2 * padd;
	int tablesize = nn1 * nn2;
	int arr1[nn1][nn2];
	int arr2[nn1][nn2];
//	readInputData(n1, n2, padd, arr1);
//	memcpy(arr2, arr1, sizeof(int) * tablesize);

//	displayInput(arr1, n1, n2, padd);

	for (int row = 0; row < nn1; row++){
		for (int col = 0; col < nn2; col++){
			arr1[row][col] = rand() % 10;
			arr2[row][col] = arr1[row][col];
		}
	}	
	
	struct timeval tbegin, tend;

	gettimeofday(&tbegin, NULL);
	
	for (int t=0; t < trial; t += 2){
		SOR(tablesize, arr1, arr2, n1, n2, padd);
		SOR(tablesize, arr2, arr1, n2, n1, padd);
	}

	gettimeofday(&tend, NULL);
	
//	displayInput(arr1, n1, n2, padd);

	double s = (double)(tend.tv_sec - tbegin.tv_sec) + (double)(tend.tv_usec - tbegin.tv_usec)/1000000.0;

	printf("execution time: %lf second.\n", s);
	free(arr1);
	free(arr2);

	return 0;
}
