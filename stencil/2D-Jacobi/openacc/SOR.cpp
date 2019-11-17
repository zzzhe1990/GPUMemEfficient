#include<iostream>
#include<cstdlib>
#include<fstream>
#include<sstream>
#include<string>
#include<cmath>
#include<sys/time.h>
#include<cstring>

using namespace std;
//#define DEBUG

void readInputData(string str1, int &n1, int &n2, int &padd, int **arr1, int **arr2){
	ifstream inputfile;
	inputfile.open( str1.c_str() );
	
	if (!inputfile){
		cout << "ERROR: Input file cannot be opened!" << endl;
		exit(EXIT_FAILURE);
	}

	inputfile >> n1 >> n2 >> padd;
	
	*arr1 = new int[(n1+2*padd) * (n2+2*padd)];
	*arr2 = new int[(n1+2*padd) * (n2+2*padd)];

	for (int j=0; j<n2+2*padd; j++){
		for (int i=0; i<n1+2*padd; i++){
			inputfile >> (*arr1)[j * (n1 +2*padd)+ i];
		}
	}
}

void displayInput(int *arr, int n1, int n2, int padd){
	cout << "SOR table: " << endl;
	for (int j=0; j<n2+2*padd; j++){
		for (int i=0; i<n1+2*padd; i++){
			cout << arr[j*(n1+2*padd) + i] << " ";
		}
		cout << '\n';
	}
}

int _5ptSOR(int* arr1, int* arr2, int n1, int n2, int idx, int idy, int padd){
	return (arr1[(idy-1)*(n1+2*padd) + idx] + arr1[idy * (n1+2*padd) + idx - 1] + arr1[idy * (n1+2*padd) + idx] + arr1[idy * (n1+2*padd) + idx + 1] + arr1[(idy+1)*(n1+2*padd) + idx]) / 5;
}

int _9pt_SQUARE_SOR(int* arr1, int* arr2, int n1, int n2, int idx, int idy, int padd){
	return (arr1[(idy-1)*(n1+2*padd) + idx - 1] + arr1[(idy - 1) * (n1+2*padd) + idx] + arr1[(idy - 1) * (n1+2*padd) + idx + 1] + arr1[idy * (n1+2*padd) + idx - 1] + arr1[idy*(n1+2*padd) + idx] + arr1[idy * (n1+2*padd) + idx + 1] + arr1[(idy+1) * (n1+2*padd) + idx - 1] + arr1[(idy+1) * (n1+2*padd) + idx] + arr1[(idy+1) * (n1+2*padd) + idx + 1]) / 9;
}

int _9pt_CROSS_SOR(int* arr1, int* arr2, int n1, int n2, int idx, int idy, int padd){
	return (arr1[(idy-1)*(n1+2*padd) + idx] + arr1[(idy - 2) * (n1+2*padd) + idx] + arr1[(idy + 1) * (n1+2*padd) + idx] + arr1[(idy + 2) * (n1+2*padd) + idx] + arr1[idy*(n1+2*padd) + idx - 2] + arr1[idy * (n1+2*padd) + idx - 1] + arr1[idy * (n1+2*padd) + idx] + arr1[idy * (n1+2*padd) + idx + 1] + arr1[idy * (n1+2*padd) + idx + 2]) / 9;
}

int _25pt_SQUARE_SOR(int* arr1, int* arr2, int n1, int n2, int idx, int idy, int padd){
	int total = 0;
	for (int i = -2; i <= 2; i++){
		for (int j = -2; j <= 2; j++){
			total += arr1[(idy + i) * (n1 + 2 * padd) + idx + j];
		}
	}
	return total / 25;
}

int _13pt_CROSS_SOR(int* arr1, int* arr2, int n1, int n2, int idx, int idy, int padd){
	int total = 0;
	for (int i = -3; i < 0; i++){
		total += arr1[(idy + i) * (n1 + 2 * padd) + idx];
	}
	for (int i = 1; i <= 3; i++){
		total += arr1[(idy + i) * (n1 + 2 * padd) + idx];
	}
	for (int i = -3; i <= 3; i++){
		total += arr1[idy * (n1 + 2 * padd) + idx + i];
	}
	return total / 13;
}

int _49pt_SQUARE_SOR(int* arr1, int* arr2, int n1, int n2, int idx, int idy, int padd){
	int total = 0;
	for (int i = -3; i <= 3; i++){
		for (int j = -3; j <= 3; j++){
			total += arr1[(idy + i) * (n1 + 2 * padd) + idx + j];
		}
	}
	return total / 49;
}

int _17pt_CROSS_SOR(int* arr1, int* arr2, int n1, int n2, int idx, int idy, int padd){
	int total = 0;
	for (int i = -4; i < 0; i++){
		total += arr1[(idy + i) * (n1 + 2 * padd) + idx];
	}
	for (int i = 1; i <= 4; i++){
		total += arr1[(idy + i) * (n1 + 2 * padd) + idx];
	}
	for (int i = -4; i <= 4; i++){
		total += arr1[idy * (n1 + 2 * padd) + idx + i];
	}
	return total / 17;
}

int _81pt_SQUARE_SOR(int* arr1, int* arr2, int n1, int n2, int idx, int idy, int padd){
	int total = 0;
	for (int i = -4; i <= 4; i++){
		for (int j = -4; j <= 4; j++){
			total += arr1[(idy + i) * (n1 + 2 * padd) + idx + j];
		}
	}
	return total / 81;
}



void SOR(int n1, int n2, int padd, int *arr1, int *arr2, int trial){
#pragma acc data copy(arr1, arr2)
	for (int t=0; t < trial; t+=2){
#pragma acc kernels loop
		for (int y = padd; y < n2 + padd; y++){
#pragma acc loop gang(16), vector(32)
			for (int x = padd; x < n1 + padd; x++){
				int idx = x;
				int idy = y;
//				arr2[idy * (n1 + 2 * padd) + idx] = _5ptSOR(arr1, arr2, n1, n2, idx, idy, padd);
//				arr2[idy * (n1 + 2 * padd) + idx] = _9pt_SQUARE_SOR(arr1, arr2, n1, n2, idx, idy, padd);
//				arr2[idy * (n1 + 2 * padd) + idx] = _9pt_CROSS_SOR(arr1, arr2, n1, n2, idx, idy, padd);
//				arr2[idy * (n1 + 2 * padd) + idx] = _25pt_SQUARE_SOR(arr1, arr2, n1, n2, idx, idy, padd);
//				arr2[idy * (n1 + 2 * padd) + idx] = _13pt_CROSS_SOR(arr1, arr2, n1, n2, idx, idy, padd);
//				arr2[idy * (n1 + 2 * padd) + idx] = _49pt_SQUARE_SOR(arr1, arr2, n1, n2, idx, idy, padd);
//				arr2[idy * (n1 + 2 * padd) + idx] = _17pt_CROSS_SOR(arr1, arr2, n1, n2, idx, idy, padd);
				arr2[idy * (n1 + 2 * padd) + idx] = _81pt_SQUARE_SOR(arr1, arr2, n1, n2, idx, idy, padd);
			}	
		}
#pragma acc kernels loop
		for (int y = padd; y < n2 + padd; y++){
#pragma acc loop gang(16), vector(32)
			for (int x = padd; x < n1 + padd; x++){
				int idx = x;
				int idy = y;
//				arr1[idy * (n1 + 2 * padd) + idx] = _5ptSOR(arr1, arr2, n1, n2, idx, idy, padd);
//				arr1[idy * (n1 + 2 * padd) + idx] = _9pt_SQUARE_SOR(arr1, arr2, n1, n2, idx, idy, padd);
//				arr1[idy * (n1 + 2 * padd) + idx] = _9pt_CROSS_SOR(arr1, arr2, n1, n2, idx, idy, padd);
//				arr1[idy * (n1 + 2 * padd) + idx] = _25pt_SQUARE_SOR(arr1, arr2, n1, n2, idx, idy, padd);
//				arr1[idy * (n1 + 2 * padd) + idx] = _13pt_CROSS_SOR(arr1, arr2, n1, n2, idx, idy, padd);
//				arr1[idy * (n1 + 2 * padd) + idx] = _49pt_SQUARE_SOR(arr1, arr2, n1, n2, idx, idy, padd);
//				arr1[idy * (n1 + 2 * padd) + idx] = _17pt_CROSS_SOR(arr1, arr2, n1, n2, idx, idy, padd);
				arr1[idy * (n1 + 2 * padd) + idx] = _81pt_SQUARE_SOR(arr2, arr1, n1, n2, idx, idy, padd);
			}	
		}
//		displayInput(*arr1, n1, n2, padd);
	}
}



int main(int argc, char **argv){
	int nn1, nn2, trial, stride;
	if (argc != 5){
		cout << "Incorrect Input Parameters. 4 variables: two string sizes, one trial, one stride." << endl;
		exit(EXIT_FAILURE);
	}
	else{
		nn1 = atoi(argv[1]);
		nn2 = atoi(argv[2]);
		trial = atoi(argv[3]);
		stride = atoi(argv[4]);
	}

	ostringstream convert1, convert2;
	convert1 << nn1;
	convert2 << nn2;

	string filename, filename1, filename2, filepath, fileformat;
	filepath = "./Data/";
	filename1 = "x_2_";
	filename2 = "_y_2_";	
	fileformat = ".txt";

	filename.append(filename1.c_str());
	filename.append(convert1.str());
	filename.append(filename2.c_str());
	filename.append(convert2.str());
	filename.append("_STRIDE_" + std::to_string(stride));

	string str1;
	str1.append(filepath);
	str1.append(filename);
	str1.append(fileformat);

	int n1, n2, padd;
	int *arr1, *arr2;
	
	readInputData(str1, n1, n2, padd,  &arr1, &arr2);

//	displayInput(arr1, n1, n2, padd);
	
	memcpy(arr2, arr1, sizeof(int) * (n1 + 2 * padd) * (n2 + 2 * padd));
	
	struct timeval tbegin, tend;

	gettimeofday(&tbegin, NULL);
	
	SOR(n1, n2, padd, arr1, arr2, trial);

	gettimeofday(&tend, NULL);
	
	displayInput(arr1, n1, n2, padd);

	double s = (double)(tend.tv_sec - tbegin.tv_sec) + (double)(tend.tv_usec - tbegin.tv_usec)/1000000.0;

	cout << "execution time: " << s << " second." << endl;
#ifdef DEBUG
	for (int i=0; i<n2+2*padd; i++){
		for (int j=0; j<n1+2*padd; j++){
			cout << res[i*(n1+2*padd)+ j] << " ";
		}
		cout << endl;
	}

	string outfile = "./Output/output_";
	outfile.append(convert1.str());
	outfile.append(".txt");

	std::ofstream output (outfile);

	if (!output){
		cout << "ERROR: output file cannot be opened!" << endl;
		exit(EXIT_FAILURE);
	}
	
	for (int i=0; i<n2+2*padd; i++){
		for (int j=0; j<n1+2*padd; j++){
			output << res[i*(n1+2*padd)+ j] << " ";
		}
		output << '\n';
	}
	output << n1 << '\n';
	output.close();
#endif
	delete[] arr1;
	delete[] arr2;

	return 0;
}
