#include<iostream>
#include<cstdlib>
#include<fstream>
#include<sstream>
#include<string>
#include<sys/time.h>
#include"GPU_2DSOR_PARALLELOGRAM.h"
using namespace std;
//#define DEBUG
//#define batchexe

void readInputData(string str1, int &n1, int &n2, int& padd, int **arr){
	ifstream inputfile;
	inputfile.open( str1.c_str() );
	
	if (!inputfile){
		cout << "ERROR: Input file cannot be opened!" << endl;
		exit(EXIT_FAILURE);
	}

	inputfile >> n1 >> n2 >> padd;
	*arr = new int[(n1+2*padd) * (n2+2*padd)];

/*	for (int j=0; j<padd -1; j++){
		inputfile.ignore(2^15+2*padd, '\n');
	}
*/
	for (int j=0; j<n2+2*padd; j++){
//		inputfile.ignore(3, '\n');
		for (int i=0; i<n1+2*padd; i++)
			inputfile >> (*arr)[j * (n1 +2*padd)+ i];
		inputfile.ignore(2^15+2*padd, '\n');
	}
}
/*
void readInputData(string str1, int &n1, int &n2, int **arr){
	ifstream inputfile;
	inputfile.open( str1.c_str() );
	
	if (!inputfile){
		cout << "ERROR: Input file cannot be opened!" << endl;
		exit(EXIT_FAILURE);
	}

	inputfile >> n1 >> n2;
	
	*arr = new int[(n1+2) * (n2+2)];

	for (int j=0; j<n2+2; j++){
		for (int i=0; i<n1+2; i++)
			inputfile >> (*arr)[j * (n1 +2)+ i];
		inputfile.ignore(2^15+2, '\n');
	}
}
*/
void displayInput(int *arr, int n1, int n2, int padd){
	cout << "SOR table: " << endl;
	for (int j=0; j<n2+2*padd; j++){
		for (int i=0; i<n1+2*padd; i++){
			cout << arr[j*(n1+2*padd) + i] << " ";
		}
		cout << '\n';
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
	int *arr;
	
	readInputData(str1, n1, n2, padd, &arr);

//	displayInput(arr, n1, n2, padd);
	
	struct timeval tbegin, tend;

	gettimeofday(&tbegin, NULL);

#ifdef batchexe
	for (int i=0; i<100; i++)
#endif	
	SOR(n1, n2, stride, padd, arr, trial);

	gettimeofday(&tend, NULL);

	double s = (double)(tend.tv_sec - tbegin.tv_sec) + (double)(tend.tv_usec - tbegin.tv_usec)/1000000.0;

//	cout << "total execution time: " << s << " second." << endl;
#ifdef DEBUG
	string outfile = "./Output/output_LCS_";
	outfile.append(convert1.str());
	outfile.append(".txt");

	std::ofstream output (outfile);

	if (!output){
		cout << "ERROR: output file cannot be opened!" << endl;
		exit(EXIT_FAILURE);
	}
	
	output << n1 << '\n';
	
	for (int i=0; i<n2+2; i++){
		for (int j=0; j<n1+2; j++){
			output << arr[i*(n1+2*padd)+ j] << " ";
		}
		output << '\n';
	}
	output.close();
#endif
	delete[] arr;

	return 0;
}
