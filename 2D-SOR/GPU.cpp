#include<iostream>
#include<cstdlib>
#include<fstream>
#include<sstream>
#include<string>
#include<sys/time.h>
#include "GPU.h"
using namespace std;
//#define DEBUG
const int MAXTRIAL = 100;
void readInputData(string str1, int &n1, int &n2, int &padd, int **arr1, int **arr2){
	ifstream inputfile;
	inputfile.open( str1.c_str() );
	
	if (!inputfile){
		cout << "ERROR: Input file cannot be opened!" << endl;
		exit(EXIT_FAILURE);
	}

	inputfile >> n1 >> n2 >> padd;
	
	*arr1 = new int[(n1+2) * (n2+2)];
	*arr2 = new int[(n1+2) * (n2+2)];

	for (int j=0; j<padd -1; j++){
		inputfile.ignore(2^15+2*padd, '\n');
	}

	for (int j=0; j<n2+2; j++){
		inputfile.ignore(3, '\n');
		for (int i=0; i<n1+2; i++)
			inputfile >> (*arr1)[j * (n1 +2)+ i];
		inputfile.ignore(2^15+2, '\n');
	}
}
/*
void readInputData(string str1, int &n1, int &n2, int **arr1, int **arr2){
	ifstream inputfile;
	inputfile.open( str1.c_str() );
	
	if (!inputfile){
		cout << "ERROR: Input file cannot be opened!" << endl;
		exit(EXIT_FAILURE);
	}

	inputfile >> n1 >> n2;
	
	*arr1 = new int[(n1+2) * (n2+2)];
	*arr2 = new int[(n1+2) * (n2+2)];

	for (int j=0; j<n2+2; j++){
		for (int i=0; i<n1+2; i++)
			inputfile >> (*arr1)[j * (n1 +2)+ i];
		inputfile.ignore(2^15+2, '\n');
	}
}
*/
void displayInput(int *arr, int n1, int n2){
	cout << "SOR table: ";
	for (int j=0; j<=n2+2; j++){
		for (int i=0; i<=n1+2; i++){
			cout << arr[j*(n1+2) + i] << " ";
		}
		cout << '\n';
	}
}


int main(int argc, char **argv){
	int nn1, nn2;
	if (argc != 3){
		cout << "Incorrect Input Parameters. Must be two string sizes." << endl;
		exit(EXIT_FAILURE);
	}
	else{
		nn1 = atoi(argv[1]);
		nn2 = atoi(argv[2]);
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

	string str1;
	str1.append(filepath);
	str1.append(filename);
	str1.append(fileformat);

	int n1, n2, padd;
	int *arr1, *arr2;
	
	readInputData(str1, n1, n2, padd, &arr1, &arr2);

//	displayInput(arr1, arr2, n1, n2);
	
	struct timeval tbegin, tend;

	gettimeofday(&tbegin, NULL);
	
	SOR(n1, n2, padd, arr1, arr2, MAXTRIAL);

	gettimeofday(&tend, NULL);

	double s = (double)(tend.tv_sec - tbegin.tv_sec) + (double)(tend.tv_usec - tbegin.tv_usec)/1000000.0;

	cout << "execution time: " << s << " second." << endl;
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
			output << table[i*(n1+2)+ j] << " ";
		}
		output << '\n';
	}
	output.close();
#endif
	delete[] arr1;
	delete[] arr2;

	return 0;
}
