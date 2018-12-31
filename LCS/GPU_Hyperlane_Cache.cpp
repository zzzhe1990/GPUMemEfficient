#include<iostream>
#include<cstdlib>
#include<fstream>
#include<sstream>
#include<string>
#include<sys/time.h>
#include "GPU_Hyperlane_Share.h"

using namespace std;
//#define DEBUG
#define batchexe
const int itr=100;

void readInputData(string str1, int &n1, int &n2, int **arr1, int **arr2){
	ifstream inputfile;
	inputfile.open( str1.c_str() );
	
	if (!inputfile){
		cout << "ERROR: Input file cannot be opened!" << endl;
		exit(EXIT_FAILURE);
	}

	inputfile >> n1 >> n2;
	
	*arr1 = new int[n1];
	*arr2 = new int[n2];

	for (int i=0; i<n1; i++)
		inputfile >> (*arr1)[i];
	inputfile.ignore(2^15,'\n');
	for (int i=0; i<n2; i++)
		inputfile >> (*arr2)[i];
	inputfile.ignore(2^15,'\n');
}

void displayInput(int *arr1, int *arr2, int n1, int n2){
	cout << "string1: ";
	for (int i=0; i<n1; i++){
		cout << arr1[i] << " ";
	}
	cout << endl << "string2: ";
	for (int i=0; i<n2; i++){
		cout << arr2[i] << " ";
	}
	cout << endl;
}


int main(int argc, char **argv){
	int nn1, nn2;
	if (argc !=3){
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
	filename1 = "str1_2_";
	filename2 = "_str2_2_";	
	fileformat = ".txt";
	
	filename.append(filename1.c_str());
	filename.append(convert1.str());
	filename.append(filename2.c_str());
	filename.append(convert2.str());

	string str1;
	str1.append(filepath);
	str1.append(filename);
	str1.append(fileformat);

	int n1, n2;
	int *arr1, *arr2, *table;
	int lcslength;	
	int paddX = 3;
	int paddY = 1;	

	readInputData(str1, n1, n2, &arr1, &arr2);
	
	cout << "input data loaded" << endl;
//	displayInput(arr1, arr2, n1, n2);

	table = new int[(n2+paddX) * (n1+paddY)];
	
	struct timeval tbegin, tend;

	gettimeofday(&tbegin, NULL);

#ifdef batchexe
	for (int i=0; i<itr; i++)
#endif	
	lcslength = LCS(n1, n2, arr1, arr2, paddX, paddY, table);

	gettimeofday(&tend, NULL);

	double s = (double)(tend.tv_sec - tbegin.tv_sec)*1000 + (double)(tend.tv_usec - tbegin.tv_usec)/1000.0;

	cout << "lcs length: " << lcslength << endl;
	cout << "execution time: " << s/itr << " ms." << endl;
#ifdef DEBUG
	string outfile = "./Output/output_Hyperlane_";
	outfile.append(convert1.str());
	outfile.append(".txt");

	std::ofstream output;
	output.open(outfile.c_str());

	if (!output){
		cout << "ERROR: output file cannot be opened!" << endl;
		exit(EXIT_FAILURE);
	}

	output << n1 << '\n';

	for (int i=0; i<n1; i++){
		for (int j=0; j<n2; j++){
			output << table[(i+paddY)*(n2+paddX) + (j+paddX)] << ' ';
		}
		output << '\n';
	}
	output.close();
#endif
	delete[] arr1;
	delete[] arr2;
	delete[] table;

	return 0;
}
