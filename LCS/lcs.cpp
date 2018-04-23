#include<iostream>
#include<cstdlib>
#include<fstream>
#include<string>
#include<sys/time.h>

using namespace std;


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

	for (int i=0; i<n2; i++)
		inputfile >> (*arr2)[i];

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

int LCS(int n1, int n2, int *arr1, int *arr2, int *table){
	int lcslength;

	for (int i=0; i<=n2; i++)
		table[i] = 0;

	for (int i=0; i<(n1+1)*(n2+1); i+=(n2+1))
		table[i] = 0;

	for (int i=0; i < n1; i++){
		for (int j=0; j < n2; j++){
			int idx = i+1;
			int idy = j+1;

			if (arr1[i] == arr2[j]){
				table[idx * (n2+1) + idy] = table[i * (n2+1) +j] + 1;
			}
			else{
				table[idx * (n2+1) + idy] = max(table[idx * (n2+1) + j], table[i * (n2+1) + idy]);
			}
		}
	}
/*
	//display table
	cout << "full table: " << endl;
	for (int i=0; i<n1; i++){
		int idx = i+1;
		for (int j=0; j<n2; j++){
			int idy = j+1;
			cout << table[idx * (n2+1) + idy] << " ";
		}
		cout << endl;
	}
*/	
	lcslength = table[ (n1+1) * (n2+1) - 1];

	return lcslength;
}



int main(){
	string filename, filepath, fileformat;
	filepath = "./Data/";
	filename = "str1_2_15_str2_2_15";	
	fileformat = ".txt";

	string str1;
	str1.append(filepath);
	str1.append(filename);
	str1.append(fileformat);

	int n1, n2;
	int *arr1, *arr2, *table;
	int lcslength;	
	
	readInputData(str1, n1, n2, &arr1, &arr2);

//	displayInput(arr1, arr2, n1, n2);

	table = new int[(n1+1) * (n2+1)];
	
	struct timeval tbegin, tend;

	gettimeofday(&tbegin, NULL);
	
	lcslength = LCS(n1, n2, arr1, arr2, table);

	gettimeofday(&tend, NULL);

	double s = (double)(tend.tv_sec - tbegin.tv_sec) + (double)(tend.tv_usec - tbegin.tv_usec)/1000000.0;

	cout << "lcs length: " << lcslength << endl;
	cout << "execution time: " << s << " second." << endl;

	delete[] arr1;
	delete[] arr2;
	delete[] table;

	return 0;
}
