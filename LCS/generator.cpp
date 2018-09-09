#include<fstream>
#include<iostream>
#include<cstdlib>
#include<string>
#include<sstream>
#include<cmath>

using namespace std;

int main(int argc, char *argv[]){
	int n1, n2;
	string path = "./Data/";	
	string filename1 = "str1_2_";
	string filename2 = "_str2_2_";
	string fileformat = ".txt";	
	
	if (argc != 3){
		cout << "Incorrect parameters have been passed." << endl;
		exit(0);
	}	
	else{
		n1 = atoi(argv[1]);
		n2 = atoi(argv[2]);	
	}

	int size1, size2;
	
	size1 = pow(2, n1);
	size2 = pow(2, n2);

	ostringstream convert1, convert2;

	int *arr1, *arr2;
	arr1 = new int[size1];
	arr2 = new int[size2];
	srand(time(NULL));
	
	for (int i=0; i<size1; i++)	
		arr1[i] = rand()%4;
	for (int i=0; i<size2; i++)
		arr2[i] = rand()%4;

	while (n1 >= 8){
		size1 = pow(2, n1);
		size2 = pow(2, n2);
		filename1 = "str1_2_";
		filename2 = "_str2_2_";
		path = "./Data/";
		filename1.append( std::to_string(n1) );
		filename2.append( std::to_string(n2) );
		filename1.append( filename2 );
		path.append(filename1);
		path.append(fileformat);

		ofstream myfile;
		myfile.open(path.c_str());

		myfile << size1 << " " << size2 << endl;
	
		for (int i=0; i< size1; i++)
			myfile << arr1[i] << " ";
		myfile << endl;

		for (int i=0; i< size2; i++)
			myfile << arr2[i] << " ";
		myfile << endl;
	
		myfile.close();
		n1--;
		n2--;
	}


	delete[] arr1;
	delete[] arr2;
	return 0;
}
