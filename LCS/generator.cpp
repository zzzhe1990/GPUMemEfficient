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
	string filename2 = "str2_2_";
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
	convert1 << n1;
	convert2 << n2;
	filename1.append( convert1.str() );
	filename2.append( convert2.str() );
	filename1.append("_");
	filename1.append( filename2 );
	path.append(filename1);
	path.append(fileformat);

	srand(time(NULL));

	ofstream myfile;
	myfile.open(path.c_str());

	myfile << size1 << " " << size2 << endl;
	
	for (int i=0; i< size1; i++)
		myfile << rand()%4 << " ";
	myfile << endl;

	for (int i=0; i< size2; i++)
		myfile << rand()%4 << " ";
	myfile << endl;

	myfile.close();

	return 0;
}
