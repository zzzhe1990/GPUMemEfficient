#include<iostream>
#include<cstdlib>
#include<ifstream>
#include<string>

using namespace std;


void readInputData(string str1){
	ifstream inputfile;
	inputfile.open( str1.c_str() );
	
	if (!inputfile){
		cout << "ERROR: Input file cannot be opened!" << endl;
		exit(EXIT_FAILURE);
	}
}


int LCS(){


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

	readInputDate(str1);

	return 0;
}
