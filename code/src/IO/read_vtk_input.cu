
# include "read_vtk_input.cuh"
# include <iostream>
# include <iomanip>
# include <fstream>
# include <sstream>
# include <stdlib.h>
using namespace std;  



// --------------------------------------------------------
// Read lattice information from file:
// --------------------------------------------------------

void read_vtk_polydata(std::string fname, int nNodes, int nFaces, 
                       float* x, float* y, float* z, int* triV1, int* triV2, int* triV3)
{
	
	// -----------------------------------------------
	// variables for reading:
	// -----------------------------------------------
	
	int N;
	string word;
	string line;
	
	// -----------------------------------------------
	// open file:
	// -----------------------------------------------
	
	ifstream infile;
	infile.open(fname, ios::in);
	
	// -----------------------------------------------
	// read header:
	// -----------------------------------------------
	
	getline(infile,line);
	getline(infile,line);
	getline(infile,line);
	getline(infile,line);
	infile >> word >> N >> word; 
	if (N != nNodes) cout << "number of IBM nodes is not consistent with vtk input file = " << N << endl;
		
	// -----------------------------------------------
	// read node positions:
	// -----------------------------------------------
	
	for (int i=0; i<nNodes; i++) {
		infile >> x[i] >> y[i] >> z[i];
	}
	
	// -----------------------------------------------
	// read face vertices:
	// -----------------------------------------------
	
	int placeholder;	
	int F;
	infile >> word >> F >> placeholder;
	if (F != nFaces) cout << "number of IBM faces is not consistent with vtk input file = " << F << endl;
	
	for (int i=0; i<nFaces; i++) {
		infile >> placeholder >> triV1[i] >> triV2[i] >> triV3[i];
	}
	
	// -----------------------------------------------
	// close file:
	// -----------------------------------------------
	
	infile.close();
	
}