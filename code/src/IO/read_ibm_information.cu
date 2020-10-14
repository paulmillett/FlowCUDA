
# include "read_ibm_information.cuh"
# include <iostream>
# include <iomanip>
# include <fstream>
# include <sstream>
# include <stdlib.h>
using namespace std;  



// --------------------------------------------------------
// Read lattice information from file:
// --------------------------------------------------------

void read_ibm_information(std::string fname, int nNodes, int nFaces, 
                          float* x, float* y, float* z, int* triV1, int* triV2, int* triV3)
{
	
	// -----------------------------------------------
	// variables for reading:
	// -----------------------------------------------
	
	int nN, nF;
		
	// -----------------------------------------------
	// open file:
	// -----------------------------------------------
	
	ifstream infile;
	infile.open(fname, ios::in);
	
	// -----------------------------------------------
	// read header:
	// -----------------------------------------------
		
	infile >> nN >> nF; 
	if (nN != nNodes) cout << "number of IBM nodes is not consistent with ibm input file = " << nN << endl;
	if (nF != nFaces) cout << "number of IBM faces is not consistent with ibm input file = " << nF << endl;
			
	// -----------------------------------------------
	// read node positions:
	// -----------------------------------------------
	
	for (int i=0; i<nNodes; i++) {
		infile >> x[i] >> y[i] >> z[i];
	}
	
	// -----------------------------------------------
	// read face vertices:
	// -----------------------------------------------
		
	for (int i=0; i<nFaces; i++) {
		infile >> triV1[i] >> triV2[i] >> triV3[i];
		triV1[i] -= 1;  // adjust from 1-based to 0-based indexing
		triV2[i] -= 1;  // " "
		triV3[i] -= 1;  // " " 
	}
	
	// -----------------------------------------------
	// close file:
	// -----------------------------------------------
	
	infile.close();
	
}