
# include "read_ibm_information.cuh"
# include <iostream>
# include <iomanip>
# include <fstream>
# include <sstream>
# include <stdlib.h>
using namespace std;  



// --------------------------------------------------------
// Read IBM mesh information from file:
// --------------------------------------------------------

void read_ibm_information(std::string fname, int nNodes, int nFaces, 
                          float3* r, int* triV1, int* triV2, int* triV3)
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
		infile >> r[i].x >> r[i].y >> r[i].z;
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



// --------------------------------------------------------
// Read IBM mesh information from file:
// This includes the edge information for verts
// --------------------------------------------------------

void read_ibm_information_long(std::string fname, int nNodes, int nFaces, int nEdges,
                               float3* r, triangle* faces, edge* edges)
{
	
	// -----------------------------------------------
	// variables for reading:
	// -----------------------------------------------
	
	int nN, nF, nE;
		
	// -----------------------------------------------
	// open file:
	// -----------------------------------------------
	
	ifstream infile;
	infile.open(fname, ios::in);
	
	// -----------------------------------------------
	// read header:
	// -----------------------------------------------
		
	infile >> nN >> nF >> nE; 
	if (nN != nNodes) cout << "number of IBM nodes is not consistent with ibm input file = " << nN << endl;
	if (nF != nFaces) cout << "number of IBM faces is not consistent with ibm input file = " << nF << endl;
	if (nE != nEdges) cout << "number of IBM edges is not consistent with ibm input file = " << nE << endl;
	
	// -----------------------------------------------
	// read node positions:
	// -----------------------------------------------
	
	for (int i=0; i<nNodes; i++) {
		infile >> r[i].x >> r[i].y >> r[i].z;
	}
	
	// -----------------------------------------------
	// read face vertices:
	// -----------------------------------------------
		
	for (int i=0; i<nFaces; i++) {
		infile >> faces[i].v0 >> faces[i].v1 >> faces[i].v2;
		faces[i].v0 -= 1;     // adjust from 1-based to 0-based indexing
		faces[i].v1 -= 1;     // " "
		faces[i].v2 -= 1;     // " " 
		faces[i].cellID = 0;  // assume this is the first cell
	}
	
	// -----------------------------------------------
	// read edge vertices:
	// -----------------------------------------------
	
	for (int i=0; i<nEdges; i++) {
		infile >> edges[i].v0 >> edges[i].v1;
		edges[i].v0 -= 1;  // adjust from 1-based to 0-based indexing
		edges[i].v1 -= 1;  // " "
	}
	
	// -----------------------------------------------
	// read edge faces:
	// -----------------------------------------------
	
	for (int i=0; i<nEdges; i++) {
		infile >> edges[i].f0 >> edges[i].f1;
		edges[i].f0 -= 1;  // adjust from 1-based to 0-based indexing
		edges[i].f1 -= 1;  // " "
	}
	
	// -----------------------------------------------
	// close file:
	// -----------------------------------------------
	
	infile.close();
	
}