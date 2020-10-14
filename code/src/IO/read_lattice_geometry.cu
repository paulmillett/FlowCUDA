
# include "read_lattice_geometry.cuh"
# include <iostream>
# include <iomanip>
# include <fstream>
# include <sstream>
# include <stdlib.h>
using namespace std;  



// --------------------------------------------------------
// Read lattice information from file:
// --------------------------------------------------------

void read_lattice_geometry_D3Q19(int nVoxels, int* x, int* y, int* z,
                                 int* voxelType, int* nList)
{
	
	// -----------------------------------------------
	// variables for reading:
	// -----------------------------------------------
	
	int N;
	string stencil;
	
	// -----------------------------------------------
	// open file:
	// -----------------------------------------------
	
	ifstream infile;
	infile.open("lattice_geometry.dat", ios::in);
	
	// -----------------------------------------------
	// read header:
	// -----------------------------------------------
	
	infile >> stencil;
	infile >> N;	
	if (stencil != "D3Q19") cout << "stencil is not D3Q19 in 'lattice_geometry.dat' ";
	if (N != nVoxels) cout << "number of voxels in 'lattice_geometry.dat' is not consistent with 'input.dat' ";
	
	// -----------------------------------------------
	// read voxel positions and types:
	// -----------------------------------------------
	
	for (int i=0; i<nVoxels; i++) {
		infile >> x[i] >> y[i] >> z[i] >> voxelType[i];
	}
	
	// -----------------------------------------------
	// read nList[] values:
	// -----------------------------------------------
	
	for (int i=0; i<nVoxels*19; i++) {
		infile >> nList[i];
	}
	
	// -----------------------------------------------
	// close file:
	// -----------------------------------------------
	
	infile.close();
	
}



// --------------------------------------------------------
// Read lattice information from file (w/o nList[]):
// --------------------------------------------------------

void read_lattice_geometry_D3Q19(int nVoxels, int* x, int* y, int* z,
                                 int* voxelType)
{
	
	// -----------------------------------------------
	// variables for reading:
	// -----------------------------------------------
	
	int N;
	string stencil;
	
	// -----------------------------------------------
	// open file:
	// -----------------------------------------------
	
	ifstream infile;
	infile.open("lattice_geometry.dat", ios::in);
	
	// -----------------------------------------------
	// read header:
	// -----------------------------------------------
	
	infile >> stencil;
	infile >> N;	
	if (stencil != "D3Q19") cout << "stencil is not D3Q19 in 'lattice_geometry.dat' ";
	if (N != nVoxels) cout << "number of voxels in 'lattice_geometry.dat' is not consistent with 'input.dat' ";
	
	// -----------------------------------------------
	// read voxel positions and types:
	// -----------------------------------------------
	
	for (int i=0; i<nVoxels; i++) {
		infile >> x[i] >> y[i] >> z[i] >> voxelType[i];
	}
		
	// -----------------------------------------------
	// close file:
	// -----------------------------------------------
	
	infile.close();
	
}



// --------------------------------------------------------
// Read lattice information from file (w/o nList[] & voxelType[]):
// --------------------------------------------------------

void read_lattice_geometry_D3Q19(int nVoxels, int* x, int* y, int* z)
{
	
	// -----------------------------------------------
	// variables for reading:
	// -----------------------------------------------
	
	int N;
	string stencil;
	
	// -----------------------------------------------
	// open file:
	// -----------------------------------------------
	
	ifstream infile;
	infile.open("lattice_geometry.dat", ios::in);
	
	// -----------------------------------------------
	// read header:
	// -----------------------------------------------
	
	infile >> stencil;
	infile >> N;	
	if (stencil != "D3Q19") cout << "stencil is not D3Q19 in 'lattice_geometry.dat' ";
	if (N != nVoxels) cout << "number of voxels in 'lattice_geometry.dat' is not consistent with 'input.dat' ";
	
	// -----------------------------------------------
	// read voxel positions and types:
	// -----------------------------------------------
	
	for (int i=0; i<nVoxels; i++) {
		infile >> x[i] >> y[i] >> z[i];
	}
		
	// -----------------------------------------------
	// close file:
	// -----------------------------------------------
	
	infile.close();
	
}