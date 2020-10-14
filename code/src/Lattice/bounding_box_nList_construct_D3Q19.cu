
# include "bounding_box_nList_construct_D3Q19.cuh"
# include <iostream>
using namespace std; 




// --------------------------------------------------------
// flattened 1D index
// --------------------------------------------------------

int box_index(int x, int y, int z, int Nx, int Ny) {
	return z*Nx*Ny + y*Nx + x;
}


			
// --------------------------------------------------------
// Assign nList[] by creating a bounding box of
// integers storing the voxel indices (or -1 if
// outside the geometry)
// --------------------------------------------------------

void bounding_box_nList_construct_D3Q19(int nVoxels,
	int* x,
	int* y,
	int* z,
	int* nList)
                                         
{
	
	// -------------------------------------
	// figure out how many cells should 
	// exist in each direction:
	// -------------------------------------
		
	int xLo = 100000;   // an arbitrary high #
	int yLo = 100000;
	int zLo = 100000;
	int xHi = -100000;  // an arbitrary low #		
	int yHi = -100000;		
	int zHi = -100000;		
	for (int i=0; i<nVoxels; i++) {
		if (x[i] < xLo) xLo = x[i];
		if (y[i] < yLo) yLo = y[i];
		if (z[i] < zLo) zLo = z[i];
		if (x[i] > xHi) xHi = x[i];
		if (y[i] > yHi) yHi = y[i];
		if (z[i] > zHi) zHi = z[i];
	}
	int NX = xHi - xLo + 1;  // range in x-dir
	int NY = yHi - yLo + 1;  // range in y-dir
	int NZ = zHi - yLo + 1;  // range in z-dir
	int nCellX = NX + 2;  // add outer buffer layer around
	int nCellY = NY + 2;  // geometry (both lower and upper)
	int nCellZ = NZ + 2;
	int nCell = nCellX*nCellY*nCellZ;
			
	// -------------------------------------
	// establish 'indexBox' array:
	// -------------------------------------
	
	int* indexBox = (int*)malloc(nCell*sizeof(int));	
	for (int i=0; i<nCell; i++) {
		indexBox[i] = -1;
	}
					
	// -------------------------------------
	// populate indexBox[] with non-zero
	// values:
	// -------------------------------------
	
	for (int i=0; i<nVoxels; i++) {
		int xi = x[i] - xLo + 1;
		int yi = y[i] - yLo + 1;
		int zi = z[i] - zLo + 1;
		int ndx = box_index(xi,yi,zi,nCellX,nCellY);
		indexBox[ndx] = i;
	}
	
	// -------------------------------------
	// initialize nList[] to values of -1:
	// -------------------------------------
	
	for (int i=0; i<nVoxels*19; i++) {
		nList[i] = -1;
	}
		
	// -------------------------------------
	// determine the neighboring index
	// for each voxel:
	// -------------------------------------
		
	for (int i=0; i<nVoxels; i++) {
		int xi = x[i] - xLo + 1;
		int yi = y[i] - yLo + 1;
		int zi = z[i] - zLo + 1;
		int offst = i*19;
		nList[offst+0]  = indexBox[box_index(xi,yi,zi,nCellX,nCellY)];
		nList[offst+1]  = indexBox[box_index(xi+1,yi,zi,nCellX,nCellY)];
		nList[offst+2]  = indexBox[box_index(xi-1,yi,zi,nCellX,nCellY)];
		nList[offst+3]  = indexBox[box_index(xi,yi+1,zi,nCellX,nCellY)];
		nList[offst+4]  = indexBox[box_index(xi,yi-1,zi,nCellX,nCellY)];
		nList[offst+5]  = indexBox[box_index(xi,yi,zi+1,nCellX,nCellY)];
		nList[offst+6]  = indexBox[box_index(xi,yi,zi-1,nCellX,nCellY)];
		nList[offst+7]  = indexBox[box_index(xi+1,yi+1,zi,nCellX,nCellY)];
		nList[offst+8]  = indexBox[box_index(xi-1,yi-1,zi,nCellX,nCellY)];
		nList[offst+9]  = indexBox[box_index(xi+1,yi,zi+1,nCellX,nCellY)];
		nList[offst+10] = indexBox[box_index(xi-1,yi,zi-1,nCellX,nCellY)];
		nList[offst+11] = indexBox[box_index(xi,yi+1,zi+1,nCellX,nCellY)];
		nList[offst+12] = indexBox[box_index(xi,yi-1,zi-1,nCellX,nCellY)];
		nList[offst+13] = indexBox[box_index(xi+1,yi-1,zi,nCellX,nCellY)];
		nList[offst+14] = indexBox[box_index(xi-1,yi+1,zi,nCellX,nCellY)];
		nList[offst+15] = indexBox[box_index(xi+1,yi,zi-1,nCellX,nCellY)];
		nList[offst+16] = indexBox[box_index(xi-1,yi,zi+1,nCellX,nCellY)];
		nList[offst+17] = indexBox[box_index(xi,yi+1,zi-1,nCellX,nCellY)];
		nList[offst+18] = indexBox[box_index(xi,yi-1,zi+1,nCellX,nCellY)];		 
	}
	
	// -------------------------------------
	// deallocate indexBox[]:
	// -------------------------------------
	
	free(indexBox);
		
}
