
# include "lattice_builders_D2Q9.cuh"
# include <iostream>



// --------------------------------------------------------
// Build a rectilinear simulation box:
// NOTE: Here, we assume periodic boundaries in all
//       directions!
// --------------------------------------------------------

void build_box_lattice_D2Q9(int nVoxels,
                            int Nx, int Ny,
                            int* voxelType, int* nList)
{
	
	// -----------------------------------------------
	// make sure dimensions match array sizes:
	// -----------------------------------------------
	
	if (Nx*Ny != nVoxels) {
		std::cout << "box size does not match nVoxels" << std::endl;
		return;
	}
	
	// -----------------------------------------------
	// build voxelType[] array...
	// -----------------------------------------------
	
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			int ndx = j*Nx+i;
			voxelType[ndx] = 0;		
		}
	}	
		
	// -----------------------------------------------
	// build the nList[] array:
	// (note: periodic boundaries are assumed)
	// -----------------------------------------------
		
	int Q = 9;
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			int ndx = voxel_index(i,j,Nx,Ny); 
			int offst = Q*ndx;
			int ip1 = (i+1)%Nx;			
			int jp1 = (j+1)%Ny;
			int im1 = (Nx+i-1)%Nx;
			int jm1 = (Ny+j-1)%Ny;	
			nList[offst+0] = ndx;
			nList[offst+1] = voxel_index(ip1, j,   Nx, Ny);	
			nList[offst+2] = voxel_index(i,   jp1, Nx, Ny);	
			nList[offst+3] = voxel_index(im1, j,   Nx, Ny);  	
			nList[offst+4] = voxel_index(i,   jm1, Nx, Ny);	
			nList[offst+5] = voxel_index(ip1, jp1, Nx, Ny);	
			nList[offst+6] = voxel_index(im1, jp1, Nx, Ny);	
			nList[offst+7] = voxel_index(im1, jm1, Nx, Ny);	
			nList[offst+8] = voxel_index(ip1, jm1, Nx, Ny);			
		}
	}
	
}



// --------------------------------------------------------
// Build a rectilinear simulation box:
// NOTE: Here, we assume shear flow in the x-direction,
//       so we have bounce-back conditions on the y-walls
// --------------------------------------------------------

void build_box_lattice_shear_D2Q9(int nVoxels,
                                  int Nx, int Ny,
                                  int* voxelType, int* nList)
{
	
	// -----------------------------------------------
	// make sure dimensions match array sizes:
	// -----------------------------------------------
	
	if (Nx*Ny != nVoxels) {
		std::cout << "box size does not match nVoxels" << std::endl;
		return;
	}
	
	// -----------------------------------------------
	// build voxelType[] array...
	// -----------------------------------------------
	
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			int ndx = j*Nx+i;
			voxelType[ndx] = 0;		
		}
	}	
		
	// -----------------------------------------------
	// build the nList[] array:
	// (note: periodic boundaries are assumed)
	// -----------------------------------------------
		
	int Q = 9;
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			int ndx = voxel_index(i,j,Nx,Ny); 
			int offst = Q*ndx;
			int ip1 = (i+1)%Nx;			
			int jp1 = j+1;         //(j+1)%Ny;
			int im1 = (Nx+i-1)%Nx;
			int jm1 = j-1;         //(Ny+j-1)%Ny;	
			nList[offst+0] = ndx;
			nList[offst+1] = voxel_index(ip1, j,   Nx, Ny);	
			nList[offst+2] = voxel_index(i,   jp1, Nx, Ny);	
			nList[offst+3] = voxel_index(im1, j,   Nx, Ny);  	
			nList[offst+4] = voxel_index(i,   jm1, Nx, Ny);	
			nList[offst+5] = voxel_index(ip1, jp1, Nx, Ny);	
			nList[offst+6] = voxel_index(im1, jp1, Nx, Ny);	
			nList[offst+7] = voxel_index(im1, jm1, Nx, Ny);	
			nList[offst+8] = voxel_index(ip1, jm1, Nx, Ny);			
		}
	}
	
}



// --------------------------------------------------------
// Build a rectilinear simulation box:
// NOTE: Here, we assume an inlet and outlet in one of the
//       directions!
// --------------------------------------------------------

void build_box_lattice_D2Q9(int nVoxels, int flowDir,
                            int Nx, int Ny,
                            int xLBC, int xUBC,
                            int yLBC, int yUBC,						    
						    int* voxelType, int* nList)
{
	
	// -----------------------------------------------
	// make sure dimensions match array sizes:
	// -----------------------------------------------
	
	if (Nx*Ny != nVoxels) {
		std::cout << "box size does not match nVoxels" << std::endl;
		return;
	}
	
	// -----------------------------------------------
	// build voxelType[] array...
	// if flow is along x-direction:
	// -----------------------------------------------
	
	if (flowDir == 0) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = j*Nx + i;							
				voxelType[ndx] = 0;                  // internal				
				if (i==0)    voxelType[ndx] = xLBC;  // west boundary
				if (i==Nx-1) voxelType[ndx] = xUBC;  // east boundary
				if (j==0 || j==Ny-1) voxelType[ndx] = 0;  // side boundary
			}
		}	
	} 
	
	// -----------------------------------------------
	// if flow is along y-direction:
	// -----------------------------------------------
	
	if (flowDir == 1) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = j*Nx + i;				
				voxelType[ndx] = 0;                  // internal					
				if (j==0)    voxelType[ndx] = yLBC;  // south boundary
				if (j==Ny-1) voxelType[ndx] = yUBC;  // north boundary
				if (i==0 || i==Nx-1) voxelType[ndx] = 0;  // side boundary
			}
		}	
	}
		
	// -----------------------------------------------
	// build the nList[] array:
	// (note: periodic boundaries are NOT considered)
	// -----------------------------------------------
	
	int Q = 9;
	for (int j=0; j<Ny; j++) {
		for (int i=0; i<Nx; i++) {
			int ndx = j*Nx + i;
			int offst = Q*ndx;
			int ip1 = i+1;  //(i+1)%Nx;			
			int jp1 = j+1;  //(j+1)%Ny;
			int im1 = i-1;  //(Nx+i-1)%Nx;
			int jm1 = j-1;  //(Ny+j-1)%Ny;			
			nList[offst+0] = ndx;
			nList[offst+1] = voxel_index(ip1, j,   Nx, Ny);	
			nList[offst+2] = voxel_index(i,   jp1, Nx, Ny);	
			nList[offst+3] = voxel_index(im1, j,   Nx, Ny);  	
			nList[offst+4] = voxel_index(i,   jm1, Nx, Ny);	
			nList[offst+5] = voxel_index(ip1, jp1, Nx, Ny);	
			nList[offst+6] = voxel_index(im1, jp1, Nx, Ny);	
			nList[offst+7] = voxel_index(im1, jm1, Nx, Ny);	
			nList[offst+8] = voxel_index(ip1, jm1, Nx, Ny);							
		}
	}
	
}



// --------------------------------------------------------
// Compute index of voxel given i,j indices:
// --------------------------------------------------------

int voxel_index(int i, int j, int Nx, int Ny) 
{
	if (i < 0 || i > Nx-1 ||
		j < 0 || j > Ny-1)
	{
		return -1;
	}
	else {
		return j*Nx + i;
	}	
}

