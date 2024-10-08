
# include "lattice_builders_D3Q19.cuh"
# include <iostream>




// --------------------------------------------------------
// Build a rectilinear simulation box:
// NOTE: Here, we assume periodic boundaries in all
//       directions!
// --------------------------------------------------------

void build_box_lattice_D3Q19(int nVoxels,
                             int Nx, int Ny, int Nz,
                             int* voxelType, int* nList)
{
	
	// -----------------------------------------------
	// make sure dimensions match array sizes:
	// -----------------------------------------------
	
	if (Nx*Ny*Nz != nVoxels) {
		std::cout << "box size does not match nVoxels" << std::endl;
		return;
	}
	
	// -----------------------------------------------
	// build voxelType[] array...
	// -----------------------------------------------
	
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = k*Nx*Ny + j*Nx + i;
				voxelType[ndx] = 0;		
			}
		}	
	}	
		
	// -----------------------------------------------
	// build the nList[] array:
	// (note: periodic boundaries are assumed)
	// -----------------------------------------------
		
	int Q = 19;
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = voxel_index(i,j,k,Nx,Ny,Nz); 
				int offst = Q*ndx;
				int ip1 = (i+1)%Nx;			
				int jp1 = (j+1)%Ny;
				int kp1 = (k+1)%Nz;
				int im1 = (Nx+i-1)%Nx;
				int jm1 = (Ny+j-1)%Ny;
				int km1 = (Nz+k-1)%Nz;
				nList[offst+0]  = ndx;
				nList[offst+1]  = voxel_index(ip1, j,   k,   Nx, Ny, Nz);	
				nList[offst+2]  = voxel_index(im1, j,   k,   Nx, Ny, Nz);	
				nList[offst+3]  = voxel_index(i,   jp1, k,   Nx, Ny, Nz);	
				nList[offst+4]  = voxel_index(i,   jm1, k,   Nx, Ny, Nz);	
				nList[offst+5]  = voxel_index(i,   j,   kp1, Nx, Ny, Nz);	
				nList[offst+6]  = voxel_index(i,   j,   km1, Nx, Ny, Nz);
				nList[offst+7]  = voxel_index(ip1, jp1, k,   Nx, Ny, Nz);	
				nList[offst+8]  = voxel_index(im1, jm1, k,   Nx, Ny, Nz);
				nList[offst+9]  = voxel_index(ip1, j,   kp1, Nx, Ny, Nz);
				nList[offst+10] = voxel_index(im1, j,   km1, Nx, Ny, Nz);
				nList[offst+11] = voxel_index(i,   jp1, kp1, Nx, Ny, Nz);
				nList[offst+12] = voxel_index(i,   jm1, km1, Nx, Ny, Nz);
				nList[offst+13] = voxel_index(ip1, jm1, k,   Nx, Ny, Nz);
				nList[offst+14] = voxel_index(im1, jp1, k,   Nx, Ny, Nz);
				nList[offst+15] = voxel_index(ip1, j,   km1, Nx, Ny, Nz);
				nList[offst+16] = voxel_index(im1, j,   kp1, Nx, Ny, Nz);
				nList[offst+17] = voxel_index(i,   jp1, km1, Nx, Ny, Nz);
				nList[offst+18] = voxel_index(i,   jm1, kp1, Nx, Ny, Nz);			
			}
		}
	}	
	
}



// --------------------------------------------------------
// Build a rectilinear simulation box:
// NOTE: Here, we assume shear flow in the x-direction,
//       so we have bounce-back conditions on the y-walls,
//       and periodic conditions for the x- and z-dir's.
// --------------------------------------------------------

void build_box_lattice_shear_D3Q19(int nVoxels,
                                   int Nx, int Ny, int Nz,
                                   int* voxelType, int* nList)
{
	
	// -----------------------------------------------
	// make sure dimensions match array sizes:
	// -----------------------------------------------
	
	if (Nx*Ny*Nz != nVoxels) {
		std::cout << "box size does not match nVoxels" << std::endl;
		return;
	}
	
	// -----------------------------------------------
	// build voxelType[] array...
	// -----------------------------------------------
	
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = k*Nx*Ny + j*Nx + i;
				voxelType[ndx] = 0;		
			}
		}	
	}	
		
	// -----------------------------------------------
	// build the nList[] array:
	// (note: periodic boundaries only along x and z)
	// -----------------------------------------------
		
	int Q = 19;
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = voxel_index(i,j,k,Nx,Ny,Nz); 
				int offst = Q*ndx;
				int ip1 = (i+1)%Nx;			
				int jp1 = j+1;
				int kp1 = (k+1)%Nz;
				int im1 = (Nx+i-1)%Nx;
				int jm1 = j-1;
				int km1 = (Nz+k-1)%Nz;
				nList[offst+0]  = ndx;
				nList[offst+1]  = voxel_index(ip1, j,   k,   Nx, Ny, Nz);	
				nList[offst+2]  = voxel_index(im1, j,   k,   Nx, Ny, Nz);	
				nList[offst+3]  = voxel_index(i,   jp1, k,   Nx, Ny, Nz);	
				nList[offst+4]  = voxel_index(i,   jm1, k,   Nx, Ny, Nz);	
				nList[offst+5]  = voxel_index(i,   j,   kp1, Nx, Ny, Nz);	
				nList[offst+6]  = voxel_index(i,   j,   km1, Nx, Ny, Nz);
				nList[offst+7]  = voxel_index(ip1, jp1, k,   Nx, Ny, Nz);	
				nList[offst+8]  = voxel_index(im1, jm1, k,   Nx, Ny, Nz);
				nList[offst+9]  = voxel_index(ip1, j,   kp1, Nx, Ny, Nz);
				nList[offst+10] = voxel_index(im1, j,   km1, Nx, Ny, Nz);
				nList[offst+11] = voxel_index(i,   jp1, kp1, Nx, Ny, Nz);
				nList[offst+12] = voxel_index(i,   jm1, km1, Nx, Ny, Nz);
				nList[offst+13] = voxel_index(ip1, jm1, k,   Nx, Ny, Nz);
				nList[offst+14] = voxel_index(im1, jp1, k,   Nx, Ny, Nz);
				nList[offst+15] = voxel_index(ip1, j,   km1, Nx, Ny, Nz);
				nList[offst+16] = voxel_index(im1, j,   kp1, Nx, Ny, Nz);
				nList[offst+17] = voxel_index(i,   jp1, km1, Nx, Ny, Nz);
				nList[offst+18] = voxel_index(i,   jm1, kp1, Nx, Ny, Nz);			
			}
		}
	}	
	
}



// --------------------------------------------------------
// Build a rectilinear simulation box:
// NOTE: Here, we assume plane poiseuille flow in the x-direction,
//       periodic conditions in the y-direction, and bounce-back
//       conditions in the z-direction.
// --------------------------------------------------------

void build_box_lattice_slit_D3Q19(int nVoxels,
                                  int Nx, int Ny, int Nz,
                                  int* voxelType, int* nList)
{
	
	// -----------------------------------------------
	// make sure dimensions match array sizes:
	// -----------------------------------------------
	
	if (Nx*Ny*Nz != nVoxels) {
		std::cout << "box size does not match nVoxels" << std::endl;
		return;
	}
	
	// -----------------------------------------------
	// build voxelType[] array...
	// -----------------------------------------------
	
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = k*Nx*Ny + j*Nx + i;
				voxelType[ndx] = 0;		
			}
		}	
	}	
		
	// -----------------------------------------------
	// build the nList[] array:
	// (note: periodic boundaries only along x and z)
	// -----------------------------------------------
		
	int Q = 19;
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = voxel_index(i,j,k,Nx,Ny,Nz); 
				int offst = Q*ndx;
				int ip1 = (i+1)%Nx;			
				int jp1 = (j+1)%Ny;
				int kp1 = k+1;
				int im1 = (Nx+i-1)%Nx;
				int jm1 = (Ny+j-1)%Ny; 
				int km1 = k-1;
				nList[offst+0]  = ndx;
				nList[offst+1]  = voxel_index(ip1, j,   k,   Nx, Ny, Nz);	
				nList[offst+2]  = voxel_index(im1, j,   k,   Nx, Ny, Nz);	
				nList[offst+3]  = voxel_index(i,   jp1, k,   Nx, Ny, Nz);	
				nList[offst+4]  = voxel_index(i,   jm1, k,   Nx, Ny, Nz);	
				nList[offst+5]  = voxel_index(i,   j,   kp1, Nx, Ny, Nz);	
				nList[offst+6]  = voxel_index(i,   j,   km1, Nx, Ny, Nz);
				nList[offst+7]  = voxel_index(ip1, jp1, k,   Nx, Ny, Nz);	
				nList[offst+8]  = voxel_index(im1, jm1, k,   Nx, Ny, Nz);
				nList[offst+9]  = voxel_index(ip1, j,   kp1, Nx, Ny, Nz);
				nList[offst+10] = voxel_index(im1, j,   km1, Nx, Ny, Nz);
				nList[offst+11] = voxel_index(i,   jp1, kp1, Nx, Ny, Nz);
				nList[offst+12] = voxel_index(i,   jm1, km1, Nx, Ny, Nz);
				nList[offst+13] = voxel_index(ip1, jm1, k,   Nx, Ny, Nz);
				nList[offst+14] = voxel_index(im1, jp1, k,   Nx, Ny, Nz);
				nList[offst+15] = voxel_index(ip1, j,   km1, Nx, Ny, Nz);
				nList[offst+16] = voxel_index(im1, j,   kp1, Nx, Ny, Nz);
				nList[offst+17] = voxel_index(i,   jp1, km1, Nx, Ny, Nz);
				nList[offst+18] = voxel_index(i,   jm1, kp1, Nx, Ny, Nz);			
			}
		}
	}	
	
}



// --------------------------------------------------------
// Build a rectilinear simulation box:
// NOTE: Here, we assume channel flow in the x-direction,
//       so we have bounce-back conditions on the y-walls
//       and z-walls, and periodic conditions for the x-dir
// --------------------------------------------------------

void build_box_lattice_channel_D3Q19(int nVoxels,
                                     int Nx, int Ny, int Nz,
                                     int* voxelType, int* nList)
{
	
	// -----------------------------------------------
	// make sure dimensions match array sizes:
	// -----------------------------------------------
	
	if (Nx*Ny*Nz != nVoxels) {
		std::cout << "box size does not match nVoxels" << std::endl;
		return;
	}
	
	// -----------------------------------------------
	// build voxelType[] array...
	// -----------------------------------------------
	
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = k*Nx*Ny + j*Nx + i;
				voxelType[ndx] = 0;		
			}
		}	
	}	
		
	// -----------------------------------------------
	// build the nList[] array:
	// (note: periodic boundaries only along x and z)
	// -----------------------------------------------
		
	int Q = 19;
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = voxel_index(i,j,k,Nx,Ny,Nz); 
				int offst = Q*ndx;
				int ip1 = (i+1)%Nx;			
				int jp1 = j+1;
				int kp1 = k+1;
				int im1 = (Nx+i-1)%Nx;
				int jm1 = j-1;
				int km1 = k-1;
				nList[offst+0]  = ndx;
				nList[offst+1]  = voxel_index(ip1, j,   k,   Nx, Ny, Nz);	
				nList[offst+2]  = voxel_index(im1, j,   k,   Nx, Ny, Nz);	
				nList[offst+3]  = voxel_index(i,   jp1, k,   Nx, Ny, Nz);	
				nList[offst+4]  = voxel_index(i,   jm1, k,   Nx, Ny, Nz);	
				nList[offst+5]  = voxel_index(i,   j,   kp1, Nx, Ny, Nz);	
				nList[offst+6]  = voxel_index(i,   j,   km1, Nx, Ny, Nz);
				nList[offst+7]  = voxel_index(ip1, jp1, k,   Nx, Ny, Nz);	
				nList[offst+8]  = voxel_index(im1, jm1, k,   Nx, Ny, Nz);
				nList[offst+9]  = voxel_index(ip1, j,   kp1, Nx, Ny, Nz);
				nList[offst+10] = voxel_index(im1, j,   km1, Nx, Ny, Nz);
				nList[offst+11] = voxel_index(i,   jp1, kp1, Nx, Ny, Nz);
				nList[offst+12] = voxel_index(i,   jm1, km1, Nx, Ny, Nz);
				nList[offst+13] = voxel_index(ip1, jm1, k,   Nx, Ny, Nz);
				nList[offst+14] = voxel_index(im1, jp1, k,   Nx, Ny, Nz);
				nList[offst+15] = voxel_index(ip1, j,   km1, Nx, Ny, Nz);
				nList[offst+16] = voxel_index(im1, j,   kp1, Nx, Ny, Nz);
				nList[offst+17] = voxel_index(i,   jp1, km1, Nx, Ny, Nz);
				nList[offst+18] = voxel_index(i,   jm1, kp1, Nx, Ny, Nz);			
			}
		}
	}	
	
}



// --------------------------------------------------------
// Build a rectilinear simulation box:
// NOTE: Here, we assume an inlet and outlet in one of the
//       directions!
// --------------------------------------------------------

void build_box_lattice_D3Q19(int nVoxels, int flowDir,
                             int Nx, int Ny, int Nz,
                             int xLBC, int xUBC,
                             int yLBC, int yUBC,
							 int zLBC, int zUBC,
							 int* voxelType, int* nList)
{
	
	// -----------------------------------------------
	// make sure dimensions match array sizes:
	// -----------------------------------------------
	
	if (Nx*Ny*Nz != nVoxels) {
		std::cout << "box size does not match nVoxels" << std::endl;
		return;
	}
	
	// -----------------------------------------------
	// build voxelType[] array...
	// if flow is along x-direction:
	// -----------------------------------------------
	
	if (flowDir == 0) {
		for (int k=0; k<Nz; k++) {
			for (int j=0; j<Ny; j++) {
				for (int i=0; i<Nx; i++) {
					int ndx = k*Nx*Ny + j*Nx + i;							
					voxelType[ndx] = 0;                  // internal				
					if (i==0)    voxelType[ndx] = xLBC;  // west boundary
					if (i==Nx-1) voxelType[ndx] = xUBC;  // east boundary
					if (j==0 || j==Ny-1) voxelType[ndx] = 0;  // side boundary
					if (k==0 || k==Nz-1) voxelType[ndx] = 0;  // side boundary			
				}
			}	
		}	
	} 
	
	// -----------------------------------------------
	// if flow is along y-direction:
	// -----------------------------------------------
	
	if (flowDir == 1) {
		for (int k=0; k<Nz; k++) {
			for (int j=0; j<Ny; j++) {
				for (int i=0; i<Nx; i++) {
					int ndx = k*Nx*Ny + j*Nx + i;				
					voxelType[ndx] = 0;                  // internal					
					if (j==0)    voxelType[ndx] = yLBC;  // south boundary
					if (j==Ny-1) voxelType[ndx] = yUBC;  // north boundary
					if (i==0 || i==Nx-1) voxelType[ndx] = 0;  // side boundary
					if (k==0 || k==Nz-1) voxelType[ndx] = 0;  // side boundary				
				}
			}	
		}	
	}
	
	// -----------------------------------------------
	// if flow is along z-direction:
	// -----------------------------------------------
	
	if (flowDir == 2) {
		for (int k=0; k<Nz; k++) {
			for (int j=0; j<Ny; j++) {
				for (int i=0; i<Nx; i++) {
					int ndx = k*Nx*Ny + j*Nx + i;				
					voxelType[ndx] = 0;                  // internal				
					if (k==0)    voxelType[ndx] = zLBC;  // bottom boundary
					if (k==Nz-1) voxelType[ndx] = zUBC;  // top boundary
					if (i==0 || i==Nx-1) voxelType[ndx] = 0;  // side boundary
					if (j==0 || j==Ny-1) voxelType[ndx] = 0;  // side boundary				
				}
			}	
		}	
	}	
	
	// -----------------------------------------------
	// build the nList[] array:
	// (note: periodic boundaries are not considered)
	// -----------------------------------------------
	
	int Q = 19;
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = k*Nx*Ny + j*Nx + i;
				int offst = Q*ndx;
				int ip1 = i+1;  //(i+1)%Nx;			
				int jp1 = j+1;  //(j+1)%Ny;
				int kp1 = k+1;  //(k+1)%Nz;
				int im1 = i-1;  //(Nx+i-1)%Nx;
				int jm1 = j-1;  //(Ny+j-1)%Ny;
				int km1 = k-1;  //(Nz+k-1)%Nz;
				nList[offst+0]  = ndx;
				nList[offst+1]  = voxel_index(ip1, j,   k,   Nx, Ny, Nz);	
				nList[offst+2]  = voxel_index(im1, j,   k,   Nx, Ny, Nz);	
				nList[offst+3]  = voxel_index(i,   jp1, k,   Nx, Ny, Nz);	
				nList[offst+4]  = voxel_index(i,   jm1, k,   Nx, Ny, Nz);	
				nList[offst+5]  = voxel_index(i,   j,   kp1, Nx, Ny, Nz);	
				nList[offst+6]  = voxel_index(i,   j,   km1, Nx, Ny, Nz);
				nList[offst+7]  = voxel_index(ip1, jp1, k,   Nx, Ny, Nz);	
				nList[offst+8]  = voxel_index(im1, jm1, k,   Nx, Ny, Nz);
				nList[offst+9]  = voxel_index(ip1, j,   kp1, Nx, Ny, Nz);
				nList[offst+10] = voxel_index(im1, j,   km1, Nx, Ny, Nz);
				nList[offst+11] = voxel_index(i,   jp1, kp1, Nx, Ny, Nz);
				nList[offst+12] = voxel_index(i,   jm1, km1, Nx, Ny, Nz);
				nList[offst+13] = voxel_index(ip1, jm1, k,   Nx, Ny, Nz);
				nList[offst+14] = voxel_index(im1, jp1, k,   Nx, Ny, Nz);
				nList[offst+15] = voxel_index(ip1, j,   km1, Nx, Ny, Nz);
				nList[offst+16] = voxel_index(im1, j,   kp1, Nx, Ny, Nz);
				nList[offst+17] = voxel_index(i,   jp1, km1, Nx, Ny, Nz);
				nList[offst+18] = voxel_index(i,   jm1, kp1, Nx, Ny, Nz);				
			}
		}
	}
	
}



// --------------------------------------------------------
// Build a rectilinear simulation box:
// NOTE: Here, we assume periodic boundaries in all
//       directions, with stationary solid nodes!
// --------------------------------------------------------

void build_box_lattice_solid_walls_D3Q19(int nVoxels,
                                         int Nx, int Ny, int Nz,
                                         int* voxelType, int* solid, int* nList)
{
	
	// -----------------------------------------------
	// make sure dimensions match array sizes:
	// -----------------------------------------------
	
	if (Nx*Ny*Nz != nVoxels) {
		std::cout << "box size does not match nVoxels" << std::endl;
		return;
	}
	
	// -----------------------------------------------
	// build voxelType[] array...
	// -----------------------------------------------
	
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = k*Nx*Ny + j*Nx + i;
				voxelType[ndx] = 0;		
			}
		}	
	}	
		
	// -----------------------------------------------
	// build the nList[] array:
	// (note: periodic boundaries are assumed)
	// -----------------------------------------------
		
	int Q = 19;
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = voxel_index(i,j,k,Nx,Ny,Nz); 
				int offst = Q*ndx;
				int ip1 = (i+1)%Nx;			
				int jp1 = (j+1)%Ny;
				int kp1 = (k+1)%Nz;
				int im1 = (Nx+i-1)%Nx;
				int jm1 = (Ny+j-1)%Ny;
				int km1 = (Nz+k-1)%Nz;
				nList[offst+0]  = ndx;
				nList[offst+1]  = voxel_index_solid(ip1, j,   k,   Nx, Ny, Nz, solid);	
				nList[offst+2]  = voxel_index_solid(im1, j,   k,   Nx, Ny, Nz, solid);	
				nList[offst+3]  = voxel_index_solid(i,   jp1, k,   Nx, Ny, Nz, solid);	
				nList[offst+4]  = voxel_index_solid(i,   jm1, k,   Nx, Ny, Nz, solid);	
				nList[offst+5]  = voxel_index_solid(i,   j,   kp1, Nx, Ny, Nz, solid);	
				nList[offst+6]  = voxel_index_solid(i,   j,   km1, Nx, Ny, Nz, solid);
				nList[offst+7]  = voxel_index_solid(ip1, jp1, k,   Nx, Ny, Nz, solid);	
				nList[offst+8]  = voxel_index_solid(im1, jm1, k,   Nx, Ny, Nz, solid);
				nList[offst+9]  = voxel_index_solid(ip1, j,   kp1, Nx, Ny, Nz, solid);
				nList[offst+10] = voxel_index_solid(im1, j,   km1, Nx, Ny, Nz, solid);
				nList[offst+11] = voxel_index_solid(i,   jp1, kp1, Nx, Ny, Nz, solid);
				nList[offst+12] = voxel_index_solid(i,   jm1, km1, Nx, Ny, Nz, solid);
				nList[offst+13] = voxel_index_solid(ip1, jm1, k,   Nx, Ny, Nz, solid);
				nList[offst+14] = voxel_index_solid(im1, jp1, k,   Nx, Ny, Nz, solid);
				nList[offst+15] = voxel_index_solid(ip1, j,   km1, Nx, Ny, Nz, solid);
				nList[offst+16] = voxel_index_solid(im1, j,   kp1, Nx, Ny, Nz, solid);
				nList[offst+17] = voxel_index_solid(i,   jp1, km1, Nx, Ny, Nz, solid);
				nList[offst+18] = voxel_index_solid(i,   jm1, kp1, Nx, Ny, Nz, solid);			
			}
		}
	}	
	
}



// --------------------------------------------------------
// Build a rectilinear simulation box:
// NOTE: Here, we assume periodic boundaries in all
//       directions, with stationary solid nodes!
// --------------------------------------------------------

void build_box_lattice_slit_solid_walls_D3Q19(int nVoxels,
                                              int Nx, int Ny, int Nz,
                                              int* voxelType, int* solid, int* nList)
{
	
	// -----------------------------------------------
	// make sure dimensions match array sizes:
	// -----------------------------------------------
	
	if (Nx*Ny*Nz != nVoxels) {
		std::cout << "box size does not match nVoxels" << std::endl;
		return;
	}
	
	// -----------------------------------------------
	// build voxelType[] array...
	// -----------------------------------------------
	
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = k*Nx*Ny + j*Nx + i;
				voxelType[ndx] = 0;		
			}
		}	
	}	
		
	// -----------------------------------------------
	// build the nList[] array:
	// (note: periodic boundaries only along x and z)
	// -----------------------------------------------
		
	int Q = 19;
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j++) {
			for (int i=0; i<Nx; i++) {
				int ndx = voxel_index(i,j,k,Nx,Ny,Nz); 
				int offst = Q*ndx;
				int ip1 = (i+1)%Nx;			
				int jp1 = (j+1)%Ny;
				int kp1 = k+1;
				int im1 = (Nx+i-1)%Nx;
				int jm1 = (Ny+j-1)%Ny; 
				int km1 = k-1;
				nList[offst+0]  = ndx;
				nList[offst+1]  = voxel_index_solid_boundary(ip1, j,   k,   Nx, Ny, Nz, solid);	
				nList[offst+2]  = voxel_index_solid_boundary(im1, j,   k,   Nx, Ny, Nz, solid);	
				nList[offst+3]  = voxel_index_solid_boundary(i,   jp1, k,   Nx, Ny, Nz, solid);	
				nList[offst+4]  = voxel_index_solid_boundary(i,   jm1, k,   Nx, Ny, Nz, solid);	
				nList[offst+5]  = voxel_index_solid_boundary(i,   j,   kp1, Nx, Ny, Nz, solid);	
				nList[offst+6]  = voxel_index_solid_boundary(i,   j,   km1, Nx, Ny, Nz, solid);
				nList[offst+7]  = voxel_index_solid_boundary(ip1, jp1, k,   Nx, Ny, Nz, solid);	
				nList[offst+8]  = voxel_index_solid_boundary(im1, jm1, k,   Nx, Ny, Nz, solid);
				nList[offst+9]  = voxel_index_solid_boundary(ip1, j,   kp1, Nx, Ny, Nz, solid);
				nList[offst+10] = voxel_index_solid_boundary(im1, j,   km1, Nx, Ny, Nz, solid);
				nList[offst+11] = voxel_index_solid_boundary(i,   jp1, kp1, Nx, Ny, Nz, solid);
				nList[offst+12] = voxel_index_solid_boundary(i,   jm1, km1, Nx, Ny, Nz, solid);
				nList[offst+13] = voxel_index_solid_boundary(ip1, jm1, k,   Nx, Ny, Nz, solid);
				nList[offst+14] = voxel_index_solid_boundary(im1, jp1, k,   Nx, Ny, Nz, solid);
				nList[offst+15] = voxel_index_solid_boundary(ip1, j,   km1, Nx, Ny, Nz, solid);
				nList[offst+16] = voxel_index_solid_boundary(im1, j,   kp1, Nx, Ny, Nz, solid);
				nList[offst+17] = voxel_index_solid_boundary(i,   jp1, km1, Nx, Ny, Nz, solid);
				nList[offst+18] = voxel_index_solid_boundary(i,   jm1, kp1, Nx, Ny, Nz, solid);			
			}
		}
	}	
	
}



// --------------------------------------------------------
// Compute index of voxel given i,j,k indices (if any
// index lies outside the box, return '-1')
// --------------------------------------------------------

int voxel_index(int i, int j, int k, int Nx, int Ny, int Nz) 
{
	if (i < 0 || i > Nx-1 ||
		j < 0 || j > Ny-1 ||
		k < 0 || k > Nz-1) 
	{
		return -1;
	}
	else {
		return k*Nx*Ny + j*Nx + i;
	}
}



// --------------------------------------------------------
// Compute index of voxel given i,j,k indices (if solid[]
// is equal to one, return '-1').  This only works for PBC's.
// --------------------------------------------------------

int voxel_index_solid(int i, int j, int k, int Nx, int Ny, int Nz, int* solid) 
{
	if (solid[k*Nx*Ny + j*Nx + i] == 1) 
	{
		return -1;
	}
	else {
		return k*Nx*Ny + j*Nx + i;
	}
}



// --------------------------------------------------------
// Compute index of voxel given i,j,k indices (if solid[]
// is equal to one, return '-1').  (if any
// index lies outside the box, return '-1')
// --------------------------------------------------------

int voxel_index_solid_boundary(int i, int j, int k, int Nx, int Ny, int Nz, int* solid) 
{
	if (i < 0 || i > Nx-1 ||
		j < 0 || j > Ny-1 ||
		k < 0 || k > Nz-1) 
	{
		return -1;
	} 
	else if (solid[k*Nx*Ny + j*Nx + i] == 1) 
	{
		return -1;
	}
	else 
	{
		return k*Nx*Ny + j*Nx + i;
	}
}



