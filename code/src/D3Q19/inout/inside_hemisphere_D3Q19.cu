
# include "inside_hemisphere_D3Q19.cuh"



// --------------------------------------------------------
// Kernel to determine if voxels are inside hemisphere:
// --------------------------------------------------------

__global__ void inside_hemisphere_D3Q19(float* weights, int* inout, 
	int Nx, int Ny, int Nz,	int nVoxels)
{	
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {	
		// determine (i,j,k) coordinates of voxel:
		int kk = i/(Nx*Ny);
		int jj = (i-kk*Nx*Ny)/Nx;
		int ii = i - Nx*jj - Nx*Ny*kk;
		// if kk == 0, loop up the z-direction:
		if (kk == 0) {
			int cross = 0;
			for (int k=0; k<Nz; k++) {
				int ndx = k*Nx*Ny + jj*Nx + ii;
				if (weights[ndx] > 0.0) cross = 1;
				if (cross == 1) inout[ndx] = 1;
				if (cross == 0) inout[ndx] = 0;
			}
		}		
	}	
}