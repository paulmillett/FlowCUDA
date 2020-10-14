
# include "extrapolate_force_IBM3D.cuh"
# include <stdio.h>

// --------------------------------------------------------
// IBM3D kernel to extrapolate IBM node force to LBM lattice
// --------------------------------------------------------

__global__ void extrapolate_force_IBM3D(
	float* x,
	float* y,
	float* z,
	float* fx,
	float* fy,
	float* fz,
	float* fxLBM,
	float* fyLBM,
	float* fzLBM,
	int Nx,
	int Ny,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nNodes) {
				
		// --------------------------------------
		// find nearest LBM voxel (rounded down)
		// --------------------------------------
		
		int i0 = int(floor(x[i]));
		int j0 = int(floor(y[i]));
		int k0 = int(floor(z[i]));
		
		// --------------------------------------
		// loop over footprint
		// --------------------------------------
		
		for (int kk=k0; kk<=k0+1; kk++) {
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {				
					int ndx = kk*Nx*Ny + jj*Nx + ii;
					float rx = x[i] - float(ii);
					float ry = y[i] - float(jj);
					float rz = z[i] - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					atomicAdd(&fxLBM[ndx],del*fx[i]);
					atomicAdd(&fyLBM[ndx],del*fy[i]);
					atomicAdd(&fzLBM[ndx],del*fz[i]);
				}
			}		
		}
		
	}
	
}
