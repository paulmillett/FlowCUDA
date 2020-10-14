
# include "extrapolate_force_IBM2D.cuh"
# include <stdio.h>

// --------------------------------------------------------
// IBM2D kernel to extrapolate IBM node force to LBM lattice
// --------------------------------------------------------

__global__ void extrapolate_force_IBM2D(
	float* x,
	float* y,
	float* fx,
	float* fy,
	float* fxLBM,
	float* fyLBM,
	int Nx,
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
		
		// --------------------------------------
		// loop over footprint
		// --------------------------------------
		
		for (int jj=j0; jj<=j0+1; jj++) {
			for (int ii=i0; ii<=i0+1; ii++) {
				int ndx = jj*Nx + ii;
				float rx = x[i] - float(ii);
				float ry = y[i] - float(jj);
				float del = (1.0-abs(rx))*(1.0-abs(ry));
				//fxLBM[ndx] += del*fx[i];
				//fyLBM[ndx] += del*fy[i];
				atomicAdd(&fxLBM[ndx],del*fx[i]);
				atomicAdd(&fyLBM[ndx],del*fy[i]);
			}
		}		
	}
	
}
