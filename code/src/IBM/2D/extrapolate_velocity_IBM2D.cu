
# include "extrapolate_velocity_IBM2D.cuh"
# include <stdio.h>

// --------------------------------------------------------
// IBM2D kernel to extrapolate IBM node velocity to LBM
// lattice
// --------------------------------------------------------

__global__ void extrapolate_velocity_IBM2D(
	float* x,
	float* y,
	float* vx,
	float* vy,
	float* uIBvox,
	float* vIBvox,
	float* weight,
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
				float del = sqrt(rx*rx + ry*ry); //(1.0-abs(rx))*(1.0-abs(ry));
				atomicAdd(&uIBvox[ndx],del*vx[i]);
				atomicAdd(&vIBvox[ndx],del*vy[i]);
				atomicAdd(&weight[ndx],del);
			}
		}		
	}
	
}
