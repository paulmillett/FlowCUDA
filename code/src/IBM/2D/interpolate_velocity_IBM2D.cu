
# include "interpolate_velocity_IBM2D.cuh"
# include <stdio.h>

// --------------------------------------------------------
// IBM2D kernel to interpolate LBM velocity to IBM node:
// --------------------------------------------------------

__global__ void interpolate_velocity_IBM2D(
	float* x,
	float* y,
	float* vx,
	float* vy,
	float* uLBM,
	float* vLBM,
	int Nx,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nNodes) {
		
		// --------------------------------------
		// zero out velocities for node "i"
		// --------------------------------------
		
		vx[i] = 0.0;
		vy[i] = 0.0;
				
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
				vx[i] += del*uLBM[ndx];
				vy[i] += del*vLBM[ndx];
			}
		}		
	}
	
}
