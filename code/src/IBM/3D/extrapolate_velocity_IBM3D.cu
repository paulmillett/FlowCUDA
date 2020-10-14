
# include "extrapolate_force_IBM3D.cuh"
# include <stdio.h>

// --------------------------------------------------------
// IBM3D kernel to extrapolate IBM node velocity to LBM lattice
// --------------------------------------------------------

__global__ void extrapolate_velocity_IBM3D(
	float* x,
	float* y,
	float* z,
	float* vx,
	float* vy,
	float* vz,
	float* uIBvox,
	float* vIBvox,
	float* wIBvox,
	float* weight,
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
					float del = sqrt(rx*rx + ry*ry + rz*rz);
					atomicAdd(&uIBvox[ndx],del*vx[i]);
					atomicAdd(&vIBvox[ndx],del*vy[i]);
					atomicAdd(&wIBvox[ndx],del*vz[i]);
					atomicAdd(&weight[ndx],del);
				}
			}		
		}
		
	}
	
}
