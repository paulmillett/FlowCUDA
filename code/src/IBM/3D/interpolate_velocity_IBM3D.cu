
# include "interpolate_velocity_IBM3D.cuh"
# include <stdio.h>

// --------------------------------------------------------
// IBM3D kernel to interpolate LBM velocity to IBM node:
// --------------------------------------------------------

__global__ void interpolate_velocity_IBM3D(
	float* x,
	float* y,
	float* z,
	float* vx,
	float* vy,
	float* vz,
	float* uLBM,
	float* vLBM,
	float* wLBM,
	int Nx,
	int Ny,
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
		vz[i] = 0.0;
				
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
					vx[i] += del*uLBM[ndx];
					vy[i] += del*vLBM[ndx];
					vz[i] += del*wLBM[ndx];
				}
			}		
		}		
	}	
}
