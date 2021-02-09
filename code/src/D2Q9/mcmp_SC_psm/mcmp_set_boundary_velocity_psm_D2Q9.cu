
# include "mcmp_set_boundary_velocity_psm_D2Q9.cuh"
# include <stdio.h>

// --------------------------------------------------------
// D2Q9 set velocity on the y=0 and y=Ny-1 boundaries: 
// --------------------------------------------------------

__global__ void mcmp_set_boundary_velocity_psm_D2Q9(float* rA,
										            float* rB,
										            float* FxA,
										            float* FxB,
										            float* FyA,
										            float* FyB,
										            float* u,
										            float* v,
													int* y,											        
											        int Ny,
										            int nVoxels) 
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		if (y[i] == 0 || y[i] == Ny-1) {
			float uBC = 0.05;
			float vBC = 0.00;
			float rTotal = rA[i] + rB[i];
			float fxBC = (uBC - u[i])*2.0*rTotal;
			float fyBC = (vBC - v[i])*2.0*rTotal;
			u[i] += 0.5*fxBC/rTotal;
			v[i] += 0.5*fyBC/rTotal;
			FxA[i] += fxBC*(rA[i]/rTotal);
			FxB[i] += fxBC*(rB[i]/rTotal);
			FyA[i] += fyBC*(rA[i]/rTotal);
			FyB[i] += fyBC*(rB[i]/rTotal);
		}		
	}
}