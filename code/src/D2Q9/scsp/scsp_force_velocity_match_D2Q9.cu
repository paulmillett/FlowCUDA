
# include "scsp_force_velocity_match_D2Q9.cuh"
# include <stdio.h>

// --------------------------------------------------------
// D2Q9 kernel to calculate the necessary force to 
// adjust the velocity from (u,v) to (uIBvox,vIBvox): 
// --------------------------------------------------------

__global__ void scsp_force_velocity_match_D2Q9(
	float* fx,
	float* fy,
	float* u,
	float* v,
	float* uIBvox,
	float* vIBvox,
	float* weights,
	float* rho,		
	int nVoxels)
{
	
	// -----------------------------------------------
	// define voxel:
	// -----------------------------------------------
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	
	if (i < nVoxels) {			
		
		// --------------------------------------------------		
		// Only do this if weights[i] is > 0:
		// --------------------------------------------------
		
		if (weights[i] > 0.0) {			
			uIBvox[i] /= weights[i];  // weighted average
			vIBvox[i] /= weights[i];  // "              "		
			fx[i] = (uIBvox[i] - u[i])*2.0*rho[i];
			fy[i] = (vIBvox[i] - v[i])*2.0*rho[i];			
		}
	}
}