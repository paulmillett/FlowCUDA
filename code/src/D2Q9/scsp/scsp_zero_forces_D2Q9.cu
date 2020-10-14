
# include "scsp_zero_forces_D2Q9.cuh"
# include <stdio.h>

// --------------------------------------------------------
// D2Q9 kernel to re-set the fluid forces to zero: 
// --------------------------------------------------------

__global__ void scsp_zero_forces_D2Q9(
	float* fx,
	float* fy,
	int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {			
		fx[i] = 0.0;
		fy[i] = 0.0;
	}
}


// --------------------------------------------------------
// D2Q9 kernel to re-set the fluid forces (and extrapolated
// IB velocities and weights) to zero:
// --------------------------------------------------------

__global__ void scsp_zero_forces_D2Q9(
	float* fx,
	float* fy,
	float* uIBvox,
	float* vIBvox,
	float* weights,
	int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {			
		fx[i] = 0.0;
		fy[i] = 0.0;
		uIBvox[i] = 0.0;
		vIBvox[i] = 0.0;
		weights[i] = 0.0;
	}
}