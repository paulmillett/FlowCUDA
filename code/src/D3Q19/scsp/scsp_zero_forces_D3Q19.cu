
# include "scsp_zero_forces_D3Q19.cuh"
# include <stdio.h>

// --------------------------------------------------------
// D3Q19 kernel to reset forces to zero: 
// --------------------------------------------------------

__global__ void scsp_zero_forces_D3Q19(
	float* fx,
	float* fy,
	float* fz,
	int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {			
		fx[i] = 0.0;
		fy[i] = 0.0;
		fz[i] = 0.0;
	}
}



// --------------------------------------------------------
// D2Q9 kernel to re-set the fluid forces (and extrapolated
// IB velocities and weights) to zero:
// --------------------------------------------------------

__global__ void scsp_zero_forces_D3Q19(
	float* fx,
	float* fy,
	float* fz,
	float* uIBvox,
	float* vIBvox,
	float* wIBvox,
	float* weights,
	int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {			
		fx[i] = 0.0;
		fy[i] = 0.0;
		fz[i] = 0.0;
		uIBvox[i] = 0.0;
		vIBvox[i] = 0.0;
		wIBvox[i] = 0.0;
		weights[i] = 0.0;
	}
}