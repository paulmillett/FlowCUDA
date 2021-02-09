
# include "mcmp_compute_density_psm_D2Q9.cuh"
# include <stdio.h>

// --------------------------------------------------------
// D2Q9 compute density for each component: 
// --------------------------------------------------------

__global__ void mcmp_compute_density_psm_D2Q9(float* fA,
                                        	  float* fB,
										      float* rA,
										      float* rB,
										      int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		int offst = i*9;			
		rA[i] = fA[offst] + fA[offst+1] + fA[offst+2] + fA[offst+3] + fA[offst+4] + fA[offst+5] + fA[offst+6] +
		        fA[offst+7] + fA[offst+8];
		rB[i] = fB[offst] + fB[offst+1] + fB[offst+2] + fB[offst+3] + fB[offst+4] + fB[offst+5] + fB[offst+6] +
		        fB[offst+7] + fB[offst+8];
	}
}