
# include "zero_forces_2D.cuh"


// --------------------------------------------------------
// Zero particle forces:
// --------------------------------------------------------

__global__ void zero_forces_2D(float* fx,
                               float* fy,
							   int nParts)
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nParts) {		
		fx[i] = 0.0;
		fy[i] = 0.0;
	}
}
