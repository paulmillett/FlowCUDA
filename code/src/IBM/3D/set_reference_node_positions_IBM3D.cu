
# include "set_reference_node_positions_IBM3D.cuh"
# include <stdio.h>

// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void set_reference_node_positions_IBM3D(
	float* x,
	float* y,
	float* z,
	float* x0,
	float* y0,
	float* z0,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		x0[i] = x[i];
		y0[i] = y[i];
		z0[i] = z[i];
	}
}
