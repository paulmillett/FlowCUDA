
# include "set_reference_node_positions_IBM2D.cuh"
# include <stdio.h>

// --------------------------------------------------------
// IBM2D node update kernel:
// --------------------------------------------------------

__global__ void set_reference_node_positions_IBM2D(
	float* x,
	float* y,
	float* x0,
	float* y0,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		x0[i] = x[i];
		y0[i] = y[i];
	}
}
