
# include "compute_node_force_IBM2D.cuh"
# include <stdio.h>

// --------------------------------------------------------
// IBM2D node update kernel:
// --------------------------------------------------------

__global__ void compute_node_force_IBM2D(
	float* x,
	float* y,
	float* x0,
	float* y0,
	float* fx,
	float* fy,
	float kstiff,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		fx[i] = -kstiff*(x[i] - x0[i]);
		fy[i] = -kstiff*(y[i] - y0[i]);
	}
}
