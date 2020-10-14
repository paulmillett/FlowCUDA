
# include "compute_node_force_IBM3D.cuh"
# include <stdio.h>

// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void compute_node_force_IBM3D(
	float* x,
	float* y,
	float* z,
	float* x0,
	float* y0,
	float* z0,
	float* fx,
	float* fy,
	float* fz,
	float kstiff,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		fx[i] = -kstiff*(x[i] - x0[i]);
		fy[i] = -kstiff*(y[i] - y0[i]);
		fz[i] = -kstiff*(z[i] - z0[i]);
	}
}
