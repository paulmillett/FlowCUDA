
# include "update_node_ref_position_IBM3D.cuh"
# include <stdio.h>



// --------------------------------------------------------
// IBM3D reference node update kernel:
// --------------------------------------------------------

__global__ void update_node_ref_position_IBM3D(
	float* x_ref,
	float* y_ref,
	float* z_ref,
	float* x_ref_delta,
	float* y_ref_delta,
	float* z_ref_delta,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		x_ref[i] += x_ref_delta[i];
		y_ref[i] += y_ref_delta[i];
		z_ref[i] += z_ref_delta[i];
	}
}


// --------------------------------------------------------
// IBM3D reference node update kernel:
// --------------------------------------------------------

__global__ void update_node_ref_position_IBM3D(
	float* x_ref,
	float* y_ref,
	float* z_ref,
	float* x_ref_start,
	float* y_ref_start,
	float* z_ref_start,
	float* x_ref_end,
	float* y_ref_end,
	float* z_ref_end,
	int step,
	int nSteps,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		float psi = sin(float(step)/float(nSteps)*M_PI/2.0);
		x_ref[i] = x_ref_start[i] + psi*(x_ref_end[i] - x_ref_start[i]);
		y_ref[i] = y_ref_start[i] + psi*(y_ref_end[i] - y_ref_start[i]);
		z_ref[i] = z_ref_start[i] + psi*(z_ref_end[i] - z_ref_start[i]);	
	}
}