
# include "update_node_position_IBM2D.cuh"
# include <stdio.h>

// --------------------------------------------------------
// IBM2D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_IBM2D(
	float* x,
	float* y,
	float* vx,
	float* vy,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		x[i] += vx[i];  // assume dt = 1
		y[i] += vy[i];
	}
}



// --------------------------------------------------------
// IBM2D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_IBM2D(
	float* x,
	float* y,
	float* x_start,
	float* y_start,
	float* x_end,
	float* y_end,
	float* vx,
	float* vy,
	int step,
	int nSteps,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		float x_old = x[i];
		float y_old = y[i];
		//float psi = sin(float(step)/float(nSteps)*M_PI/2.0);
		float psi = 0.5*(sin(M_PI*(float(step)/float(nSteps) - 0.5)) + 1.0); 
		x[i] = x_start[i] + psi*(x_end[i] - x_start[i]);
		y[i] = y_start[i] + psi*(y_end[i] - y_start[i]);
		vx[i] = x[i] - x_old;  // assume dt = 1
		vy[i] = y[i] - y_old;  // "           "
	}
}

