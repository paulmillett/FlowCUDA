
# include "kernels_ibm2D.cuh"
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



// --------------------------------------------------------
// IBM2D reference node update kernel:
// --------------------------------------------------------

__global__ void update_node_ref_position_IBM2D(
	float* x_ref,
	float* y_ref,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		x_ref[i] -= 0.001;
		y_ref[i] += 0.00;
	}
}



// --------------------------------------------------------
// IBM2D reference node update kernel:
// --------------------------------------------------------

__global__ void update_node_ref_position_IBM2D(
	float* x_ref,
	float* y_ref,
	float* x_ref_delta,
	float* y_ref_delta,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		x_ref[i] += x_ref_delta[i];
		y_ref[i] += y_ref_delta[i];
	}
}



// --------------------------------------------------------
// IBM2D reference node update kernel:
// --------------------------------------------------------

__global__ void update_node_ref_position_IBM2D(
	float* x_ref,
	float* y_ref,
	float* x_ref_start,
	float* y_ref_start,
	float* x_ref_end,
	float* y_ref_end,
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
	}
}



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



// --------------------------------------------------------
// IBM2D kernel to extrapolate IBM node force to LBM lattice
// --------------------------------------------------------

__global__ void extrapolate_force_IBM2D(
	float* x,
	float* y,
	float* fx,
	float* fy,
	float* fxLBM,
	float* fyLBM,
	int Nx,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nNodes) {
				
		// --------------------------------------
		// find nearest LBM voxel (rounded down)
		// --------------------------------------
		
		int i0 = int(floor(x[i]));
		int j0 = int(floor(y[i]));
		
		// --------------------------------------
		// loop over footprint
		// --------------------------------------
		
		for (int jj=j0; jj<=j0+1; jj++) {
			for (int ii=i0; ii<=i0+1; ii++) {
				int ndx = jj*Nx + ii;
				float rx = x[i] - float(ii);
				float ry = y[i] - float(jj);
				float del = (1.0-abs(rx))*(1.0-abs(ry));
				//fxLBM[ndx] += del*fx[i];
				//fyLBM[ndx] += del*fy[i];
				atomicAdd(&fxLBM[ndx],del*fx[i]);
				atomicAdd(&fyLBM[ndx],del*fy[i]);
			}
		}		
	}
	
}



// --------------------------------------------------------
// IBM2D kernel to extrapolate IBM node velocity to LBM
// lattice
// --------------------------------------------------------

__global__ void extrapolate_velocity_IBM2D(
	float* x,
	float* y,
	float* vx,
	float* vy,
	float* uIBvox,
	float* vIBvox,
	float* weight,
	int Nx,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nNodes) {
				
		// --------------------------------------
		// find nearest LBM voxel (rounded down)
		// --------------------------------------
		
		int i0 = int(floor(x[i]));
		int j0 = int(floor(y[i]));
		
		// --------------------------------------
		// loop over footprint
		// --------------------------------------
		
		for (int jj=j0; jj<=j0+1; jj++) {
			for (int ii=i0; ii<=i0+1; ii++) {
				int ndx = jj*Nx + ii;
				float rx = x[i] - float(ii);
				float ry = y[i] - float(jj);
				float del = sqrt(rx*rx + ry*ry); //(1.0-abs(rx))*(1.0-abs(ry));
				atomicAdd(&uIBvox[ndx],del*vx[i]);
				atomicAdd(&vIBvox[ndx],del*vy[i]);
				atomicAdd(&weight[ndx],del);
			}
		}		
	}
	
}



// --------------------------------------------------------
// IBM2D kernel to interpolate LBM velocity to IBM node:
// --------------------------------------------------------

__global__ void interpolate_velocity_IBM2D(
	float* x,
	float* y,
	float* vx,
	float* vy,
	float* uLBM,
	float* vLBM,
	int Nx,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nNodes) {
		
		// --------------------------------------
		// zero out velocities for node "i"
		// --------------------------------------
		
		vx[i] = 0.0;
		vy[i] = 0.0;
				
		// --------------------------------------
		// find nearest LBM voxel (rounded down)
		// --------------------------------------
		
		int i0 = int(floor(x[i]));
		int j0 = int(floor(y[i]));
				
		// --------------------------------------
		// loop over footprint
		// --------------------------------------
		
		for (int jj=j0; jj<=j0+1; jj++) {
			for (int ii=i0; ii<=i0+1; ii++) {
				int ndx = jj*Nx + ii;
				float rx = x[i] - float(ii);
				float ry = y[i] - float(jj);
				float del = (1.0-abs(rx))*(1.0-abs(ry));
				vx[i] += del*uLBM[ndx];
				vy[i] += del*vLBM[ndx];
			}
		}		
	}
	
}


