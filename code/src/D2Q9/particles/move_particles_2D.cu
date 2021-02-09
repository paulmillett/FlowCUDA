
# include "move_particles_2D.cuh"
# include <stdio.h>


// --------------------------------------------------------
// Update particle velocities and positions:
// --------------------------------------------------------

__global__ void move_particles_2D(float* x,
                                  float* y,
								  float* vx,
								  float* vy,
								  float* fx,
								  float* fy,
								  int nParts)
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nParts) {		
		/*
		if (i==1) printf("fx = %f \n",fx[i]); 
		float pvel = 0.005;
		if (i == 0) {
			vx[i] = -pvel;
			vy[i] = 0.00;
		}
		if (i == 1) {
			vx[i] = pvel;
			vy[i] = 0.00;
		}	
		*/	
		x[i] += vx[i];  // assume dt = 1
		y[i] += vy[i];
	}
}
