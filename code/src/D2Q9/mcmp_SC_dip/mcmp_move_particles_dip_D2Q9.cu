
# include "mcmp_move_particles_dip_D2Q9.cuh"
# include <stdio.h>


// --------------------------------------------------------
// Update particle velocities and positions:
// --------------------------------------------------------

__global__ void mcmp_move_particles_dip_D2Q9(particle2D_dip* pt,
   								             int nParts)
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nParts) {		
		
		//if (i==1) printf("fx = %f \n",pt[i].f.x); 
		float pvel = 0.005;
		if (i == 0) {
			pt[i].v.x = -pvel;
			pt[i].v.y = 0.00;
		}
		if (i == 1) {
			pt[i].v.x = pvel;
			pt[i].v.y = 0.00;
		}	
		pt[i].r.x += pt[i].v.x;  // assume dt = 1
		pt[i].r.y += pt[i].v.y; 
	}
}
