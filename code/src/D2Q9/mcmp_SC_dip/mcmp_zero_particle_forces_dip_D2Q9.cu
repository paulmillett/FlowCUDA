
# include "mcmp_zero_particle_forces_dip_D2Q9.cuh"


// --------------------------------------------------------
// Zero particle forces:
// --------------------------------------------------------

__global__ void mcmp_zero_particle_forces_dip_D2Q9(particle2D_dip* pt,
							                       int nParts)
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nParts) {		
		pt[i].f.x = 0.0;
		pt[i].f.y = 0.0;
	}
}
