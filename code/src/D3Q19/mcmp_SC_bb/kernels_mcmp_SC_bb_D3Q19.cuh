# ifndef KERNELS_MCMP_MCMP_SC_BB_D3Q19_H
# define KERNELS_MCMP_MCMP_SC_BB_D3Q19_H
# include <cuda.h>
# include "../../Utils/helper_math.h"



// --------------------------------------------------------
// Struct containing 3D particle data:
// --------------------------------------------------------

struct particle3D_bb {
	float3 r,v,f;
	float rad;
	float mass;
};



// --------------------------------------------------------
// Kernels:
// --------------------------------------------------------

__global__ void mcmp_zero_particle_forces_bb_D3Q19(
	particle3D_bb*,
	int); 


__global__ void mcmp_move_particles_bb_D3Q19(
	particle3D_bb*,
	int);
	
	
__global__ void mcmp_fix_particle_velocity_bb_D3Q19(
	particle3D_bb*,
	float,
	int);
		
		
__global__ void mcmp_map_particles_to_lattice_bb_D3Q19(
	particle3D_bb*,
	int*,
	int*,
	int*,
	int*,
	int*,
	int*,
	int,
    int);


__global__ void mcmp_set_boundary_velocity_bb_D3Q19(
	float,
	float,
	float,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	int,
	int);


__global__ void mcmp_initial_equilibrium_bb_D3Q19(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int);
	
	
__device__ void equilibrium_populations_bb_D3Q19(
	float*,
	float*,
	float,
	float,
	float,
	float,
	float,
	int);


__global__ void mcmp_compute_velocity_bb_D3Q19(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	particle3D_bb*,
	int*,	
	int*,
	int);


__global__ void mcmp_compute_density_bb_D3Q19(
	float*,
	float*,
	float*,
	float*,
	int);


__global__ void mcmp_compute_SC_forces_bb_D3Q19(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	particle3D_bb*,
	int*,
	int*,
	int*,
	float,											    
	int);


__global__ void mcmp_collide_stream_bb_D3Q19(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*, 
	float*,
	float*,	
	float*,
	float*,
	float*,										 
	int*,                                         	 
	float,
	int);


__global__ void mcmp_bounce_back_D3Q19(
	float*,
	float*,
	int*,
	int*,
	int*,
	int);


__global__ void mcmp_bounce_back_moving_D3Q19(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	particle3D_bb*,
	int*,											 
	int*,
	int*,
	int*,
	int);


# endif  // KERNELS_MCMP_MCMP_SC_BB_D3Q19_H