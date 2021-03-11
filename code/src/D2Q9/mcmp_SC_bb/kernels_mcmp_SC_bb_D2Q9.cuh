# ifndef KERNELS_MCMP_SC_BB_D2Q9_H
# define KERNELS_MCMP_SC_BB_D2Q9_H
# include <cuda.h>
# include "../../Utils/helper_math.h"



// --------------------------------------------------------
// Struct containing 2D particle data:
// --------------------------------------------------------

struct particle2D_bb {
	float2 r,v,f;
	float rad;
	float mass;
};



// --------------------------------------------------------
// Kernels:
// --------------------------------------------------------


__global__ void mcmp_zero_particle_forces_bb_D2Q9(
	particle2D_bb*,
	int); 


__global__ void mcmp_move_particles_bb_D2Q9(
	particle2D_bb*,
	int);
	
	
__global__ void mcmp_fix_particle_velocity_bb_D2Q9(
	particle2D_bb*,
	float,
	int);


__global__ void mcmp_initial_equilibrium_bb_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int);


__device__ void equilibrium_populations_bb_D2Q9(
	float*,
	float*,
	float,
	float,
	float,
	float,
	int);


__global__ void mcmp_map_particles_on_lattice_bb_D2Q9(
	particle2D_bb*,
	int*,
	int*,
	int*,
	int*,
	int,
	int);


__global__ void mcmp_update_particles_on_lattice_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	particle2D_bb*,
	int*,
	int*,
	int*,
	int*,
	int*,
	int,
	int);


__global__ void mcmp_compute_velocity_bb_D2Q9(
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
	int);


__global__ void mcmp_set_boundary_velocity_bb_D2Q9(
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
	int*,
	int,
	int);
		
		
__global__ void mcmp_compute_density_bb_D2Q9(
	float*,
	float*,
	float*,
	float*,
	int);


__global__ void mcmp_compute_SC_forces_bb_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	int*,
	float,
	float,
	float,
	int);


__global__ void mcmp_collide_stream_bb_D2Q9(
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
	int*,
	float,
	int);


__global__ void mcmp_bounce_back_D2Q9(
	float*,
	float*,
	int*,
	int*,
	int*,
	int);


__global__ void mcmp_bounce_back_moving_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	particle2D_bb*,
	int*,											 
	int*,
	int*,
	int*,
	int);


# endif  // KERNELS_MCMP_SC_BB_D2Q9_H