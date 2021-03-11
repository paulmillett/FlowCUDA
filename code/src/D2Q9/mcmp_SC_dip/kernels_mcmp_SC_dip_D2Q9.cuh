# ifndef KERNELS_MCMP_MCMP_SC_DIP_D2Q9_H
# define KERNELS_MCMP_MCMP_SC_DIP_D2Q9_H
# include <cuda.h>
# include "../../Utils/helper_math.h"



// --------------------------------------------------------
// Struct containing 2D particle data for a diffuse-
// interface particle (dip):
// --------------------------------------------------------

struct particle2D_dip {
	float2 r,v,f;
	float rInner;
	float rOuter;
	float mass;
};



// --------------------------------------------------------
// Kernels:
// --------------------------------------------------------

__global__ void mcmp_zero_particle_forces_dip_D2Q9(
	particle2D_dip*,
	int); 


__global__ void mcmp_move_particles_dip_D2Q9(
	particle2D_dip*,
	int);
	
	
__global__ void mcmp_fix_particle_velocity_dip_D2Q9(
	particle2D_dip*,
	float,
	int);
		
		
__global__ void mcmp_map_particles_to_lattice_dip_D2Q9(
	float*,
	particle2D_dip*,
	int*,
	int*,
	int*,
	int,
    int);


__global__ void mcmp_set_boundary_velocity_dip_D2Q9(
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


__global__ void mcmp_initial_equilibrium_dip_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int);


__global__ void mcmp_compute_velocity_dip_D2Q9(
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
	particle2D_dip*,
	int*,
	int);
	
	
__global__ void mcmp_compute_velocity_dip_2_D2Q9(
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
	particle2D_dip*,
	int*,
	int);


__global__ void mcmp_compute_density_dip_D2Q9(
	float*,
	float*,
	float*,
	float*,
	int);


__global__ void mcmp_compute_SC_forces_dip_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	particle2D_dip*,
	int*,
	int*,
	float,
	float,
	float,
	float,											    
	int);


__global__ void mcmp_collide_stream_dip_D2Q9(
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


# endif  // KERNELS_MCMP_MCMP_SC_DIP_D2Q9_H