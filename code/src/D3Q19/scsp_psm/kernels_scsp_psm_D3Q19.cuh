# ifndef KERNELS_SCSP_PSM_D3Q19_H
# define KERNELS_SCSP_PSM_D3Q19_H

# include "sphere.h"
# include <cuda.h>



__global__ void scsp_psm_initial_equilibrium_D3Q19(
	float*,
	float*,
	float*,
	float*,
	float*,										  
	int);


__device__ void equilibrium_populations_psm_D3Q19(
	float*,
	const float,
	const float,
	const float,
	const float,
	const int);
									 

__global__ void scsp_psm_set_boundary_shear_velocity_D3Q19(
	float,
	float,
	float*,
	float*,
	float*,
	float*,
	float*,											  
	int,
	int,
	int,
	int);


__global__ void scsp_psm_map_particles_to_lattice_D3Q19(
	float*,
	int*,
	sphere*,
	int,
	int,
	int,
	int,
	int);													 


__global__ void scsp_psm_stream_collide_save_D3Q19(
	float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    int*,
    int*,
    sphere*,
    float,
    int,
    int,
    int,
    int);


__device__ void equilibrium_populations_psm_D3Q19(
	float*,
	const float,
	const float,
	const float,
	const float);
	
	
__global__ void zero_sphere_forces_torques(
	sphere*,	
	int);


__global__ void update_sphere_position_orientation(
	sphere*,
	float,
	int);



		
# endif  // KERNELS_SCSP_PSM_D3Q19_H