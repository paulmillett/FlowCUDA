# ifndef KERNELS_SCSP_PSM_D3Q19_H
# define KERNELS_SCSP_PSM_D3Q19_H

# include "sphere.h"
# include "radix.h"
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
	int,
	float3,
	int3);													 
		

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
    int,
	float3,
	int3);


__device__ void equilibrium_populations_psm_D3Q19(
	float*,
	const float,
	const float,
	const float,
	const float);
	
	
__global__ void zero_sphere_forces_torques_D3Q19(
	sphere*,	
	int);


__global__ void update_sphere_position_orientation_verlet_1_D3Q19(
	sphere*,
	float,
	int);


__global__ void update_sphere_position_orientation_verlet_2_D3Q19(
	sphere*,
	float,
	int);


__global__ void sphere_interactions_with_friction_radix_PSM(
	sphere*,
	radixdata_psm,
	const contact,
	int,
	float3,	
	int3);
		

__device__ inline void pairwise_sphere_interaction_forces_with_friction(
	const int, 
	const int,
	const contact,
	sphere*,
	float3,
	int3);		


__global__ void build_radix_cellMap_PSM(
	radixdata_psm);
			
			
__device__ inline int cell_index_PSM(
	int, 
	int,
	int, 
	const int3);
	
	
__global__ void calculate_radix_cellIDs_per_sphere_PSM(
	sphere*,
	radixdata_psm,
	int);


__global__ void calculate_radix_cell_offsets_PSM(
	sphere*,
	radixdata_psm,
	int);


__global__ void update_new_bead_indices_PSM(
	radixdata_psm,
	int);



		
# endif  // KERNELS_SCSP_PSM_D3Q19_H