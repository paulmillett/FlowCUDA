# ifndef KERNELS_FILAMENTS_IBM3D_H
# define KERNELS_FILAMENTS_IBM3D_H
# include <cuda.h>
# include "filament_data.h"
# include "../../Utils/helper_math.h"


__global__ void update_bead_position_verlet_1_IBM3D(
	bead*,
	float,
	float,
	int);


__global__ void update_bead_position_verlet_2_IBM3D(
	bead*,
	float,
	float,
	int);


__global__ void zero_bead_velocities_forces_IBM3D(
	bead*,
	int);
	

__global__ void enforce_max_bead_force_IBM3D(
	bead*,
	float,
	int);
		
		
__global__ void add_drag_force_to_bead_IBM3D(
	bead*,
	float,
	int);


__global__ void zero_bead_forces_IBM3D(
	bead*,	
	int);
		
	
__global__ void compute_bead_force_IBM3D(
	bead*,
	edgefilam*,
	filament*,
	int);
	
	
__global__ void compute_bead_force_bending_IBM3D(
	bead*,
	filament*,
	int);
	
	
__global__ void viscous_force_velocity_difference_bead_IBM3D(
	bead*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float,
	int,
	int,
	int,
	int);
			
	
__global__ void unwrap_bead_coordinates_IBM3D(
	bead*,
	filament*,
	float3,
	int3,
	int);
	
	
__global__ void wrap_bead_coordinates_IBM3D(
	bead*,	
	float3,
	int3,
	int);
	
	
__global__ void bead_wall_forces_ydir_IBM3D(
	bead*,
	float3,
	float,
	float,
	int);
	
	
__global__ void bead_wall_forces_zdir_IBM3D(
	bead*,
	float3,
	float,
	float,
	int);
	
	
__global__ void bead_wall_forces_ydir_zdir_IBM3D(
	bead*,
	float3,
	float,
	float,
	int);
	
	
__global__ void build_binMap_for_beads_IBM3D(
	int*,
	int3,
	int,
	int);
	
	
__global__ void reset_bin_lists_for_beads_IBM3D(
	int*,
	int*,
	int,
	int);
	
	
__global__ void build_bin_lists_for_beads_IBM3D(
	bead*,
	int*,
	int*,	
	int3,	
	float,
	int,
	int);
			

__global__ void nonbonded_bead_interactions_IBM3D(
	bead*,
	int*,
	int*,
	int*,
	int3,	
	float,
	float,
	float,
	int,
	int,
	int,
	float3,	
	int3);


__device__ inline void pairwise_bead_interaction_forces(
	const int, 
	const int,
	const float,
	const float,
	bead*,
	float3,
	int3);


__device__ inline void add_force_to_bead(
	int,
	bead*,
	const float3);
	
	
__device__ inline int bin_index_for_beads(
	int, 
	int,
	int, 
	const int3);


__device__ inline int bead_voxel_ndx(
	int,
	int,
	int,
	int,
	int,
	int);


# endif  // KERNELS_FILAMENTS_IBM3D_H