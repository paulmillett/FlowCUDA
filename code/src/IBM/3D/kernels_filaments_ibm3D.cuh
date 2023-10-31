# ifndef KERNELS_FILAMENTS_IBM3D_H
# define KERNELS_FILAMENTS_IBM3D_H
# include <cuda.h>
# include <curand.h>
# include <curand_kernel.h>
# include "data_structs/filament_data.h"
# include "data_structs/membrane_data.h"
# include "data_structs/neighbor_bins_data.h"
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
	
	
__global__ void update_bead_position_verlet_1_drag_IBM3D(
	bead*,
	float,
	float,
	float,
	int);


__global__ void update_bead_position_verlet_2_drag_IBM3D(
	bead*,
	float,
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
		
	
__global__ void compute_bead_force_spring_IBM3D(
	bead*,
	edgefilam*,
	filament*,
	int);
	

__global__ void compute_bead_force_FENE_IBM3D(
	bead*,
	edgefilam*,
	filament*,
	float delta,
	int);


__global__ void compute_bead_force_bending_IBM3D(
	bead*,
	filament*,
	int);
	

__global__ void compute_propulsion_force_IBM3D(
	bead*,
	edgefilam*,
	filament*,
	int);


__global__ void compute_propulsion_force_2_IBM3D(
	bead*,
	filament*,
	int);


__global__ void compute_propulsion_force_3_IBM3D(
	bead*,
	filament*,
	int);


__global__ void compute_thermal_force_IBM3D(
	bead*,
	filament*,
	curandState*,
	float,
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
	
	
__global__ void push_beads_into_sphere_IBM3D(
	bead*,
	float,
	float,
	float,
	float,
	int);	
	
	
__global__ void build_binMap_for_beads_IBM3D(
	bindata);
	

__global__ void reset_bin_lists_for_beads_IBM3D(
	bindata);


__global__ void build_bin_lists_for_beads_IBM3D(
	bead*,
	bindata,
	int);
	
	
__global__ void nonbonded_bead_interactions_IBM3D(
	bead*,
	bindata,
	float,
	float,
	int,
	float3,	
	int3);


__global__ void nonbonded_bead_node_interactions_IBM3D(
	bead*,
	node*,
	bindata,
	float,
	float,
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


__device__ inline void pairwise_bead_node_interaction_forces(
	const int, 
	const int,
	const float,
	const float,
	bead*,
	node*,
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


__global__ void init_curand_IBM3D(
	curandState*,
	unsigned long,
	int);


# endif  // KERNELS_FILAMENTS_IBM3D_H