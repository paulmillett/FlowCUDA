# ifndef KERNELS_RIGIDS_IBM3D_H
# define KERNELS_RIGIDS_IBM3D_H
# include <cuda.h>
# include <curand.h>
# include <curand_kernel.h>
# include "data_structs/rigid_data.h"
# include "data_structs/neighbor_bins_data.h"
# include "../../Utils/helper_math.h"


__global__ void zero_rigid_forces_torques_IBM3D(
	rigid*,	
	int);


__global__ void enforce_max_rigid_force_torque_IBM3D(
	rigid*,
	float,
	float,
	int);


__global__ void update_node_positions_rigids_IBM3D(
	rigidnode*,
	rigid*,
	int);


__global__ void update_rigid_position_orientation_IBM3D(
	rigid*,
	float,
	int);


__global__ void sum_rigid_forces_torques_IBM3D(
	rigidnode*,
	rigid*,
	int);


__global__ void unwrap_node_coordinates_rigid_IBM3D(
	rigidnode*,
	rigid*,
	float3,
	int3,
	int);


__global__ void wrap_node_coordinates_rigid_IBM3D(
	rigidnode*,
	float3,
	int3,
	int);

	
__global__ void wrap_rigid_coordinates_IBM3D(
	rigid* rigids,
	float3 Box,
	int3 pbcFlag,
	int nRigids);


__global__ void rigid_node_wall_forces_ydir_IBM3D(
	rigidnode*,
	float3,
	float,
	float,
	int);
		
		
__global__ void rigid_node_wall_forces_zdir_IBM3D(
	rigidnode*,
	float3,
	float,
	float,
	int);
			
			
__global__ void rigid_node_wall_forces_ydir_zdir_IBM3D(
	rigidnode*,
	float3,
	float,
	float,
	int);


__global__ void rigid_node_wall_forces_cylinder_IBM3D(
	rigidnode*,
	float3,
	float,
	float,
	float,
	int);

/*
__global__ void hydrodynamic_force_bead_rod_IBM3D(
	beadrod*,
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
	int,
	int);
*/
			
		
__global__ void build_bin_lists_for_rigid_nodes_IBM3D(
	rigidnode*,
	bindata,
	int);
		
		
__global__ void nonbonded_rigid_node_interactions_IBM3D(
	rigidnode*,
	bindata,
	float,
	float,
	int,
	float3,	
	int3);
			
			
__device__ inline void pairwise_rigid_node_interaction_forces_WCA(
	const int, 
	const int,
	const float,
	const float,
	rigidnode*,
	float3,
	int3);


# endif  // KERNELS_RIGIDS_IBM3D_H