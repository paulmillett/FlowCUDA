# ifndef KERNELS_NONBONDED_IBM3D_H
# define KERNELS_NONBONDED_IBM3D_H
# include "../../Utils/helper_math.h"
# include "data_structs/membrane_data.h"
# include "data_structs/neighbor_bins_data.h"
# include <cuda.h>



__global__ void build_binMap_IBM3D(
	bindata);


__global__ void reset_bin_lists_IBM3D(
	bindata);


__global__ void build_bin_lists_IBM3D(
	float3*,
	bindata,
	int);


__global__ void nonbonded_node_interactions_IBM3D(
	float3*,
	float3*,
	int*,
	cell*,
	bindata,
	float,
	float,
	int,
	float3,	
	int3);


__device__ inline void pairwise_interaction_forces(
	const int, 
	const int,
	const int,
	const float,
	const float,
	float3*,
	float3*,
	cell*,
	float3,
	int3);


__global__ void wall_forces_ydir_IBM3D(
	float3*,
	float3*,
	float3,
	int);


__global__ void wall_forces_zdir_IBM3D(
	float3*,
	float3*,
	float3,
	float,
	float,
	int);
		
		
__global__ void wall_forces_ydir_zdir_IBM3D(
	float3*,
	float3*,
	float3,
	float,
	float,
	int);


__device__ inline int bin_index(
	const int, 
	const int,
	const int, 
	const int3);

			

# endif  // KERNELS_NONBONDED_IBM3D_H