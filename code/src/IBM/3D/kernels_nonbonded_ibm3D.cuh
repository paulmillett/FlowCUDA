# ifndef KERNELS_NONBONDED_IBM3D_H
# define KERNELS_NONBONDED_IBM3D_H
# include "../../Utils/helper_math.h"
# include "membrane_data.h"
# include <cuda.h>


__global__ void reset_bin_lists_IBM3D(
	int*,
	int*,
	int,
	int);


__global__ void build_bin_lists_IBM3D(
	float3*,
	int*,
	int*,
	int3,	
	float,
	int,
	int);


__global__ void nonbonded_node_interactions_IBM3D(
	float3*,
	float3*,
	int*,
	int*,
	int*,
	int*,
	int3,	
	float,
	int,
	int,
	int);


__device__ inline void node_interaction_forces(
	const int, 
	const int, 
	const float3*,
	float3*);
			

__global__ void build_binMap_IBM3D(
	int*,
	int3,
	int,
	int);
		

__device__ inline int bin_index(
	const int, 
	const int,
	const int, 
	const int3);

			

# endif  // KERNELS_NONBONDED_IBM3D_H