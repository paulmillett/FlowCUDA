# ifndef KERNELS_MEMBRANE_IBM3D_H
# define KERNELS_MEMBRANE_IBM3D_H
# include "data.h"
# include <cuda.h>
# include "../../Utils/helper_math.h"


__global__ void compute_node_force_membrane_area_IBM3D(
	triangle*,
    float3*,
	float3*,
	cell*,
    float,
	int);
	
	
__global__ void compute_node_force_membrane_edge_IBM3D(
	triangle*,
	float3*,
	float3*,
	edge*,
    float,
	float,
    int);


__global__ void compute_node_force_membrane_volume_IBM3D(
	triangle*,
	float3*,
	cell*,
    float,
    int);
		

__device__ inline float triangle_signed_volume(
	const float3,
	const float3,
	const float3); 
	
	
__device__ inline void triangle_area_normalvector(
	const float3,
	const float3,
	const float3,
	float,
	float3); 
	
	
__device__ inline float angle_between_faces(
	const float3,
	const float3,
	const float3); 
		
	
__device__ inline void add_force_to_vertex(
	float3,
	const float3); 


__global__ void zero_node_forces_IBM3D(
    float3*,
	int);
	
	
__global__ void zero_cell_volumes_IBM3D(
    cell*,
	int);


# endif  // KERNELS_MEMBRANE_IBM3D_H