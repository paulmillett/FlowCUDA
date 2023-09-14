# ifndef KERNELS_CAPSULE_IBM3D_H
# define KERNELS_CAPSULE_IBM3D_H
# include "../../Utils/helper_math.h"
# include "membrane_data.h"
# include <cuda.h>


__global__ void zero_reference_vol_area_IBM3D(
	cell*, 
	int);
		
		
__global__ void rest_triangle_skalak_IBM3D(
	float3*,
	triangle*,
	cell*, 
	int);


__global__ void rest_edge_lengths_IBM3D(
    float3*,
	edge*,
	int);


__global__ void rest_edge_angles_IBM3D(
	float3*,
	edge*,
	triangle*,
	int);


__global__ void rest_triangle_areas_IBM3D(
	float3*,
	triangle*,
	cell*,
	int);
	
	
__global__ void rest_cell_volumes_IBM3D(
	float3*,
	triangle*,
	cell*,
	int);


__global__ void compute_node_force_membrane_skalak_IBM3D(
	triangle*,
	float3*,
	float3*,
	cell*,	
	int);


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
    int);


__global__ void compute_node_force_membrane_bending_IBM3D(
	triangle*,
	float3*,
	float3*,
	edge*,
	cell*,
    int);
		

__global__ void compute_node_force_membrane_volume_IBM3D(
	triangle*,
	float3*,
	cell*,
    int);


__global__ void compute_node_force_membrane_globalarea_IBM3D(
	triangle*,
	float3*,
	float3*,
	cell*,	
	float,
	int);


__device__ inline float triangle_signed_volume(
	const float3,
	const float3,
	const float3); 


__device__ inline float3 triangle_normalvector(
	const float3,
	const float3,
	const float3); 


__device__ inline float angle_between_faces(
	const float3,
	const float3,
	const float3); 


__device__ inline int unique_triangle_vertex(
	const int,
	const int,
	const int,
	const int,
	const int);
				

__device__ inline void add_force_to_vertex(
	int,
	float3*,
	const float3); 


__global__ void zero_node_forces_IBM3D(
    float3*,
	int);


__global__ void zero_cell_volumes_IBM3D(
    cell*,
	int);


__global__ void unwrap_node_coordinates_IBM3D(
	float3*,
	cell*,
	int*,
	float3,
	int3,
	int);


__global__ void wrap_node_coordinates_IBM3D(
	float3*,	
	float3,
	int3,
	int);


__global__ void change_cell_volumes_IBM3D(
    cell*,
	float,
	int);


__global__ void scale_edge_lengths_IBM3D(
	edge*,
	float,
	int);


__global__ void scale_face_areas_IBM3D(
	triangle*,
	float,
	int);


__global__ void scale_cell_areas_volumes_IBM3D(
    cell*,
	float,
	int);
		

__global__ void cells_center_of_mass_IBM3D(
	float3*,
	cell*,
	int*,
	int);


# endif  // KERNELS_CAPSULE_IBM3D_H