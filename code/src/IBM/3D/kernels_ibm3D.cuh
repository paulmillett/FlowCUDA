# ifndef KERNELS_IBM3D_H
# define KERNELS_IBM3D_H
# include <cuda.h>
# include "../../Utils/helper_math.h"
# include "data_structs/membrane_data.h"


__global__ void update_node_position_IBM3D(
	node*,
	int);


__global__ void update_node_position_dt_IBM3D(
	node*,
	float,
	int);
	

__global__ void update_node_position_include_force_IBM3D(
	node*,
	float,
	float,
	int);


__global__ void update_node_position_overdamped_IBM3D(
	node*,
	float,
	float,
	int);
		
	
__global__ void update_node_position_verlet_1_IBM3D(
	node*,
	float,
	float,
	int);


__global__ void update_node_position_verlet_2_IBM3D(
	node*,
	float,
	float,
	int);


__global__ void update_node_position_verlet_1_drag_IBM3D(
	node*,
	float,
	float,
	float,
	int);


__global__ void update_node_position_verlet_2_drag_IBM3D(
	node*,
	float,
	float,
	float,
	int);


__global__ void update_node_position_vacuum_IBM3D(
	node*,
	float,	
	int);
	

__global__ void update_node_position_IBM3D(
	node*,
    float3*,
	float3*,
    int,
	int,
    int);
	
	
__global__ void zero_velocities_forces_IBM3D(
	node*,
	int);


__global__ void enforce_max_node_force_IBM3D(
	node*,
	float,
	int);


__global__ void add_drag_force_to_node_IBM3D(
	node*,
	float,
	int);


__global__ void add_xdir_force_IBM3D(
	node*,
	float,
	int);
		
	
__global__ void update_node_ref_position_IBM3D(
	float3*,
    float3*,
	int);


__global__ void update_node_ref_position_IBM3D(
	float3*,
    float3*,
	float3*,
	int,
	int,
    int);

	
__global__ void set_reference_node_positions_IBM3D(
	node*,
    float3*,
    int);
		
		
__global__ void interpolate_velocity_IBM3D(
	node*,
    float*,
	float*,
    float*,
	int,
	int,
	int,
    int);
	
	
__global__ void extrapolate_velocity_IBM3D(
	node*,
    float*,
	float*,
    float*,
	float*,
	int,
    int,
	int);
	
	
__global__ void extrapolate_force_IBM3D(
	node*,
    float*,
	float*,
    float*,
	int,
    int,
	int,
	int);
	

__global__ void viscous_force_velocity_difference_IBM3D(
	node*,
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
	
	
__global__ void repulsive_force_solid_lattice_IBM3D(
	node*,
	int*,
	float,
	float,
	int,
	int,
	int,
	int);
			

__global__ void compute_node_force_IBM3D(
	node*,
    float3*,
	float,
    int);
	
	
__device__ inline int voxel_ndx(
	int,
	int,
	int,
	int,
	int,
	int);
		
		
# endif  // KERNELS_IBM3D_H