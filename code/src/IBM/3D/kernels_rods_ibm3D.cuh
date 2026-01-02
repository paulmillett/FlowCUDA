# ifndef KERNELS_RODS_IBM3D_H
# define KERNELS_RODS_IBM3D_H
# include <cuda.h>
# include <curand.h>
# include <curand_kernel.h>
# include "data_structs/rod_data.h"
# include "data_structs/cell_data.h"
# include "data_structs/neighbor_bins_data.h"
# include "../../Utils/helper_math.h"


__global__ void zero_rod_forces_torques_moments_IBM3D(
	rod*,	
	int);
	

__global__ void zero_bead_forces_IBM3D(
	beadrod*,	
	int);
	

__global__ void set_rod_position_orientation_IBM3D(
	beadrod*,
	rod*,	
	int);
		

__global__ void enforce_max_bead_force_IBM3D(
	beadrod*,
	float,
	int);
		

__global__ void enforce_max_rod_force_torque_IBM3D(
	rod*,
	float,
	float,
	int);

	
__global__ void update_bead_positions_rods_IBM3D(
	beadrod*,
	rod*,
	float,
	float,
	int);
	
	
__global__ void update_bead_positions_rods_singlet_IBM3D(
	beadrod*,
	rod*,
	float,
	int);


__global__ void update_bead_velocity_rods_IBM3D(
	beadrod*,
	float3,
	int3,
	float,
	int);
		
	
__global__ void update_rod_position_orientation_IBM3D(
	rod*,
	float,
	float,
	float,
	int);
	

__global__ void update_rod_position_orientation_fluid_IBM3D(
	rod*,
	float,
	float,
	float,
	int);


__global__ void update_rod_position_fluid_IBM3D(
	rod*,
	float,
	float,
	int);
	

__global__ void sum_rod_forces_torques_moments_IBM3D(
	beadrod*,
	rod*,
	int,	
	int);


__global__ void unwrap_bead_coordinates_rods_IBM3D(
	beadrod*,
	rod*,
	float3,
	int3,
	int);


__global__ void wrap_bead_coordinates_IBM3D(
	beadrod*,
	float3,
	int3,
	int);


__global__ void wrap_rod_coordinates_IBM3D(
	rod*,
	float3,
	int3,
	int);
		

__global__ void bead_wall_forces_ydir_IBM3D(
	beadrod*,
	float3,
	float,
	float,
	int);
		
		
__global__ void bead_wall_forces_zdir_IBM3D(
	beadrod*,
	float3,
	float,
	float,
	int);
			
			
__global__ void bead_wall_forces_ydir_zdir_IBM3D(
	beadrod*,
	float3,
	float,
	float,
	int);


__global__ void bead_wall_forces_cylinder_IBM3D(
	beadrod*,
	float3,
	float,
	float,
	float,
	int);


__global__ void push_beads_into_sphere_IBM3D(
	beadrod*,
	float,
	float,
	float,
	float,
	int);


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


__global__ void extrapolate_force_bead_rod_IBM3D(
	beadrod*,
	rod*,
	float*,
	float*,
	float*,
	int,
	int,
	int,
	int,
	int);
		

__global__ void interpolate_gradient_of_velocity_bead_IBM3D(
	beadrod*,
	float*,
	float*,
	float*,
	int,
	int,
	int,
	int);
		
		
__global__ void build_bin_lists_for_beads_IBM3D(
	beadrod*,
	bindata,
	int);
		
		
__global__ void nonbonded_bead_interactions_IBM3D(
	beadrod*,
	bindata,
	float,
	float,
	int,
	float3,	
	int3);
			
			
__device__ inline void pairwise_bead_interaction_forces_WCA(
	const int, 
	const int,
	const float,
	const float,
	beadrod*,
	float3,
	int3);


__device__ inline float x_deriv(
	const int,
	const int,
	const int,  
	const int,
	const int,
	const int,
	float*);
		

__device__ inline float y_deriv(
	const int,
	const int,
	const int,  
	const int,
	const int,
	const int,
	float*);
		
		
__device__ inline float z_deriv(
	const int,
	const int,
	const int,  
	const int,
	const int,
	const int,
	float*);	


__device__ inline int rod_voxel_ndx(
	int,
	int,
	int,
	int,
	int,
	int);
	

/*
__device__ inline float3 solve_angular_acceleration(
	const float,
	const float,
	const float,
	const float,
	const float,
	const float,
	const float3);


__device__ inline float determinantOfMatrix(
	float[3][3]);
*/


# endif  // KERNELS_RODS_IBM3D_H