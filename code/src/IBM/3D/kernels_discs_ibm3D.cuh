# ifndef KERNELS_DISCS_IBM3D_H
# define KERNELS_DISCS_IBM3D_H
# include <cuda.h>
# include <curand.h>
# include <curand_kernel.h>
# include "data_structs/disc_data.h"
# include "data_structs/neighbor_bins_data.h"
# include "../../Utils/helper_math.h"



__global__ void init_rand_kernel_discs_IBM3D(
	curandState *state, 
    unsigned long seed,
	int);


__global__ void zero_disc_forces_torques_moments_IBM3D(
	disc*,	
	int);
	

__global__ void zero_bead_forces_IBM3D(
	beaddisc*,	
	int);
	

__global__ void set_disc_position_orientation_IBM3D(
	beaddisc*,
	disc*,	
	int);


__global__ void enforce_max_bead_force_IBM3D(
	beaddisc*,
	float,
	int);
		

__global__ void enforce_max_disc_force_torque_IBM3D(
	disc*,
	float,
	float,
	int);


__global__ void add_gravity_force_to_beads_IBM3D(
	beaddisc*,
	float,
	int);

	
__global__ void update_bead_positions_discs_IBM3D(
	beaddisc*,
	disc*,
	int);


__global__ void update_bead_velocity_discs_IBM3D(
	beaddisc*,
	float3,
	int3,
	float,
	int);
	

__global__ void update_disc_position_orientation_fluid_IBM3D(
	disc*,
	float,
	int);
	
	
__global__ void update_disc_position_orientation_no_fluid_IBM3D(
	disc*,
	float,
	int);


__global__ void assign_velocity_to_backfill_discs_IBM3D(
	disc*,
	float,
	int);
	
	
__global__ void move_disc_back_to_inlet_random_IBM3D(
	disc*,
	float3,
	float,
	float,
	float,
	int,
	curandState*);


__global__ void sum_disc_forces_torques_moments_IBM3D(
	beaddisc*,
	disc*,
	int);


__global__ void unwrap_bead_coordinates_discs_IBM3D(
	beaddisc*,
	disc*,
	float3,
	int3,
	int);


__global__ void wrap_bead_coordinates_IBM3D(
	beaddisc*,
	float3,
	int3,
	int);


__global__ void wrap_disc_coordinates_IBM3D(
	disc*,
	float3,
	int3,
	int);
		

__global__ void bead_wall_forces_ydir_IBM3D(
	beaddisc*,
	float3,
	float,
	float,
	int);
		
		
__global__ void bead_wall_forces_zdir_IBM3D(
	beaddisc*,
	float3,
	float,
	float,
	int);
			
			
__global__ void bead_wall_forces_ydir_zdir_IBM3D(
	beaddisc*,
	float3,
	float,
	float,
	int);


__global__ void bead_wall_forces_cylinder_IBM3D(
	beaddisc*,
	float3,
	float,
	float,
	float,
	float,
	int);


__global__ void bead_wall_forces_nozzle_IBM3D(
	beaddisc*,
	float3,
	float,
	float,
	float,
	float,
	float,
	float,
	float,
	int);


__global__ void push_beads_into_sphere_IBM3D(
	beaddisc*,
	float,
	float,
	float,
	float,
	int);


__global__ void push_beads_into_cylinder_IBM3D(
	beaddisc*,
	float3,
	float,
	float,
	float,
	int);
	
	
__global__ void push_beads_into_duct_IBM3D(
	beaddisc*,
	float3,
	float,
	float,
	int);
	
	
__global__ void push_beads_into_slit_IBM3D(
	beaddisc*,
	float3,
	float,
	float,
	int);


__global__ void push_beads_into_nozzle_IBM3D(
	beaddisc*,
	float3,
	float,
	float,
	float,
	float,
	float,
	int);
			

__global__ void hydrodynamic_force_bead_rod_IBM3D(
	beaddisc*,
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


__global__ void extrapolate_force_bead_disc_IBM3D(
	beaddisc*,
	disc*,
	float*,
	float*,
	float*,
	int,
	int,
	int,
	int);
		

__global__ void interpolate_gradient_of_velocity_bead_IBM3D(
	beaddisc*,
	float*,
	float*,
	float*,
	int,
	int,
	int,
	int);
		
	
__global__ void build_bin_lists_for_beads_IBM3D(
	beaddisc*,
	bindata,
	int);
		
		
__global__ void nonbonded_bead_interactions_IBM3D(
	beaddisc*,
	bindata,
	float,
	float,
	float,
	int,
	float3,	
	int3);


__global__ void nonbonded_bead_interactions_with_friction_IBM3D(
	beaddisc*,
	bindata,
	float,
	float,
	float,
	int,
	float3,	
	int3);


__global__ void nonbonded_bead_interactions_with_virial_IBM3D(
	beaddisc*,
	tensor*,
	bindata,
	float,
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
	const float,
	beaddisc*,
	float3,
	int3);


__device__ inline void pairwise_bead_interaction_forces_with_friction(
	const int, 
	const int,
	const float,
	const float,
	const float,
	beaddisc*,
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


__device__ inline int disc_voxel_ndx(
	int,
	int,
	int,
	int,
	int,
	int);






# endif  // KERNELS_DISCS_IBM3D_H