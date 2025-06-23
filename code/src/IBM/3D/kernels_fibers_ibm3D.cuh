# ifndef KERNELS_FIBERS_IBM3D_H
# define KERNELS_FIBERS_IBM3D_H
# include <cuda.h>
# include <curand.h>
# include <curand_kernel.h>
# include "data_structs/fiber_data.h"
# include "data_structs/neighbor_bins_data.h"
# include "../../Utils/helper_math.h"


__global__ void zero_bead_forces_fibers_IBM3D(
	beadfiber*,	
	int);


__global__ void calculate_bead_velocity_fibers_IBM3D(
	beadfiber*,	
	float,
	int);


__global__ void update_rstar_fibers_IBM3D(
	beadfiber*,
	int);


__global__ void update_bead_positions_fibers_IBM3D(
	beadfiber*,
	float*,
	float*,
	float*,
	int);


__global__ void compute_Laplacian_fibers_IBM3D(
	beadfiber*,
	float,
	int);


__global__ void compute_bending_force_fibers_IBM3D(
	beadfiber*,
	float,
	float,
	int);
	
	
__global__ void compute_tension_RHS_fibers_IBM3D(
	beadfiber*,
	edgefiber*,
	float*,
	float,
	float,
	int);


__global__ void compute_tension_tridiag_fibers_IBM3D(
	beadfiber*,
	edgefiber*,
	float*,
	float*,
	float*,
	float,
	float,
	int);
	
	
__global__ void compute_bead_update_matrices_fibers_IBM3D(
	beadfiber*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float,
	float,
	int);
		

__global__ void hydrodynamic_force_bead_fluid_IBM3D(
	beadfiber*,
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
	
	
__device__ inline int bead_fiber_voxel_ndx(
	int,
	int,
	int,
	int,
	int,
	int);
		

__global__ void unwrap_bead_coordinates_IBM3D(
	beadfiber*,
	fiber*,
	float3,
	int3,
	int);


__global__ void wrap_bead_coordinates_IBM3D(
	beadfiber*,	
	float3,
	int3,
	int);


__global__ void bead_wall_forces_ydir_IBM3D(
	beadfiber*,
	float3,
	float,
	float,
	int);


__global__ void bead_wall_forces_zdir_IBM3D(
	beadfiber*,
	float3,
	float,
	float,
	int);


__global__ void bead_wall_forces_ydir_zdir_IBM3D(
	beadfiber*,
	float3,
	float,
	float,
	int);


__global__ void build_binMap_for_beads_fibers_IBM3D(
	bindata);


__global__ void reset_bin_lists_for_beads_fibers_IBM3D(
	bindata);


__global__ void build_bin_lists_for_beads_fibers_IBM3D(
	beadfiber*,
	bindata,
	int);


__global__ void nonbonded_bead_interactions_IBM3D(
	beadfiber*,
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
	beadfiber*,
	float3,
	int3);


__device__ inline void pairwise_bead_interaction_forces_WCA(
	const int, 
	const int,
	const float,
	const float,
	beadfiber*,
	float3,
	int3);


__device__ inline int bin_index_for_beads_fibers(
	int, 
	int,
	int, 
	const int3);
	

# endif  // KERNELS_FIBERS_IBM3D_H