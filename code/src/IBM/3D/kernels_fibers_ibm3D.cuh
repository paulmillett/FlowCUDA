# ifndef KERNELS_FIBERS_IBM3D_H
# define KERNELS_FIBERS_IBM3D_H
# include <cuda.h>
# include <curand.h>
# include <curand_kernel.h>
# include "data_structs/fiber_data.h"
# include "data_structs/neighbor_bins_data.h"
# include "../../Utils/helper_math.h"


__global__ void update_rstar_fibers_IBM3D(
	beadfiber*,
	int);


__global__ void update_rstar_fibers_IBM3D(
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
	fiber*,
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
	
	
__global__ void compute_bead_update_matrices_IBM3D(
	beadfiber*,
	fiber*,
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
		
		

# endif  // KERNELS_FIBERS_IBM3D_H