# ifndef KERNELS_POISSON_IBM3D_H
# define KERNELS_POISSON_IBM3D_H

# include "../../Utils/helper_math.h"
# include "data_structs/membrane_data.h"
# include <cuda.h>
# include <cufft.h>


__global__ void zero_G_poisson_IBM3D(
	float3*,
	int);
		

__global__ void extrapolate_interface_normal_poisson_IBM3D(
	node*,
	float3*,
	int,
	int,
	int,
	int,
	int,
	cell*,
	triangle*);


__global__ void test_interface_normal_poisson_IBM3D(
	float3*,
	int,
	int,
	int,
	int);
			

__global__ void calculate_rhs_poisson_IBM3D(
	float3*,
	cufftComplex*,
	int,
	int,
	int,
	int);


__global__ void solve_poisson_inplace(
	cufftComplex*,
	float*,
	float*,
	float*, 
	int,
	int,
	int);


__global__ void complex2real(
	cufftComplex*,
	float*, 
	int);


__global__ void rescale_indicator_array(
	float*,
	int);
			

__device__ inline int voxel_ndx(
	int,
	int,
	int,
	int,
	int,
	int);


# endif  // KERNELS_POISSON_IBM3D_H