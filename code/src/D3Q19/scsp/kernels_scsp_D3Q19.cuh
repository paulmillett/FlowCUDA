# ifndef KERNELS_SCSP_D3Q19_H
# define KERNELS_SCSP_D3Q19_H

# include "../iolets/boundary_condition_iolet.cuh"
# include <cuda.h>



__global__ void scsp_zero_forces_D3Q19(
	float*,
	float*,
	float*,
	int);


__global__ void scsp_zero_forces_D3Q19(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int);


__global__ void scsp_initial_equilibrium_D3Q19(
	float*,
	float*,
	float*,
	float*,
	float*,										  
	int);


__device__ void equilibrium_populations_bb_D3Q19(
	float*,
	const float,
	const float,
	const float,
	const float,
	const int);
														 

__global__ void scsp_set_boundary_shear_velocity_D3Q19(
	float,
	float,
	float*,
	float*,
	float*,
	float*,
	float*,											  
	int,
	int,
	int,
	int);


__global__ void scsp_stream_collide_save_D3Q19(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	int*,
	iolet*,
	float,
	int,
	bool);


__global__ void scsp_stream_collide_save_forcing_D3Q19(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	int*,
	iolet*,
	float,
	int);
	
	
__global__ void scsp_stream_collide_save_forcing_dt_D3Q19(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	int*,
	iolet*,
	float,
	float,
	int);
				

__global__ void scsp_stream_collide_save_IBforcing_D3Q19(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	int*,
	iolet*,
	float,
	int);

		
# endif  // KERNELS_SCSP_D3Q19_H