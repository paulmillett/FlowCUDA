# ifndef KERNELS_SCSP_ACTIVE_D2Q9_H
# define KERNELS_SCSP_ACTIVE_D2Q9_H

# include "../iolets/boundary_condition_iolet.cuh"
# include "tensor2D.h" 
# include <cuda.h>



__global__ void scsp_active_zero_forces_D2Q9(
	float2*,
	int);


__global__ void scsp_active_initial_equilibrium_D2Q9(
	float*,
	float*,
	float2*,
	int);


__global__ void scsp_active_stream_collide_save_D2Q9(
	float*,
	float*,
	float*,
	float2*,
	int*,
	int*,
	float,
	int);


__global__ void scsp_active_stream_collide_save_forcing_D2Q9(
	float*,
    float*,
	float*,
	float2*,
	float2*,
	int*,
	int*,
	float,
	int);


__global__ void scsp_active_update_orientation_D2Q9(
	float2*,
	float2*,
	int*,
	float,
	float,
	int);


__global__ void scsp_active_fluid_stress_D2Q9(
	float2*,
	tensor2D*,
	float,
	int);


__global__ void scsp_active_fluid_forces_D2Q9(
	float2*,
	tensor2D*,
	int*,
	int);
		

# endif  // KERNELS_SCSP_ACTIVE_D2Q9_H