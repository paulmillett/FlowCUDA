# ifndef KERNELS_SCSP_D2Q9_H
# define KERNELS_SCSP_D2Q9_H

# include "../iolets/boundary_condition_iolet.cuh"
# include <cuda.h>



__global__ void scsp_zero_forces_D2Q9(
	float*,
	float*,
	int);


__global__ void scsp_zero_forces_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	int);


__global__ void scsp_initial_equilibrium_D2Q9(
	float*,
	float*,
	float*,
	float*,
	int);


__global__ void scsp_force_velocity_match_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int);


__global__ void scsp_stream_collide_save_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	int*,
	iolet2D*,
	float,
	int,
	bool);


__global__ void scsp_stream_collide_save_forcing_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	int*,
	iolet2D*, 
	float,
	int);
	

__global__ void scsp_stream_collide_save_IBforcing_D2Q9(
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
	iolet2D*,
	float,
	int);


# endif  // KERNELS_SCSP_D2Q9_H