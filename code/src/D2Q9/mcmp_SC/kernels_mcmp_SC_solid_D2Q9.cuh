
# ifndef KERNELS_MCMP_SC_SOLID_D2Q9_H
# define KERNELS_MCMP_SC_SOLID_D2Q9_H
# include <cuda.h>
# include "../particles/particle_struct_D2Q9.cuh"



__global__ void mcmp_compute_velocity_solid_D2Q9(
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
	float*,
	int*,
	particle2D*,
	int);


__global__ void mcmp_compute_SC_forces_solid_1_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	float,
	float,
	float,
	int);


__global__ void mcmp_compute_SC_forces_solid_2_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	float,
	float,
	float,
	int);


__global__ void mcmp_compute_SC_forces_solid_3_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	float,
	float,
	float,
	int);


__global__ void mcmp_collide_stream_solid_D2Q9(
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
	float*, 
	float*,
	float*,
	int*,
	float,
	int);
												   

# endif  // KERNELS_MCMP_SC_SOLID_D2Q9_H