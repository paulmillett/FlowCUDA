# ifndef KERNELS_MCMP_SC_D2Q9_H
# define KERNELS_MCMP_SC_D2Q9_H
# include <cuda.h>



__global__ void mcmp_initial_equilibrium_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int);
	
	
__global__ void mcmp_compute_density_D2Q9(
	float*,
	float*,
	float*,
	float*,
	int);
	
	
__global__ void mcmp_compute_velocity_D2Q9(
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
	int);
	
	
__global__ void mcmp_compute_SC_pressure_D2Q9(
	float*,
	float*,
	float*,
	float,
	int);
	
	
__global__ void mcmp_compute_SC_forces_1_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	float,
	int);
											  
										
__global__ void mcmp_compute_SC_forces_2_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int*,
	float,
	int);
	
	
__global__ void mcmp_collide_stream_D2Q9(
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
		

# endif  // KERNELS_MCMP_SC_D2Q9_H