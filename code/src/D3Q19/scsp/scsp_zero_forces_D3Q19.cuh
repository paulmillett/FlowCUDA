# ifndef SCSP_ZERO_FORCES_D3Q19_H
# define SCSP_ZERO_FORCES_D3Q19_H
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
		
# endif  // SCSP_ZERO_FORCES_D3Q19_H