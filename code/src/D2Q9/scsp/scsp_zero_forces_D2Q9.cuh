# ifndef SCSP_ZERO_FORCES_D2Q9_H
# define SCSP_ZERO_FORCES_D2Q9_H
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

# endif  // SCSP_ZERO_FORCES_D2Q9_H