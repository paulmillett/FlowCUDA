# ifndef EXTRAPOLATE_FORCE_IBM3D_H
# define EXTRAPOLATE_FORCE_IBM3D_H
# include <cuda.h>

__global__ void extrapolate_force_IBM3D(
	float*,
    float*,
    float*,
	float*,
    float*,
	float*,
	float*,
    float*,
	float*,
	int,
    int,
	int);

# endif  // EXTRAPOLATE_FORCE_IBM3D_H