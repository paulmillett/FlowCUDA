# ifndef EXTRAPOLATE_VELOCITY_IBM3D_H
# define EXTRAPOLATE_VELOCITY_IBM3D_H
# include <cuda.h>

__global__ void extrapolate_velocity_IBM3D(
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
	int,
    int,
	int);

# endif  // EXTRAPOLATE_VELOCITY_IBM3D_H