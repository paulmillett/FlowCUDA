# ifndef INTERPOLATE_VELOCITY_IBM3D_H
# define INTERPOLATE_VELOCITY_IBM3D_H
# include <cuda.h>

__global__ void interpolate_velocity_IBM3D(
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

# endif  // INTERPOLATE_VELOCITY_IBM3D_H