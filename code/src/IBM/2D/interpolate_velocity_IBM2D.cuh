# ifndef INTERPOLATE_VELOCITY_IBM2D_H
# define INTERPOLATE_VELOCITY_IBM2D_H
# include <cuda.h>

__global__ void interpolate_velocity_IBM2D(
	float*,
    float*,
    float*,
	float*,
    float*,
	float*,
	int,
    int);

# endif  // INTERPOLATE_VELOCITY_IBM2D_H