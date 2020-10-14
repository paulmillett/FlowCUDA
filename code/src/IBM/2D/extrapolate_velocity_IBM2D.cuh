# ifndef EXTRAPOLATE_VELOCITY_IBM2D_H
# define EXTRAPOLATE_VELOCITY_IBM2D_H
# include <cuda.h>

__global__ void extrapolate_velocity_IBM2D(
	float*,
    float*,
    float*,
	float*,
    float*,
	float*,
	float*,
	int,
    int);

# endif  // EXTRAPOLATE_VELOCITY_IBM2D_H