# ifndef EXTRAPOLATE_FORCE_IBM2D_H
# define EXTRAPOLATE_FORCE_IBM2D_H
# include <cuda.h>

__global__ void extrapolate_force_IBM2D(
	float*,
    float*,
    float*,
	float*,
    float*,
	float*,
	int,
    int);

# endif  // EXTRAPOLATE_FORCE_IBM2D_H