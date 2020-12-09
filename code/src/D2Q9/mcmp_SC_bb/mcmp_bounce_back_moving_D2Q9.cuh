# ifndef MCMP_BOUNCE_BACK_MOVING_D2Q9_H
# define MCMP_BOUNCE_BACK_MOVING_D2Q9_H
# include <cuda.h>

__global__ void mcmp_bounce_back_moving_D2Q9(float*,
                                             float*,
											 float*,
											 float*,
											 float*,
											 float*,												 
                                             int*,
									         int*,
                                             int*,
                                             int);

# endif  // MCMP_BOUNCE_BACK_MOVING_D2Q9_H