# ifndef MCMP_BOUNCE_BACK_D2Q9_H
# define MCMP_BOUNCE_BACK_D2Q9_H
# include <cuda.h>

__global__ void mcmp_bounce_back_D2Q9(float*,
                                      float*,
                                      int*,
									  int*,
                                      int*,
                                      int);

# endif  // MCMP_BOUNCE_BACK_D2Q9_H