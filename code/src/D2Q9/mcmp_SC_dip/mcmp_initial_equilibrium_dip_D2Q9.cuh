# ifndef MCMP_INITIAL_EQUILIBRIUM_DIP_D2Q9_H
# define MCMP_INITIAL_EQUILIBRIUM_DIP_D2Q9_H
# include <cuda.h>

__global__ void mcmp_initial_equilibrium_dip_D2Q9(float*,
                                                  float*,
                                                  float*,
                                                  float*,
											      float*,
											      float*,
                                                  int);

# endif  // MCMP_INITIAL_EQUILIBRIUM_DIP_D2Q9_H