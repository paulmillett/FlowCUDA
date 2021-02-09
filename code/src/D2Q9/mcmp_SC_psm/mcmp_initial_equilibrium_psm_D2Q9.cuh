# ifndef MCMP_INITIAL_EQUILIBRIUM_PSM_D2Q9_H
# define MCMP_INITIAL_EQUILIBRIUM_PSM_D2Q9_H
# include <cuda.h>

__global__ void mcmp_initial_equilibrium_psm_D2Q9(float*,
                                                  float*,
                                                  float*,
                                                  float*,
											      float*,
											      float*,
                                                  int);

# endif  // MCMP_INITIAL_EQUILIBRIUM_PSM_D2Q9_H