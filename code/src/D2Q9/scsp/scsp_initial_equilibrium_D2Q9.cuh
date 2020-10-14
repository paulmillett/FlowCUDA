# ifndef SCSP_INITIAL_EQUILIBRIUM_D2Q9_H
# define SCSP_INITIAL_EQUILIBRIUM_D2Q9_H
# include <cuda.h>

__global__ void scsp_initial_equilibrium_D2Q9(float*,
                                              float*,
                                              float*,
                                              float*,
                                              int);

# endif  // SCSP_INITIAL_EQUILIBRIUM_D2Q9_H