# ifndef MCMP_INITIAL_EQUILIBRIUM_D2Q9
# define MCMP_INITIAL_EQUILIBRIUM_D2Q9
# include <cuda.h>

__global__ void mcmp_initial_equilibrium_D2Q9(float*,
                                              float*,
                                              float*,
                                              float*,
											  float*,
											  float*,
                                              int);

# endif  // MCMP_INITIAL_EQUILIBRIUM_D2Q9