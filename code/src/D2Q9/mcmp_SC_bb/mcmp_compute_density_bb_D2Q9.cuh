# ifndef MCMP_COMPUTE_DENSITY_BB_D2Q9
# define MCMP_COMPUTE_DENSITY_BB_D2Q9
# include <cuda.h>

__global__ void mcmp_compute_density_bb_D2Q9(float*,
                                             float*,
                                          	 float*,
                                          	 float*,
                                          	 int);

# endif  // MCMP_COMPUTE_DENSITY_BB_D2Q9