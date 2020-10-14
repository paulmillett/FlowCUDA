# ifndef MCMP_COMPUTE_VELOCITY_D2Q9
# define MCMP_COMPUTE_VELOCITY_D2Q9
# include <cuda.h>

__global__ void mcmp_compute_velocity_D2Q9(float*,
                                           float*,
                                           float*,
                                           float*,
										   float*,
										   float*,
										   float*,
										   float*,
										   float*,
										   float*,
                                           int);

# endif  // MCMP_COMPUTE_VELOCITY_D2Q9