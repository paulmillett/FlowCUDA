# ifndef MCMP_COMPUTE_VELOCITY_SOLID_D2Q9_H
# define MCMP_COMPUTE_VELOCITY_SOLID_D2Q9_H
# include <cuda.h>
# include "../particles/particle_struct_D2Q9.cuh"

__global__ void mcmp_compute_velocity_solid_D2Q9(float*,
                                                 float*,
                                                 float*,
                                                 float*,
												 float*,
										         float*,
										         float*,
										         float*,
										         float*,
										         float*,
										         float*,
												 int*,
												 particle2D*,
                                                 int);

# endif  // MCMP_COMPUTE_VELOCITY_SOLID_D2Q9_H