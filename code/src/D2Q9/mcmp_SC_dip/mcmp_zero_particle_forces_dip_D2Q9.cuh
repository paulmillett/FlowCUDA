# ifndef MCMP_ZERO_PARTICLE_FORCES_DIP_D2Q9_H
# define MCMP_ZERO_PARTICLE_FORCES_DIP_D2Q9_H
# include <cuda.h>
# include "particle2D_dip.cuh"

__global__ void mcmp_zero_particle_forces_dip_D2Q9(particle2D_dip*,int); 

# endif  // MCMP_ZERO_PARTICLE_FORCES_DIP_D2Q9_H