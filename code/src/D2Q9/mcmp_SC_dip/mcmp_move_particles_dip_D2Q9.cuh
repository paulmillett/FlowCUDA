# ifndef MCMP_MOVE_PARTICLES_DIP_D2Q9_H
# define MCMP_MOVE_PARTICLES_DIP_D2Q9_H
# include <cuda.h>
# include "particle2D_dip.cuh"

__global__ void mcmp_move_particles_dip_D2Q9(particle2D_dip*,int);

# endif  // MCMP_MOVE_PARTICLES_DIP_D2Q9_H