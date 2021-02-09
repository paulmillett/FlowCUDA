# ifndef MOVE_PARTICLES_2D
# define MOVE_PARTICLES_2D
# include <cuda.h>

__global__ void move_particles_2D(float*,float*,float*,float*,float*,float*,int);

# endif  // MOVE_PARTICLES_2D