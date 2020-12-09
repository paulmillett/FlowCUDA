# ifndef MCMP_INITIAL_PARTICLES_ON_LATTICE_D2Q9_H
# define MCMP_INITIAL_PARTICLES_ON_LATTICE_D2Q9_H
# include <cuda.h>

__global__ void mcmp_initial_particles_on_lattice_D2Q9(float*,float*,float*,int*,int*,int*,int*,int,int);

# endif  // MCMP_INITIAL_PARTICLES_ON_LATTICE_D2Q9_H