# ifndef MCMP_MCMP_UPDATE_PARTICLES_ON_LATTICE_DIP_D2Q9_H
# define MCMP_MCMP_UPDATE_PARTICLES_ON_LATTICE_DIP_D2Q9_H
# include <cuda.h>
# include "particle2D_dip.cuh"

__global__ void mcmp_update_particles_on_lattice_dip_D2Q9(float*,
                                                          particle2D_dip*,											              
													      int*,
													      int*,
													      int*,
													      int,
                                                          int);

# endif  // MCMP_MCMP_UPDATE_PARTICLES_ON_LATTICE_PSM_D2Q9_H