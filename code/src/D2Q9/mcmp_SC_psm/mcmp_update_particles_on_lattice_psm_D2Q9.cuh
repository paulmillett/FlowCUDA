# ifndef MCMP_MCMP_UPDATE_PARTICLES_ON_LATTICE_PSM_D2Q9_H
# define MCMP_MCMP_UPDATE_PARTICLES_ON_LATTICE_PSM_D2Q9_H
# include <cuda.h>

__global__ void mcmp_update_particles_on_lattice_psm_D2Q9(float*,
                                                          float*,
                                                          float*,
                                                          float*,
											              float*,											              
													      int*,
													      int*,
													      int*,
														  float,													      
													      int,
                                                          int);

# endif  // MCMP_MCMP_UPDATE_PARTICLES_ON_LATTICE_PSM_D2Q9_H