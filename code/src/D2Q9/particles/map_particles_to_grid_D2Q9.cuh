# ifndef MAP_PARTICLES_TO_GRID_D2Q9_H
# define MAP_PARTICLES_TO_GRID_D2Q9_H
# include <cuda.h>
# include "particle_struct_D2Q9.cuh"

__global__ void map_particles_to_grid_D2Q9(float*,
                                           int*,
										   int*,
										   int*,
										   particle2D*,
										   int,
										   int);

# endif  // MAP_PARTICLES_TO_GRID_D2Q9_H