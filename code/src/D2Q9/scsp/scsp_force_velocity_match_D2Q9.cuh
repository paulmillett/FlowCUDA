# ifndef SCSP_FORCE_VELOCITY_MATCH_D2Q9_H
# define SCSP_FORCE_VELOCITY_MATCH_D2Q9_H
# include <cuda.h>

__global__ void scsp_force_velocity_match_D2Q9(
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	float*,
	int);

# endif  // SCSP_FORCE_VELOCITY_MATCH_D2Q9_H