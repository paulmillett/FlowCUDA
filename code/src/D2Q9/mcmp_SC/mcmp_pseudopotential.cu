
# include "mcmp_pseudopotential.cuh"



// --------------------------------------------------------
// pseudopotential as a function of density: 
// --------------------------------------------------------

__device__ float psi(float r)
{
	float r0 = 1.0;
	return r0*(1.0 - exp(-r/r0));
}