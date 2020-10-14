
# include "mcmp_compute_SC_pressure_D2Q9.cuh"
# include "mcmp_pseudopotential.cuh"



// --------------------------------------------------------
// D2Q9 compute Shan-Chen pressure for the components: 
// --------------------------------------------------------

__global__ void mcmp_compute_SC_pressure_D2Q9(float* rA,
										      float* rB,
											  float* pr,											
											  float gAB,
										      int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {
		pr[i] = (rA[i]+rB[i])/3.0 + gAB*psi(rA[i])*psi(rB[i])/6.0;		
	}
}

