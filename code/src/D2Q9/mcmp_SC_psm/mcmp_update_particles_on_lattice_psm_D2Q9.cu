
# include "mcmp_update_particles_on_lattice_psm_D2Q9.cuh"
# include <stdio.h>

// --------------------------------------------------------
// D2Q9 kernel to update the particle fields on the lattice: 
// --------------------------------------------------------

__global__ void mcmp_update_particles_on_lattice_psm_D2Q9(float* B,			                  
													      float* prx,
													      float* pry,
													      float* rOuter,
													      float* rInner,
													      int* x,
													      int* y,
													      int* pIDgrid,
														  float nu,
										                  int nVoxels,
													      int nParts)
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
				
		// --------------------------------------------------	
		// default values:
		// --------------------------------------------------
				
		B[i] = 0.0;
		pIDgrid[i] = -1;
		
		// --------------------------------------------------	
		// loop over particles:
		// --------------------------------------------------
		
		for (int j=0; j<nParts; j++) {
			
			// ---------------------------	
			// distance to particle c.o.m:
			// ---------------------------
			
			float dx = float(x[i]) - prx[j];
			float dy = float(y[i]) - pry[j];
			float r = sqrt(dx*dx + dy*dy);
						
			// ---------------------------	
			// assign values:
			// ---------------------------
			
			float rI = rInner[j];
			float rO = rOuter[j];			
			if (r <= rO) {
				if (r < rI) {
					B[i] = 1.0;
				}
				else {
					float rr = r - rI;
					B[i] = 1.0 - rr/(rO-rI);					
				}
				pIDgrid[i] = j;				
			}			
		}
	}
}