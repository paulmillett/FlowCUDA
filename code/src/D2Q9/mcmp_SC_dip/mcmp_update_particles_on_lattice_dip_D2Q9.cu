
# include "mcmp_update_particles_on_lattice_dip_D2Q9.cuh"
# include <stdio.h>

// --------------------------------------------------------
// D2Q9 kernel to update the particle fields on the lattice: 
// --------------------------------------------------------

__global__ void mcmp_update_particles_on_lattice_dip_D2Q9(float* rS,			                  
                                                          particle2D_dip* pt,
													      int* x,
													      int* y,
													      int* pIDgrid,
										                  int nVoxels,
													      int nParts)
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
				
		// --------------------------------------------------	
		// default values:
		// --------------------------------------------------
				
		rS[i] = 0.0;
		pIDgrid[i] = -1;
		
		// --------------------------------------------------	
		// loop over particles:
		// --------------------------------------------------
		
		for (int j=0; j<nParts; j++) {
			
			// ---------------------------	
			// distance to particle c.o.m:
			// ---------------------------
			
			float dx = float(x[i]) - pt[j].r.x;
			float dy = float(y[i]) - pt[j].r.y;
			float r = sqrt(dx*dx + dy*dy);
						
			// ---------------------------	
			// assign values:
			// ---------------------------
			
			float rI = pt[j].rInner;
			float rO = pt[j].rOuter;			
			if (r <= rO) {
				if (r < rI) {
					rS[i] = 1.0;
				}
				else {
					float rr = r - rI;
					rS[i] = 1.0 - rr/(rO-rI);					
				}
				pIDgrid[i] = j;			
			}			
		}
	}
}