
# include "map_particles_to_grid_D2Q9.cuh"


// --------------------------------------------------------
// Map particles to grid by updating rS[] and pID[] arrays:
// --------------------------------------------------------

__global__ void map_particles_to_grid_D2Q9(float* rS,
                                           int* x,
										   int* y,										   
										   int* pID,										   
										   particle2D* p,
										   int nVoxels,
										   int nParticles)
{

	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
		
		// --------------------------------------------------	
		// default values:
		// --------------------------------------------------
				
		rS[i] = 0.0;
		pID[i] = -1;
		
		// --------------------------------------------------	
		// loop over particles:
		// --------------------------------------------------
		
		for (int j=0; j<nParticles; j++) {
			
			// ---------------------------	
			// distance to particle c.o.m:
			// ---------------------------
			
			float dx = float(x[i]) - p[j].rx;
			float dy = float(y[i]) - p[j].ry;
			float r2 = dx*dx + dy*dy;
			float r = sqrt(r2);			
			
			// ---------------------------	
			// assign values:
			// ---------------------------
			
			if (r <= p[j].rOuter) {
				if (r < p[j].rInner) {
					rS[i] = 1.0;
				}
				else {
					float rr = r - p[j].rInner;
					rS[i] = exp(-rr*rr/5.0);
				}
				pID[i] = j;
			}			
		}
							
	}
}
