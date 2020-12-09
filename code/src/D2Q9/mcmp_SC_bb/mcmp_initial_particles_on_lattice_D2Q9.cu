
# include "mcmp_initial_particles_on_lattice_D2Q9.cuh"



// --------------------------------------------------------
// Map particles to grid by updating rS[] and pID[] arrays:
// --------------------------------------------------------

__global__ void mcmp_initial_particles_on_lattice_D2Q9(float* prx,
                                                       float* pry,
					  								   float* prad,
                                                       int* x,
										               int* y,
													   int* s,									   
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
				
		s[i] = 0;
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
			float rr = sqrt(dx*dx + dy*dy);
			
			// ---------------------------	
			// assign values:
			// ---------------------------
			
			if (rr <= prad[j]) {
				s[i] = 1;
				pIDgrid[i] = j;	
			}		
		}							
	}
}
