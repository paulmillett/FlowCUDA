
# include "mcmp_update_particles_on_lattice_D2Q9.cuh"
# include <stdio.h>

// --------------------------------------------------------
// D2Q9 kernel to update the particle fields on the lattice: 
// --------------------------------------------------------

__global__ void mcmp_update_particles_on_lattice_D2Q9(float* fA,
                                                      float* fB,
										              float* rA,
											          float* rB,
										              float* u,
										              float* v,
													  float* prx,
													  float* pry,
													  float* pvx,
													  float* pvy,
													  float* prad,
													  int* x,
													  int* y,
													  int* s,
													  int* pIDgrid,
													  int* nList,													  
										              int nVoxels,
													  int nParts)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {	
		
		int offst = 9*i;
		
		// --------------------------------------------	
		// get the current state of voxel:
		// --------------------------------------------
		
		int s0 = s[i];
		
		// --------------------------------------------	
		// get the new state of voxel by seeing if any
		// particles overlay it:
		// --------------------------------------------
				
		int s1 = 0;
		int pID = 0;
		float partvx = 0.0;
		float partvy = 0.0;
		for (int p=0; p<nParts; p++) {
			float dx = float(x[i]) - prx[p];
			float dy = float(y[i]) - pry[p];
			float rp = sqrt(dx*dx + dy*dy);
			if (rp <= prad[p]) {
				s1 = 1;
				pID = p;
				partvx = pvx[p];
				partvy = pvy[p];
			}		
		}
		s[i] = s1;
		
		// --------------------------------------------	
		// decide course of action:
		// --------------------------------------------
		
		// fluid site STAYS fluid site
		if (s0 == 0 && s1 == 0) {
			pIDgrid[i] = -1;  // this is redundant, but that's OK
		}
		
		// particle site STAYS particle site
		else if (s0 == 1 && s1 == 1) {			
			u[i] = partvx;
			v[i] = partvy;
			pIDgrid[i] = pID;
		}
		
		// fluid site becomes particle site (COVERING)
		else if (s0 == 0 && s1 == 1) {						
			// update velocity to particle's velocity
			u[i] = partvx;
			v[i] = partvy;
			pIDgrid[i] = pID;
			// zero all the populations			
			for (int n=0; n<9; n++) {
				fA[offst+n] = 0.0;
				fB[offst+n] = 0.0;
			}
		}
		
		// particle site becomes fluid site (UNCOVERING)
		else if (s0 == 1 && s1 == 0) {				
			// assign voxel velocity with particle velocity
			u[i] = pvx[pIDgrid[i]]; 
			v[i] = pvy[pIDgrid[i]];
			pIDgrid[i] = -1;
			// get average density of surrounding fluid sites:
			int num_fluid_nabors = 0;
			float aver_rA_nabors = 0.0;
			float aver_rB_nabors = 0.0;			
			for (int n=1; n<9; n++) {  // do not include self, n=0
				int nID = nList[offst+n];
				if (s[nID] == 0) {
					num_fluid_nabors++;
					aver_rA_nabors += rA[nID];
					aver_rB_nabors += rB[nID];
				}
			}			
			if (num_fluid_nabors > 0) {
				rA[i] = aver_rA_nabors/num_fluid_nabors;
				rB[i] = aver_rB_nabors/num_fluid_nabors;
			}
			// set populations to the equilibrium for the given
			// velocity and density:
			const float w0 = 4.0/9.0;
			const float ws = 1.0/9.0;
			const float wd = 1.0/36.0;
			const float omusq = 1.0 - 1.5*(u[i]*u[i] + v[i]*v[i]);		
			// dir 0
			float feq = w0*omusq;
			fA[offst+0] = feq*rA[i];
			fB[offst+0] = feq*rB[i];		
			// dir 1
			float evel = u[i];
			feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
			fA[offst+1] = feq*rA[i];
			fB[offst+1] = feq*rB[i];		
			// dir 2
			evel = v[i];
			feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
			fA[offst+2] = feq*rA[i];
			fB[offst+2] = feq*rB[i];		
			// dir 3
			evel = -u[i];
			feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
			fA[offst+3] = feq*rA[i];
			fB[offst+3] = feq*rB[i];		
			// dir 4
			evel = -v[i];
			feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
			fA[offst+4] = feq*rA[i];
			fB[offst+4] = feq*rB[i];		
			// dir 5
			evel = u[i] + v[i];
			feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
			fA[offst+5] = feq*rA[i];
			fB[offst+5] = feq*rB[i];		
			// dir 6
			evel = -u[i] + v[i];
			feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
			fA[offst+6] = feq*rA[i];
			fB[offst+6] = feq*rB[i];		
			// dir 7
			evel = -u[i] - v[i];
			feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
			fA[offst+7] = feq*rA[i];
			fB[offst+7] = feq*rB[i];		
			// dir 8
			evel = u[i] - v[i];
			feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
			fA[offst+8] = feq*rA[i];
			fB[offst+8] = feq*rB[i];
		}				
	}
}