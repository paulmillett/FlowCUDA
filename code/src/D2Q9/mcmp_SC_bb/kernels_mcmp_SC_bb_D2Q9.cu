
# include "kernels_mcmp_SC_bb_D2Q9.cuh"
# include "../mcmp_SC/mcmp_pseudopotential.cuh"
# include <stdio.h>



// --------------------------------------------------------
// D2Q9 initialize kernel: 
// --------------------------------------------------------

__global__ void mcmp_initial_equilibrium_bb_D2Q9(float* fA,
                                                 float* fB,
										         float* rA,
											     float* rB,
										         float* u,
										         float* v,
										         int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	// initialize populations to equilibrium values:
	if (i < nVoxels) {	
		
		int offst = 9*i;
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
			// zero all the populations			
			for (int n=0; n<9; n++) {
				fA[offst+n] = 0.0;
				fB[offst+n] = 0.0;
			}
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



// --------------------------------------------------------
// D2Q9 compute velocity (barycentric) for the system: 
// --------------------------------------------------------

__global__ void mcmp_compute_velocity_bb_D2Q9(float* fA,
                                              float* fB,
										      float* rA,
										      float* rB,
										      float* FxA,
										      float* FxB,
										      float* FyA,
										      float* FyB,
										      float* u,
										      float* v,
											  int* s,
										      int nVoxels) 
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		
		// --------------------------------------------------	
		// if this is a fluid site:
		// --------------------------------------------------
		
		if (s[i] == 0) {
			int offst = i*9;			
			float uA = fA[offst+1] + fA[offst+5] + fA[offst+8] - (fA[offst+3] + fA[offst+6] + fA[offst+7]) + 0.5*FxA[i];
			float uB = fB[offst+1] + fB[offst+5] + fB[offst+8] - (fB[offst+3] + fB[offst+6] + fB[offst+7]) + 0.5*FxB[i];
			float vA = fA[offst+2] + fA[offst+5] + fA[offst+6] - (fA[offst+4] + fA[offst+7] + fA[offst+8]) + 0.5*FyA[i];
			float vB = fB[offst+2] + fB[offst+5] + fB[offst+6] - (fB[offst+4] + fB[offst+7] + fB[offst+8]) + 0.5*FyB[i];
			float rTotal = rA[i] + rB[i];
			u[i] = (uA + uB)/rTotal;
			v[i] = (vA + vB)/rTotal;	
		}
		
		// --------------------------------------------------	
		// if this is a solid site:
		// --------------------------------------------------
		
		else if (s[i] == 1) {
			//u[i] = 0.0;  // later, fill in with particle velocity
			//v[i] = 0.0;  
		}
					
	}
}



// --------------------------------------------------------
// D2Q9 compute density for each component: 
// --------------------------------------------------------

__global__ void mcmp_compute_density_bb_D2Q9(float* fA,
                                        	 float* fB,
										     float* rA,
										     float* rB,
										     int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		int offst = i*9;			
		rA[i] = fA[offst] + fA[offst+1] + fA[offst+2] + fA[offst+3] + fA[offst+4] + fA[offst+5] + fA[offst+6] +
		        fA[offst+7] + fA[offst+8];
		rB[i] = fB[offst] + fB[offst+1] + fB[offst+2] + fB[offst+3] + fB[offst+4] + fB[offst+5] + fB[offst+6] +
		        fB[offst+7] + fB[offst+8];
	}
}



// --------------------------------------------------------
// D2Q9 compute Shan-Chen forces for the components
// using pseudo-potential, psi = rho_0(1-exp(-rho/rho_o))
// --------------------------------------------------------

__global__ void mcmp_compute_SC_forces_bb_D2Q9(float* rA,
										       float* rB,
										       float* FxA,
										       float* FxB,
										       float* FyA,
										       float* FyB,
											   int* s,
											   int* nList,
											   float gAB,
											   float gAS,
											   float gBS,
										       int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		
		// --------------------------------------------	
		// if this is a fluid site:
		// --------------------------------------------
		
		if (s[i] == 0) {
			
			int offst = i*9;
			
			float r0A = psi(rA[i]);	 			
			float r1A = psi(rA[nList[offst+1]]);
			float r2A = psi(rA[nList[offst+2]]);
			float r3A = psi(rA[nList[offst+3]]);
			float r4A = psi(rA[nList[offst+4]]);
			float r5A = psi(rA[nList[offst+5]]);
			float r6A = psi(rA[nList[offst+6]]);
			float r7A = psi(rA[nList[offst+7]]);
			float r8A = psi(rA[nList[offst+8]]);
		
			float r0B = psi(rB[i]);		
			float r1B = psi(rB[nList[offst+1]]);
			float r2B = psi(rB[nList[offst+2]]);
			float r3B = psi(rB[nList[offst+3]]);
			float r4B = psi(rB[nList[offst+4]]);
			float r5B = psi(rB[nList[offst+5]]);
			float r6B = psi(rB[nList[offst+6]]);
			float r7B = psi(rB[nList[offst+7]]);
			float r8B = psi(rB[nList[offst+8]]);
			
			float s1 = float(s[nList[offst+1]]);
			float s2 = float(s[nList[offst+2]]);
			float s3 = float(s[nList[offst+3]]);
			float s4 = float(s[nList[offst+4]]);
			float s5 = float(s[nList[offst+5]]);
			float s6 = float(s[nList[offst+6]]);
			float s7 = float(s[nList[offst+7]]);
			float s8 = float(s[nList[offst+8]]);
		
			float ws = 1.0/9.0;
			float wd = 1.0/36.0;		
			float sumNbrRhoAx = ws*r1A + wd*r5A + wd*r8A - (ws*r3A + wd*r6A + wd*r7A);
			float sumNbrRhoAy = ws*r2A + wd*r5A + wd*r6A - (ws*r4A + wd*r7A + wd*r8A);
			float sumNbrRhoBx = ws*r1B + wd*r5B + wd*r8B - (ws*r3B + wd*r6B + wd*r7B);
			float sumNbrRhoBy = ws*r2B + wd*r5B + wd*r6B - (ws*r4B + wd*r7B + wd*r8B);
			float sumNbrSx = ws*s1 + wd*s5 + wd*s8 - (ws*s3 + wd*s6 + wd*s7);
			float sumNbrSy = ws*s2 + wd*s5 + wd*s6 - (ws*s4 + wd*s7 + wd*s8);
			
			FxA[i] = -r0A*(gAB*sumNbrRhoBx + gAS*sumNbrSx);
			FxB[i] = -r0B*(gAB*sumNbrRhoAx + gBS*sumNbrSx);
			FyA[i] = -r0A*(gAB*sumNbrRhoBy + gAS*sumNbrSy);
			FyB[i] = -r0B*(gAB*sumNbrRhoAy + gBS*sumNbrSy);
						
		}		
	}
}



// --------------------------------------------------------
// D2Q9 update kernel:
// --------------------------------------------------------

__global__ void mcmp_collide_stream_bb_D2Q9(float* f1A,
                                         	float* f1B,
										 	float* f2A,
										 	float* f2B,
										 	float* rA,
										 	float* rB,
										 	float* u,
										 	float* v,
										 	float* FxA,
										 	float* FxB,
										 	float* FyA,
										 	float* FyB,
											int* s,
										 	int* streamIndex,
										 	float nu,
										 	int nVoxels)
{

	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
		
		// --------------------------------------------------	
		// Only do collide & stream if "i" is a fluid node:
		// --------------------------------------------------
		
		if (s[i] == 0) {
			
			// --------------------------------------------------	
			// FORCING - this step includes the Guo forcing
			//           scheme applied to the Shan-Chen
			//           MCMP model according to Kruger et al.
			// --------------------------------------------------
		
			float w0 = 4.0/9.0;
			float ws = 1.0/9.0;
			float wd = 1.0/36.0;
		
			float evel = 0.0;       // e dot velocity
			float emiu = 0.0-u[i];  // e minus u
			float emiv = 0.0-v[i];  // e minus v
			float frc0A = w0*( FxA[i]*(3.0*emiu) + FyA[i]*(3.0*emiv) );
			float frc0B = w0*( FxB[i]*(3.0*emiu) + FyB[i]*(3.0*emiv) );
		
			evel = u[i];
			emiu = 1.0-u[i];
			emiv = 0.0-v[i];
			float frc1A = ws*( FxA[i]*(3.0*emiu + 9.0*evel) + FyA[i]*(3.0*emiv) );
			float frc1B = ws*( FxB[i]*(3.0*emiu + 9.0*evel) + FyB[i]*(3.0*emiv) );
		
			evel = v[i]; 
			emiu = 0.0-u[i];
			emiv = 1.0-v[i];
			float frc2A = ws*( FxA[i]*(3.0*emiu) + FyA[i]*(3.0*emiv + 9.0*evel) );
			float frc2B = ws*( FxB[i]*(3.0*emiu) + FyB[i]*(3.0*emiv + 9.0*evel) );
		
			evel = -u[i];
			emiu = -1.0-u[i];
			emiv =  0.0-v[i];
			float frc3A = ws*( FxA[i]*(3.0*emiu - 9.0*evel) + FyA[i]*(3.0*emiv) );
			float frc3B = ws*( FxB[i]*(3.0*emiu - 9.0*evel) + FyB[i]*(3.0*emiv) );
		
			evel = -v[i];
			emiu =  0.0-u[i];
			emiv = -1.0-v[i];
			float frc4A = ws*( FxA[i]*(3.0*emiu) + FyA[i]*(3.0*emiv - 9.0*evel) );
			float frc4B = ws*( FxB[i]*(3.0*emiu) + FyB[i]*(3.0*emiv - 9.0*evel) );
		
			evel = u[i] + v[i];
			emiu = 1.0-u[i];
			emiv = 1.0-v[i];
			float frc5A = wd*( FxA[i]*(3.0*emiu + 9.0*evel) + FyA[i]*(3.0*emiv + 9.0*evel) );
			float frc5B = wd*( FxB[i]*(3.0*emiu + 9.0*evel) + FyB[i]*(3.0*emiv + 9.0*evel) );
		
			evel = -u[i] + v[i];
			emiu = -1.0-u[i];
			emiv =  1.0-v[i];
			float frc6A = wd*( FxA[i]*(3.0*emiu - 9.0*evel) + FyA[i]*(3.0*emiv + 9.0*evel) );
			float frc6B = wd*( FxB[i]*(3.0*emiu - 9.0*evel) + FyB[i]*(3.0*emiv + 9.0*evel) );
				
			evel = -u[i] - v[i];
			emiu = -1.0-u[i];
			emiv = -1.0-v[i];
			float frc7A = wd*( FxA[i]*(3.0*emiu - 9.0*evel) + FyA[i]*(3.0*emiv - 9.0*evel) );
			float frc7B = wd*( FxB[i]*(3.0*emiu - 9.0*evel) + FyB[i]*(3.0*emiv - 9.0*evel) );
		
			evel = u[i] - v[i];
			emiu =  1.0-u[i];
			emiv = -1.0-v[i];
			float frc8A = wd*( FxA[i]*(3.0*emiu + 9.0*evel) + FyA[i]*(3.0*emiv - 9.0*evel) );
			float frc8B = wd*( FxB[i]*(3.0*emiu + 9.0*evel) + FyB[i]*(3.0*emiv - 9.0*evel) );
				
			// --------------------------------------------------	
			// COLLISION & STREAMING - standard BGK operator with
			//                         a PUSH propagator.
			// --------------------------------------------------
		
			int offst = 9*i;
			const float omega = 2.0/(6.0*nu + 1.0);   // 1/tau
			const float omomega = 1.0 - omega;        // 1 - 1/tau
			const float omomega2 = 1.0 - 0.5*omega;   // 1 - 1/(2tau)
			const float omusq = 1.0 - 1.5*(u[i]*u[i] + v[i]*v[i]);
		
			// dir 0
			float feq = w0*omusq;
			f2A[streamIndex[offst+0]] = omomega*f1A[offst+0] + omega*feq*rA[i] + omomega2*frc0A;
			f2B[streamIndex[offst+0]] = omomega*f1B[offst+0] + omega*feq*rB[i] + omomega2*frc0B;
		
			// dir 1
			evel = u[i];
			feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
			f2A[streamIndex[offst+1]] = omomega*f1A[offst+1] + omega*feq*rA[i] + omomega2*frc1A;
			f2B[streamIndex[offst+1]] = omomega*f1B[offst+1] + omega*feq*rB[i] + omomega2*frc1B;
		
			// dir 2
			evel = v[i];
			feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
			f2A[streamIndex[offst+2]] = omomega*f1A[offst+2] + omega*feq*rA[i] + omomega2*frc2A;
			f2B[streamIndex[offst+2]] = omomega*f1B[offst+2] + omega*feq*rB[i] + omomega2*frc2B;
		
			// dir 3
			evel = -u[i];
			feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
			f2A[streamIndex[offst+3]] = omomega*f1A[offst+3] + omega*feq*rA[i] + omomega2*frc3A;
			f2B[streamIndex[offst+3]] = omomega*f1B[offst+3] + omega*feq*rB[i] + omomega2*frc3B;
		
			// dir 4
			evel = -v[i];
			feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
			f2A[streamIndex[offst+4]] = omomega*f1A[offst+4] + omega*feq*rA[i] + omomega2*frc4A;
			f2B[streamIndex[offst+4]] = omomega*f1B[offst+4] + omega*feq*rB[i] + omomega2*frc4B;
		
			// dir 5
			evel = u[i] + v[i];
			feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
			f2A[streamIndex[offst+5]] = omomega*f1A[offst+5] + omega*feq*rA[i] + omomega2*frc5A;
			f2B[streamIndex[offst+5]] = omomega*f1B[offst+5] + omega*feq*rB[i] + omomega2*frc5B;
		
			// dir 6
			evel = -u[i] + v[i];
			feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
			f2A[streamIndex[offst+6]] = omomega*f1A[offst+6] + omega*feq*rA[i] + omomega2*frc6A;
			f2B[streamIndex[offst+6]] = omomega*f1B[offst+6] + omega*feq*rB[i] + omomega2*frc6B;
		
			// dir 7
			evel = -u[i] - v[i];
			feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
			f2A[streamIndex[offst+7]] = omomega*f1A[offst+7] + omega*feq*rA[i] + omomega2*frc7A;
			f2B[streamIndex[offst+7]] = omomega*f1B[offst+7] + omega*feq*rB[i] + omomega2*frc7B;
		
			// dir 8
			evel = u[i] - v[i];
			feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
			f2A[streamIndex[offst+8]] = omomega*f1A[offst+8] + omega*feq*rA[i] + omomega2*frc8A;
			f2B[streamIndex[offst+8]] = omomega*f1B[offst+8] + omega*feq*rB[i] + omomega2*frc8B;
			
		}					
	}
}



// --------------------------------------------------------
// D2Q9 implement bounce-back conditions:
// --------------------------------------------------------

__global__ void mcmp_bounce_back_D2Q9(float* f2A, 
									  float* f2B,
									  int* s,
									  int* nList,									  
									  int* streamIndex,
									  int nVoxels)
{

	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
								
		// --------------------------------------------------	
		// If the current voxel is solid, then bounce-back
		// the populations just received via streaming 
		// back to the neighboring voxel:
		// --------------------------------------------------
		
		if (s[i] == 1) {
			
			int offst = 9*i;
						
			// dir 1 bounce-back to nabor 3 as dir 3:
			if (s[nList[offst+3]] == 0) {
				f2A[streamIndex[offst+3]] = f2A[offst+1];
				f2B[streamIndex[offst+3]] = f2B[offst+1];
				f2A[offst+1] = 0.0;
				f2B[offst+1] = 0.0;
			}
			
			// dir 2 bounce-back to nabor 4 as dir 4:
			if (s[nList[offst+4]] == 0) {
				f2A[streamIndex[offst+4]] = f2A[offst+2];
				f2B[streamIndex[offst+4]] = f2B[offst+2];
				f2A[offst+2] = 0.0;
				f2B[offst+2] = 0.0;
			}
			
			// dir 3 bounce-back to nabor 1 as dir 1:
			if (s[nList[offst+1]] == 0) {
				f2A[streamIndex[offst+1]] = f2A[offst+3];
				f2B[streamIndex[offst+1]] = f2B[offst+3];
				f2A[offst+3] = 0.0;
				f2B[offst+3] = 0.0;
			}
			
			// dir 4 bounce-back to nabor 2 as dir 2:
			if (s[nList[offst+2]] == 0) {
				f2A[streamIndex[offst+2]] = f2A[offst+4];
				f2B[streamIndex[offst+2]] = f2B[offst+4];
				f2A[offst+4] = 0.0;
				f2B[offst+4] = 0.0;
			}
			
			// dir 5 bounce-back to nabor 7 as dir 7:
			if (s[nList[offst+7]] == 0) {
				f2A[streamIndex[offst+7]] = f2A[offst+5];
				f2B[streamIndex[offst+7]] = f2B[offst+5];
				f2A[offst+5] = 0.0;
				f2B[offst+5] = 0.0;
			}
			
			// dir 6 bounce-back to nabor 8 as dir 8:
			if (s[nList[offst+8]] == 0) {
				f2A[streamIndex[offst+8]] = f2A[offst+6];
				f2B[streamIndex[offst+8]] = f2B[offst+6];
				f2A[offst+6] = 0.0;
				f2B[offst+6] = 0.0;
			}
			
			// dir 7 bounce-back to nabor 5 as dir 5:
			if (s[nList[offst+5]] == 0) {
				f2A[streamIndex[offst+5]] = f2A[offst+7];
				f2B[streamIndex[offst+5]] = f2B[offst+7];
				f2A[offst+7] = 0.0;
				f2B[offst+7] = 0.0;
			}
			
			// dir 8 bounce-back to nabor 6 as dir 6:
			if (s[nList[offst+6]] == 0) {
				f2A[streamIndex[offst+6]] = f2A[offst+8];
				f2B[streamIndex[offst+6]] = f2B[offst+8];
				f2A[offst+8] = 0.0;
				f2B[offst+8] = 0.0;
			}
			
		}	
	}		
}



// --------------------------------------------------------
// D2Q9 implement bounce-back conditions for moving
// solids:
// --------------------------------------------------------

__global__ void mcmp_bounce_back_moving_D2Q9(float* f2A, 
									         float* f2B,
											 float* rA,
											 float* rB,
											 float* u,
											 float* v,
									         int* s,
									         int* nList,									  
									         int* streamIndex,
									         int nVoxels)
{

	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
								
		// --------------------------------------------------	
		// If the current voxel is solid, then bounce-back
		// the populations just received via streaming 
		// back to the neighboring voxel:
		// --------------------------------------------------
		
		if (s[i] == 1) {
			
			int offst = 9*i;
			const float ws = 1.0/9.0;
			const float wd = 1.0/36.0;
						
			// dir 1 bounce-back to nabor 3 as dir 3:
			if (s[nList[offst+3]] == 0) {
				float evel = u[i];
				f2A[streamIndex[offst+3]] = f2A[offst+1] - 6.0*ws*rA[nList[offst+3]]*evel;
				f2B[streamIndex[offst+3]] = f2B[offst+1] - 6.0*ws*rB[nList[offst+3]]*evel;
				f2A[offst+1] = 0.0;
				f2B[offst+1] = 0.0;
			}
			
			// dir 2 bounce-back to nabor 4 as dir 4:
			if (s[nList[offst+4]] == 0) {
				float evel = v[i];
				f2A[streamIndex[offst+4]] = f2A[offst+2] - 6.0*ws*rA[nList[offst+4]]*evel;
				f2B[streamIndex[offst+4]] = f2B[offst+2] - 6.0*ws*rB[nList[offst+4]]*evel;
				f2A[offst+2] = 0.0;
				f2B[offst+2] = 0.0;
			}
			
			// dir 3 bounce-back to nabor 1 as dir 1:
			if (s[nList[offst+1]] == 0) {
				float evel = -u[i];
				f2A[streamIndex[offst+1]] = f2A[offst+3] - 6.0*ws*rA[nList[offst+1]]*evel;
				f2B[streamIndex[offst+1]] = f2B[offst+3] - 6.0*ws*rB[nList[offst+1]]*evel;
				f2A[offst+3] = 0.0;
				f2B[offst+3] = 0.0;
			}
			
			// dir 4 bounce-back to nabor 2 as dir 2:
			if (s[nList[offst+2]] == 0) {
				float evel = -v[i];
				f2A[streamIndex[offst+2]] = f2A[offst+4] - 6.0*ws*rA[nList[offst+2]]*evel;
				f2B[streamIndex[offst+2]] = f2B[offst+4] - 6.0*ws*rB[nList[offst+2]]*evel;
				f2A[offst+4] = 0.0;
				f2B[offst+4] = 0.0;
			}
			
			// dir 5 bounce-back to nabor 7 as dir 7:
			if (s[nList[offst+7]] == 0) {
				float evel = u[i] + v[i];
				f2A[streamIndex[offst+7]] = f2A[offst+5] - 6.0*wd*rA[nList[offst+7]]*evel;
				f2B[streamIndex[offst+7]] = f2B[offst+5] - 6.0*wd*rB[nList[offst+7]]*evel;
				f2A[offst+5] = 0.0;
				f2B[offst+5] = 0.0;
			}
			
			// dir 6 bounce-back to nabor 8 as dir 8:
			if (s[nList[offst+8]] == 0) {
				float evel = -u[i] + v[i];
				f2A[streamIndex[offst+8]] = f2A[offst+6] - 6.0*wd*rA[nList[offst+8]]*evel;
				f2B[streamIndex[offst+8]] = f2B[offst+6] - 6.0*wd*rB[nList[offst+8]]*evel;
				f2A[offst+6] = 0.0;
				f2B[offst+6] = 0.0;
			}
			
			// dir 7 bounce-back to nabor 5 as dir 5:
			if (s[nList[offst+5]] == 0) {
				float evel = -u[i] - v[i];
				f2A[streamIndex[offst+5]] = f2A[offst+7] - 6.0*wd*rA[nList[offst+5]]*evel;
				f2B[streamIndex[offst+5]] = f2B[offst+7] - 6.0*wd*rB[nList[offst+5]]*evel;
				f2A[offst+7] = 0.0;
				f2B[offst+7] = 0.0;
			}
			
			// dir 8 bounce-back to nabor 6 as dir 6:
			if (s[nList[offst+6]] == 0) {
				float evel = u[i] - v[i];
				f2A[streamIndex[offst+6]] = f2A[offst+8] - 6.0*wd*rA[nList[offst+6]]*evel;
				f2B[streamIndex[offst+6]] = f2B[offst+8] - 6.0*wd*rB[nList[offst+6]]*evel;
				f2A[offst+8] = 0.0;
				f2B[offst+8] = 0.0;
			}
			
		}	
	}		
}



