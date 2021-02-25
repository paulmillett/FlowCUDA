
# include "kernels_mcmp_SC_psm_D2Q9.cuh"
# include "../mcmp_SC/mcmp_pseudopotential.cuh"
# include <stdio.h>



// --------------------------------------------------------
// D2Q9 initialize kernel: 
// --------------------------------------------------------

__global__ void mcmp_initial_equilibrium_psm_D2Q9(float* fA,
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



// --------------------------------------------------------
// D2Q9 set velocity on the y=0 and y=Ny-1 boundaries: 
// --------------------------------------------------------

__global__ void mcmp_set_boundary_velocity_psm_D2Q9(float* rA,
										            float* rB,
										            float* FxA,
										            float* FxB,
										            float* FyA,
										            float* FyB,
										            float* u,
										            float* v,
													int* y,											        
											        int Ny,
										            int nVoxels) 
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		if (y[i] == 0 || y[i] == Ny-1) {
			float uBC = 0.05;
			float vBC = 0.00;
			float rTotal = rA[i] + rB[i];
			float fxBC = (uBC - u[i])*2.0*rTotal;
			float fyBC = (vBC - v[i])*2.0*rTotal;
			u[i] += 0.5*fxBC/rTotal;
			v[i] += 0.5*fyBC/rTotal;
			FxA[i] += fxBC*(rA[i]/rTotal);
			FxB[i] += fxBC*(rB[i]/rTotal);
			FyA[i] += fyBC*(rA[i]/rTotal);
			FyB[i] += fyBC*(rB[i]/rTotal);
		}		
	}
}



// --------------------------------------------------------
// D2Q9 compute density for each component: 
// --------------------------------------------------------

__global__ void mcmp_compute_density_psm_D2Q9(float* fA,
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
// D2Q9 compute velocity (barycentric) for the system: 
// --------------------------------------------------------

__global__ void mcmp_compute_velocity_psm_D2Q9(float* fA,
                                               float* fB,
										       float* rA,
										       float* rB,
										       float* FxA,
										       float* FxB,
										       float* FyA,
										       float* FyB,
										       float* u,
										       float* v,
											   float* pvx,
											   float* pvy,
											   float* pfx,
											   float* pfy,
											   float* B,
											   int* pIDgrid,
										       int nVoxels) 
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		// barycentric velocity		
		int offst = i*9;			
		float uA = fA[offst+1] + fA[offst+5] + fA[offst+8] - (fA[offst+3] + fA[offst+6] + fA[offst+7]) + 0.5*FxA[i];
		float uB = fB[offst+1] + fB[offst+5] + fB[offst+8] - (fB[offst+3] + fB[offst+6] + fB[offst+7]) + 0.5*FxB[i];
		float vA = fA[offst+2] + fA[offst+5] + fA[offst+6] - (fA[offst+4] + fA[offst+7] + fA[offst+8]) + 0.5*FyA[i];
		float vB = fB[offst+2] + fB[offst+5] + fB[offst+6] - (fB[offst+4] + fB[offst+7] + fB[offst+8]) + 0.5*FyB[i];
		float rTotal = rA[i] + rB[i];
		u[i] = (uA + uB)/rTotal;
		v[i] = (vA + vB)/rTotal;
		// modification due to particles
		int pID = pIDgrid[i];
		if (pID > -1) {
			float partvx = pvx[pID];
			float partvy = pvy[pID];
			float partfx = (partvx - u[i])*2.0*rTotal*B[i];
			float partfy = (partvy - v[i])*2.0*rTotal*B[i];
			// ammend fluid velocity
			u[i] += 0.5*partfx/rTotal;
			v[i] += 0.5*partfy/rTotal;
			// ammend fluid forces
			FxA[i] += partfx*(rA[i]/rTotal); 
			FxB[i] += partfx*(rB[i]/rTotal);
			FyA[i] += partfy*(rA[i]/rTotal);
			FyB[i] += partfy*(rB[i]/rTotal);
			// ammend particle forces
			pfx[pID] -= partfx;
			pfy[pID] -= partfy;
		}							
	}
}



// --------------------------------------------------------
// D2Q9 compute Shan-Chen forces for the components
// using pseudo-potential, psi = rho_0(1-exp(-rho/rho_o))
// --------------------------------------------------------

__global__ void mcmp_compute_SC_forces_psm_D2Q9(float* rA,
										        float* rB,
												float* B,
										        float* FxA,
										        float* FxB,
										        float* FyA,
										        float* FyB,
												float* pfx,
												float* pfy,
											    int* nList,
												int* pIDgrid,												
											    float gAB,	
												float gAS,
												float gBS,
												float omega,										    
										        int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
				
		// index for nList[]
		int offst = i*9;
		
		// values of neighbor psi
		float p0A = psi(rA[i]);				
		float p1A = psi(rA[nList[offst+1]]);
		float p2A = psi(rA[nList[offst+2]]);
		float p3A = psi(rA[nList[offst+3]]);
		float p4A = psi(rA[nList[offst+4]]);
		float p5A = psi(rA[nList[offst+5]]);
		float p6A = psi(rA[nList[offst+6]]);
		float p7A = psi(rA[nList[offst+7]]);
		float p8A = psi(rA[nList[offst+8]]);
		
		float p0B = psi(rB[i]);				
		float p1B = psi(rB[nList[offst+1]]);
		float p2B = psi(rB[nList[offst+2]]);
		float p3B = psi(rB[nList[offst+3]]);
		float p4B = psi(rB[nList[offst+4]]);
		float p5B = psi(rB[nList[offst+5]]);
		float p6B = psi(rB[nList[offst+6]]);
		float p7B = psi(rB[nList[offst+7]]);
		float p8B = psi(rB[nList[offst+8]]);
		
		float r0S = B[i];
		float r1S = B[nList[offst+1]];
		float r2S = B[nList[offst+2]];
		float r3S = B[nList[offst+3]];
		float r4S = B[nList[offst+4]];
		float r5S = B[nList[offst+5]];
		float r6S = B[nList[offst+6]];
		float r7S = B[nList[offst+7]];
		float r8S = B[nList[offst+8]];
		
		float p0SA = r0S + omega*r0S*(1.0-r0S);		
		float p1SA = r1S + omega*r1S*(1.0-r1S);
		float p2SA = r2S + omega*r2S*(1.0-r2S);
		float p3SA = r3S + omega*r3S*(1.0-r3S);
		float p4SA = r4S + omega*r4S*(1.0-r4S);
		float p5SA = r5S + omega*r5S*(1.0-r5S);
		float p6SA = r6S + omega*r6S*(1.0-r6S);
		float p7SA = r7S + omega*r7S*(1.0-r7S);
		float p8SA = r8S + omega*r8S*(1.0-r8S);
		
		float p0SB = r0S - omega*r0S*(1.0-r0S);
		float p1SB = r1S - omega*r1S*(1.0-r1S);
		float p2SB = r2S - omega*r2S*(1.0-r2S);
		float p3SB = r3S - omega*r3S*(1.0-r3S);
		float p4SB = r4S - omega*r4S*(1.0-r4S);
		float p5SB = r5S - omega*r5S*(1.0-r5S);
		float p6SB = r6S - omega*r6S*(1.0-r6S);
		float p7SB = r7S - omega*r7S*(1.0-r7S);
		float p8SB = r8S - omega*r8S*(1.0-r8S);
		
		// sum neighbor psi values times wi times ei
		float ws = 1.0/9.0;
		float wd = 1.0/36.0;		
		float sumNbrPsiAx = ws*p1A + wd*p5A + wd*p8A - (ws*p3A + wd*p6A + wd*p7A);
		float sumNbrPsiAy = ws*p2A + wd*p5A + wd*p6A - (ws*p4A + wd*p7A + wd*p8A);
		float sumNbrPsiBx = ws*p1B + wd*p5B + wd*p8B - (ws*p3B + wd*p6B + wd*p7B);
		float sumNbrPsiBy = ws*p2B + wd*p5B + wd*p6B - (ws*p4B + wd*p7B + wd*p8B);
		float sumNbrPsiSAx = ws*p1SA + wd*p5SA + wd*p8SA - (ws*p3SA + wd*p6SA + wd*p7SA);
		float sumNbrPsiSBx = ws*p1SB + wd*p5SB + wd*p8SB - (ws*p3SB + wd*p6SB + wd*p7SB);
		float sumNbrPsiSAy = ws*p2SA + wd*p5SA + wd*p6SA - (ws*p4SA + wd*p7SA + wd*p8SA);
		float sumNbrPsiSBy = ws*p2SB + wd*p5SB + wd*p6SB - (ws*p4SB + wd*p7SB + wd*p8SB);
		
		// fluid forces
		FxA[i] = -p0A*(gAB*sumNbrPsiBx + gAS*sumNbrPsiSAx);
		FxB[i] = -p0B*(gAB*sumNbrPsiAx + gBS*sumNbrPsiSBx);
		FyA[i] = -p0A*(gAB*sumNbrPsiBy + gAS*sumNbrPsiSAy);
		FyB[i] = -p0B*(gAB*sumNbrPsiAy + gBS*sumNbrPsiSBy);
		
		// particle forces
		int pID = pIDgrid[i];
		if (pID > -1) {
			float FxS = -(p0SA*gAS*sumNbrPsiAx + p0SB*gBS*sumNbrPsiBx);
			float FyS = -(p0SA*gAS*sumNbrPsiAy + p0SB*gBS*sumNbrPsiBy);
			atomicAdd(&pfx[pID], FxS);
			atomicAdd(&pfy[pID], FyS);
		}
								
	}
}



// --------------------------------------------------------
// D2Q9 update kernel:
// --------------------------------------------------------

__global__ void mcmp_collide_stream_psm_D2Q9(float* f1A,
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
											 float* pvx,
											 float* pvy,
											 float* B,
											 int* pIDgrid,
										 	 int* streamIndex,
											 float rApart,
											 float rBpart,
										 	 float nu,
										 	 int nVoxels)
{

	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
				
		// --------------------------------------------------	
		// COLLISION & STREAMING - standard BGK operator with
		//                         a PUSH propagator.  This step
		//                         includes the Guo forcing
		//                         scheme applied to the Shan-Chen
		//                         MCMP model according to Kruger et al.
		// --------------------------------------------------
				
		// useful constants
		int offst = 9*i;
		const float w0 = 4.0/9.0;
		const float ws = 1.0/9.0;
		const float wd = 1.0/36.0;		
		const float omega = 2.0/(6.0*nu + 1.0);   // 1/tau
		const float omomega = 1.0 - omega;        // 1 - 1/tau
		const float omomega2 = 1.0 - 0.5*omega;   // 1 - 1/(2tau)
		const float omusq = 1.0 - 1.5*(u[i]*u[i] + v[i]*v[i]);
		const float ux = u[i];
		const float vy = v[i];
		const float Bi = B[i];
		const int   pID = pIDgrid[i];
		float uxp = 0.0; 
		float vyp = 0.0;
		if (pID >= 0) {
			uxp = pvx[pID];
			vyp = pvy[pID];
		}
				
		// direction 0
		float evel = 0.0;       // e dot velocity
		float emiu = 0.0-ux;    // e minus u
		float emiv = 0.0-vy;    // e minus v
		float feq = w0*omusq;
		float frcA = w0*( FxA[i]*(3.0*emiu) + FyA[i]*(3.0*emiv) );
		float frcB = w0*( FxB[i]*(3.0*emiu) + FyB[i]*(3.0*emiv) );		
		f2A[streamIndex[offst+0]] = omomega*f1A[offst+0] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+0]] = omomega*f1B[offst+0] + omega*feq*rB[i] + omomega2*frcB;
				
		// direction 1
		evel = ux;
		emiu = 1.0-ux;
		emiv = 0.0-vy;
		feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = ws*( FxA[i]*(3.0*emiu + 9.0*evel) + FyA[i]*(3.0*emiv) );
		frcB = ws*( FxB[i]*(3.0*emiu + 9.0*evel) + FyB[i]*(3.0*emiv) );
		f2A[streamIndex[offst+1]] = omomega*f1A[offst+1] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+1]] = omomega*f1B[offst+1] + omega*feq*rB[i] + omomega2*frcB;
		
		// direction 2
		evel = vy; 
		emiu = 0.0-ux;
		emiv = 1.0-vy;
		feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = ws*( FxA[i]*(3.0*emiu) + FyA[i]*(3.0*emiv + 9.0*evel) );
		frcB = ws*( FxB[i]*(3.0*emiu) + FyB[i]*(3.0*emiv + 9.0*evel) );
		f2A[streamIndex[offst+2]] = omomega*f1A[offst+2] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+2]] = omomega*f1B[offst+2] + omega*feq*rB[i] + omomega2*frcB;
				
		// direction 3
		evel = -ux;
		emiu = -1.0-ux;
		emiv =  0.0-vy;
		feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = ws*( FxA[i]*(3.0*emiu - 9.0*evel) + FyA[i]*(3.0*emiv) );
		frcB = ws*( FxB[i]*(3.0*emiu - 9.0*evel) + FyB[i]*(3.0*emiv) );		
		f2A[streamIndex[offst+3]] = omomega*f1A[offst+3] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+3]] = omomega*f1B[offst+3] + omega*feq*rB[i] + omomega2*frcB;
		
		// direction 4
		evel = -vy;
		emiu =  0.0-ux;
		emiv = -1.0-vy;
		feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = ws*( FxA[i]*(3.0*emiu) + FyA[i]*(3.0*emiv - 9.0*evel) );
		frcB = ws*( FxB[i]*(3.0*emiu) + FyB[i]*(3.0*emiv - 9.0*evel) );
		f2A[streamIndex[offst+4]] = omomega*f1A[offst+4] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+4]] = omomega*f1B[offst+4] + omega*feq*rB[i] + omomega2*frcB;
		
		// direction 5
		evel = ux + vy;
		emiu = 1.0-ux;
		emiv = 1.0-vy;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*( FxA[i]*(3.0*emiu + 9.0*evel) + FyA[i]*(3.0*emiv + 9.0*evel) );
		frcB = wd*( FxB[i]*(3.0*emiu + 9.0*evel) + FyB[i]*(3.0*emiv + 9.0*evel) );
		f2A[streamIndex[offst+5]] = omomega*f1A[offst+5] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+5]] = omomega*f1B[offst+5] + omega*feq*rB[i] + omomega2*frcB;
		
		// direction 6
		evel = -ux + vy;
		emiu = -1.0-ux;
		emiv =  1.0-vy;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*( FxA[i]*(3.0*emiu - 9.0*evel) + FyA[i]*(3.0*emiv + 9.0*evel) );
		frcB = wd*( FxB[i]*(3.0*emiu - 9.0*evel) + FyB[i]*(3.0*emiv + 9.0*evel) );		
		f2A[streamIndex[offst+6]] = omomega*f1A[offst+6] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+6]] = omomega*f1B[offst+6] + omega*feq*rB[i] + omomega2*frcB;
		
		// direction 7
		evel = -ux - vy;
		emiu = -1.0-ux;
		emiv = -1.0-vy;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*( FxA[i]*(3.0*emiu - 9.0*evel) + FyA[i]*(3.0*emiv - 9.0*evel) );
		frcB = wd*( FxB[i]*(3.0*emiu - 9.0*evel) + FyB[i]*(3.0*emiv - 9.0*evel) );		
		f2A[streamIndex[offst+7]] = omomega*f1A[offst+7] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+7]] = omomega*f1B[offst+7] + omega*feq*rB[i] + omomega2*frcB;
		
		// direction 8
		evel = ux - vy;
		emiu =  1.0-ux;
		emiv = -1.0-vy;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*( FxA[i]*(3.0*emiu + 9.0*evel) + FyA[i]*(3.0*emiv - 9.0*evel) );
		frcB = wd*( FxB[i]*(3.0*emiu + 9.0*evel) + FyB[i]*(3.0*emiv - 9.0*evel) );		
		f2A[streamIndex[offst+8]] = omomega*f1A[offst+8] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+8]] = omomega*f1B[offst+8] + omega*feq*rB[i] + omomega2*frcB;		
		
	}
}





