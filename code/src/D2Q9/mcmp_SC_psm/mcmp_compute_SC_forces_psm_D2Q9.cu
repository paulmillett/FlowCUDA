
# include "mcmp_compute_SC_forces_psm_D2Q9.cuh"
# include "../mcmp_SC/mcmp_pseudopotential.cuh"


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



