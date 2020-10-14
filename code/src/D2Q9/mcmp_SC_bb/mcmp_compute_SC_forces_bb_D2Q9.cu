
# include "mcmp_compute_SC_forces_bb_D2Q9.cuh"
# include "../mcmp_SC/mcmp_pseudopotential.cuh"


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



