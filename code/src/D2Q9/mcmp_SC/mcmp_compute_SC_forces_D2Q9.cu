
# include "mcmp_compute_SC_forces_D2Q9.cuh"
# include "mcmp_pseudopotential.cuh"



// --------------------------------------------------------
// D2Q9 compute Shan-Chen forces for the components
// using pseudo-potential, psi = rho_0(1-exp(-rho/rho_o))
// --------------------------------------------------------

__global__ void mcmp_compute_SC_forces_1_D2Q9(float* rA,
										      float* rB,
										      float* FxA,
										      float* FxB,
										      float* FyA,
										      float* FyB,
											  int* nList,
											  float gAB,
										      int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		
		int offst = i*9;
			
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
		
		float ws = 1.0/9.0;
		float wd = 1.0/36.0;		
		float sumNbrPsiAx = ws*p1A + wd*p5A + wd*p8A - (ws*p3A + wd*p6A + wd*p7A);
		float sumNbrPsiAy = ws*p2A + wd*p5A + wd*p6A - (ws*p4A + wd*p7A + wd*p8A);
		float sumNbrPsiBx = ws*p1B + wd*p5B + wd*p8B - (ws*p3B + wd*p6B + wd*p7B);
		float sumNbrPsiBy = ws*p2B + wd*p5B + wd*p6B - (ws*p4B + wd*p7B + wd*p8B);
			
		FxA[i] = -gAB*p0A*sumNbrPsiBx;
		FxB[i] = -gAB*p0B*sumNbrPsiAx;
		FyA[i] = -gAB*p0A*sumNbrPsiBy;
		FyB[i] = -gAB*p0B*sumNbrPsiAy;
		
	}
}



// --------------------------------------------------------
// D2Q9 compute Shan-Chen forces for the components
// using pseudo-potential, psi = rho
// --------------------------------------------------------

__global__ void mcmp_compute_SC_forces_2_D2Q9(float* rA,
										      float* rB,
										      float* FxA,
										      float* FxB,
										      float* FyA,
										      float* FyB,
											  int* nList,
											  float gAB,
										      int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		
		int offst = i*9;
			
		float p0A = rA[i];				
		float p1A = rA[nList[offst+1]];
		float p2A = rA[nList[offst+2]];
		float p3A = rA[nList[offst+3]];
		float p4A = rA[nList[offst+4]];
		float p5A = rA[nList[offst+5]];
		float p6A = rA[nList[offst+6]];
		float p7A = rA[nList[offst+7]];
		float p8A = rA[nList[offst+8]];
		
		float p0B = rB[i];		
		float p1B = rB[nList[offst+1]];
		float p2B = rB[nList[offst+2]];
		float p3B = rB[nList[offst+3]];
		float p4B = rB[nList[offst+4]];
		float p5B = rB[nList[offst+5]];
		float p6B = rB[nList[offst+6]];
		float p7B = rB[nList[offst+7]];
		float p8B = rB[nList[offst+8]];
		
		float ws = 1.0/9.0;
		float wd = 1.0/36.0;		
		float sumNbrPsiAx = ws*p1A + wd*p5A + wd*p8A - (ws*p3A + wd*p6A + wd*p7A);
		float sumNbrPsiAy = ws*p2A + wd*p5A + wd*p6A - (ws*p4A + wd*p7A + wd*p8A);
		float sumNbrPsiBx = ws*p1B + wd*p5B + wd*p8B - (ws*p3B + wd*p6B + wd*p7B);
		float sumNbrPsiBy = ws*p2B + wd*p5B + wd*p6B - (ws*p4B + wd*p7B + wd*p8B);
			
		FxA[i] = -gAB*p0A*sumNbrPsiBx;
		FxB[i] = -gAB*p0B*sumNbrPsiAx;
		FyA[i] = -gAB*p0A*sumNbrPsiBy;
		FyB[i] = -gAB*p0B*sumNbrPsiAy;
		
	}
}

