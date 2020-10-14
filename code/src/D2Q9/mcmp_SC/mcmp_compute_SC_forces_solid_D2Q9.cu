
# include "mcmp_compute_SC_forces_solid_D2Q9.cuh"
# include "mcmp_pseudopotential.cuh"



// --------------------------------------------------------
// D2Q9 compute Shan-Chen forces for the components
// using pseudo-potential, psi = rho_0(1-exp(-rho/rho_o))
// --------------------------------------------------------

__global__ void mcmp_compute_SC_forces_solid_1_D2Q9(float* rA,
										            float* rB,
											        float* rS,
										            float* FxA,
										            float* FxB,
										            float* FyA,
										            float* FyB,
											        int* nList,
											        float gAB,
											        float gAS,
											        float gBS,
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
		
		float p1S = psi(rS[nList[offst+1]]);
		float p2S = psi(rS[nList[offst+2]]);
		float p3S = psi(rS[nList[offst+3]]);
		float p4S = psi(rS[nList[offst+4]]);
		float p5S = psi(rS[nList[offst+5]]);
		float p6S = psi(rS[nList[offst+6]]);
		float p7S = psi(rS[nList[offst+7]]);
		float p8S = psi(rS[nList[offst+8]]);
		
		float ws = 1.0/9.0;
		float wd = 1.0/36.0;		
		float sumNbrPsiAx = ws*p1A + wd*p5A + wd*p8A - (ws*p3A + wd*p6A + wd*p7A);
		float sumNbrPsiAy = ws*p2A + wd*p5A + wd*p6A - (ws*p4A + wd*p7A + wd*p8A);
		float sumNbrPsiBx = ws*p1B + wd*p5B + wd*p8B - (ws*p3B + wd*p6B + wd*p7B);
		float sumNbrPsiBy = ws*p2B + wd*p5B + wd*p6B - (ws*p4B + wd*p7B + wd*p8B);
		float sumNbrPsiSx = ws*p1S + wd*p5S + wd*p8S - (ws*p3S + wd*p6S + wd*p7S);
		float sumNbrPsiSy = ws*p2S + wd*p5S + wd*p6S - (ws*p4S + wd*p7S + wd*p8S);
			
		FxA[i] = -p0A*(gAB*sumNbrPsiBx + gAS*sumNbrPsiSx);
		FxB[i] = -p0B*(gAB*sumNbrPsiAx + gBS*sumNbrPsiSx);
		FyA[i] = -p0A*(gAB*sumNbrPsiBy + gAS*sumNbrPsiSy);
		FyB[i] = -p0B*(gAB*sumNbrPsiAy + gBS*sumNbrPsiSy);
		
	}
}



// --------------------------------------------------------
// D2Q9 compute Shan-Chen forces for the components
// using pseudo-potential, psi = rho
// --------------------------------------------------------

__global__ void mcmp_compute_SC_forces_solid_2_D2Q9(float* rA,
										            float* rB,
											        float* rS,
										            float* FxA,
										            float* FxB,
										            float* FyA,
										            float* FyB,
											        int* nList,
											        float gAB,
											        float gAS,
											        float gBS,
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
		
		float p1S = rS[nList[offst+1]];
		float p2S = rS[nList[offst+2]];
		float p3S = rS[nList[offst+3]];
		float p4S = rS[nList[offst+4]];
		float p5S = rS[nList[offst+5]];
		float p6S = rS[nList[offst+6]];
		float p7S = rS[nList[offst+7]];
		float p8S = rS[nList[offst+8]];
		
		float ws = 1.0/9.0;
		float wd = 1.0/36.0;		
		float sumNbrPsiAx = ws*p1A + wd*p5A + wd*p8A - (ws*p3A + wd*p6A + wd*p7A);
		float sumNbrPsiAy = ws*p2A + wd*p5A + wd*p6A - (ws*p4A + wd*p7A + wd*p8A);
		float sumNbrPsiBx = ws*p1B + wd*p5B + wd*p8B - (ws*p3B + wd*p6B + wd*p7B);
		float sumNbrPsiBy = ws*p2B + wd*p5B + wd*p6B - (ws*p4B + wd*p7B + wd*p8B);
		float sumNbrPsiSx = ws*p1S + wd*p5S + wd*p8S - (ws*p3S + wd*p6S + wd*p7S);
		float sumNbrPsiSy = ws*p2S + wd*p5S + wd*p6S - (ws*p4S + wd*p7S + wd*p8S);
			
		FxA[i] = -p0A*(gAB*sumNbrPsiBx + gAS*sumNbrPsiSx);
		FxB[i] = -p0B*(gAB*sumNbrPsiAx + gBS*sumNbrPsiSx);
		FyA[i] = -p0A*(gAB*sumNbrPsiBy + gAS*sumNbrPsiSy);
		FyB[i] = -p0B*(gAB*sumNbrPsiAy + gBS*sumNbrPsiSy);
		
	}
}



// --------------------------------------------------------
// D2Q9 compute Shan-Chen forces for the components
// using pseudo-potential, psi = rho
// --------------------------------------------------------

__global__ void mcmp_compute_SC_forces_solid_3_D2Q9(float* rA,
										            float* rB,
											        float* eta,
										            float* FxA,
										            float* FxB,
										            float* FyA,
										            float* FyB,
											        int* nList,
											        float gAB,
											        float rAinS,
											        float rBinS,
										            int nVoxels)
{
	
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		
		int offst = i*9;
		
		/*	
		float r0A = rA[i];		
		float r1A = rA[nList[offst+1]];
		float r2A = rA[nList[offst+2]];
		float r3A = rA[nList[offst+3]];
		float r4A = rA[nList[offst+4]];
		float r5A = rA[nList[offst+5]];
		float r6A = rA[nList[offst+6]];
		float r7A = rA[nList[offst+7]];
		float r8A = rA[nList[offst+8]];
		
		float r0B = rB[i];		
		float r1B = rB[nList[offst+1]];
		float r2B = rB[nList[offst+2]];
		float r3B = rB[nList[offst+3]];
		float r4B = rB[nList[offst+4]];
		float r5B = rB[nList[offst+5]];
		float r6B = rB[nList[offst+6]];
		float r7B = rB[nList[offst+7]];
		float r8B = rB[nList[offst+8]];
		
		float eta1 = eta[nList[offst+1]];
		float eta2 = eta[nList[offst+2]];
		float eta3 = eta[nList[offst+3]];
		float eta4 = eta[nList[offst+4]];
		float eta5 = eta[nList[offst+5]];
		float eta6 = eta[nList[offst+6]];
		float eta7 = eta[nList[offst+7]];
		float eta8 = eta[nList[offst+8]];
		
		float p1A = (1.0 - eta1)*r1A + (eta1)*rAinS;
		float p2A = (1.0 - eta2)*r2A + (eta2)*rAinS;
		float p3A = (1.0 - eta3)*r3A + (eta3)*rAinS;
		float p4A = (1.0 - eta4)*r4A + (eta4)*rAinS;
		float p5A = (1.0 - eta5)*r5A + (eta5)*rAinS;
		float p6A = (1.0 - eta6)*r6A + (eta6)*rAinS;
		float p7A = (1.0 - eta7)*r7A + (eta7)*rAinS;
		float p8A = (1.0 - eta8)*r8A + (eta8)*rAinS;
		
		float p1B = (1.0 - eta1)*r1B + (eta1)*rBinS;
		float p2B = (1.0 - eta2)*r2B + (eta2)*rBinS;
		float p3B = (1.0 - eta3)*r3B + (eta3)*rBinS;
		float p4B = (1.0 - eta4)*r4B + (eta4)*rBinS;
		float p5B = (1.0 - eta5)*r5B + (eta5)*rBinS;
		float p6B = (1.0 - eta6)*r6B + (eta6)*rBinS;
		float p7B = (1.0 - eta7)*r7B + (eta7)*rBinS;
		float p8B = (1.0 - eta8)*r8B + (eta8)*rBinS;		
		
		float ws = 1.0/9.0;
		float wd = 1.0/36.0;		
		float sumNbrPsiAx = ws*p1A + wd*p5A + wd*p8A - (ws*p3A + wd*p6A + wd*p7A);
		float sumNbrPsiAy = ws*p2A + wd*p5A + wd*p6A - (ws*p4A + wd*p7A + wd*p8A);
		float sumNbrPsiBx = ws*p1B + wd*p5B + wd*p8B - (ws*p3B + wd*p6B + wd*p7B);
		float sumNbrPsiBy = ws*p2B + wd*p5B + wd*p6B - (ws*p4B + wd*p7B + wd*p8B);
					
		FxA[i] = -r0A*(gAB*sumNbrPsiBx);
		FxB[i] = -r0B*(gAB*sumNbrPsiAx);
		FyA[i] = -r0A*(gAB*sumNbrPsiBy);
		FyB[i] = -r0B*(gAB*sumNbrPsiAy);
		
		*/
		
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
		
		//float eta0 = eta[i];
		//float gABinS = gAB*(1.0-eta0) + 7.0*eta0;
							
		FxA[i] = -p0A*(gAB*sumNbrPsiBx); //*(1.0-eta0);
		FxB[i] = -p0B*(gAB*sumNbrPsiAx); //*(1.0-eta0);
		FyA[i] = -p0A*(gAB*sumNbrPsiBy); //*(1.0-eta0);
		FyB[i] = -p0B*(gAB*sumNbrPsiAy); //*(1.0-eta0);
		
	}
}

