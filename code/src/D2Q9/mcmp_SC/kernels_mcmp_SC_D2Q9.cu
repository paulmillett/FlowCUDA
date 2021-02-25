
# include "kernels_mcmp_SC_D2Q9.cuh"
# include "mcmp_pseudopotential.cuh"
# include <stdio.h>



// --------------------------------------------------------
// D2Q9 initialize kernel: 
// --------------------------------------------------------

__global__ void mcmp_initial_equilibrium_D2Q9(float* fA,
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
// D2Q9 compute density for each component: 
// --------------------------------------------------------

__global__ void mcmp_compute_density_D2Q9(float* fA,
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

__global__ void mcmp_compute_velocity_D2Q9(float* fA,
                                           float* fB,
										   float* rA,
										   float* rB,
										   float* FxA,
										   float* FxB,
										   float* FyA,
										   float* FyB,
										   float* u,
										   float* v,
										   int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		int offst = i*9;			
		float uA = fA[offst+1] + fA[offst+5] + fA[offst+8] - (fA[offst+3] + fA[offst+6] + fA[offst+7]) + 0.5*FxA[i];
		float uB = fB[offst+1] + fB[offst+5] + fB[offst+8] - (fB[offst+3] + fB[offst+6] + fB[offst+7]) + 0.5*FxB[i];
		float vA = fA[offst+2] + fA[offst+5] + fA[offst+6] - (fA[offst+4] + fA[offst+7] + fA[offst+8]) + 0.5*FyA[i];
		float vB = fB[offst+2] + fB[offst+5] + fB[offst+6] - (fB[offst+4] + fB[offst+7] + fB[offst+8]) + 0.5*FyB[i];
		float rTotal = rA[i] + rB[i];
		u[i] = (uA + uB)/rTotal;
		v[i] = (vA + vB)/rTotal;				
	}
}



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



// --------------------------------------------------------
// D2Q9 compute Shan-Chen pressure for the components: 
// --------------------------------------------------------

__global__ void mcmp_compute_SC_pressure_D2Q9(float* rA,
										      float* rB,
											  float* pr,											
											  float gAB,
										      int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {
		pr[i] = (rA[i]+rB[i])/3.0 + gAB*psi(rA[i])*psi(rB[i])/6.0;		
	}
}



// --------------------------------------------------------
// D2Q9 update kernel:
// --------------------------------------------------------

__global__ void mcmp_collide_stream_D2Q9(float* f1A,
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
										 int* streamIndex,
										 float nu,
										 int nVoxels)
{

	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
				
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

