
# include "mcmp_collide_stream_bb_D2Q9.cuh"
# include "../iolets/zou_he_BC_D2Q9.cuh"
# include <stdio.h>

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
