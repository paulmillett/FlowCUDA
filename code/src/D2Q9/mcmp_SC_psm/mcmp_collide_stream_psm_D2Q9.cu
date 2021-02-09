
# include "mcmp_collide_stream_psm_D2Q9.cuh"
# include "../iolets/zou_he_BC_D2Q9.cuh"
# include <stdio.h>

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



