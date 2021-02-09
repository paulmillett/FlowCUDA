
# include "mcmp_compute_velocity_dip_D2Q9.cuh"
# include <stdio.h>

// --------------------------------------------------------
// D2Q9 compute velocity (barycentric) for the system: 
// --------------------------------------------------------

__global__ void mcmp_compute_velocity_dip_D2Q9(float* fA,
                                               float* fB,
										       float* rA,
										       float* rB,
											   float* rS,
										       float* FxA,
										       float* FxB,
										       float* FyA,
										       float* FyB,
										       float* u,
										       float* v,											   
											   particle2D_dip* pt,											   
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
			float partvx = pt[pID].v.x;
			float partvy = pt[pID].v.y;
			float partfx = (partvx - u[i])*2.0*rTotal*rS[i];
			float partfy = (partvy - v[i])*2.0*rTotal*rS[i];
			// ammend fluid velocity
			u[i] += 0.5*partfx/rTotal;
			v[i] += 0.5*partfy/rTotal;
			// ammend fluid forces
			FxA[i] += partfx*(rA[i]/rTotal); 
			FxB[i] += partfx*(rB[i]/rTotal);
			FyA[i] += partfy*(rA[i]/rTotal);
			FyB[i] += partfy*(rB[i]/rTotal);
			// ammend particle forces  (AtomicAdd!)
			atomicAdd(&pt[pID].f.x, -partfx);
			atomicAdd(&pt[pID].f.y, -partfy);			
		}							
	}
}