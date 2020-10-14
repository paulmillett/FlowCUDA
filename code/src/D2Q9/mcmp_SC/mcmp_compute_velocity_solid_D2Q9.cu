
# include "mcmp_compute_velocity_solid_D2Q9.cuh"
# include <stdio.h>

// --------------------------------------------------------
// D2Q9 compute velocity (barycentric) for the system: 
// --------------------------------------------------------

__global__ void mcmp_compute_velocity_solid_D2Q9(float* fA,
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
												 int* pID,
												 particle2D* p,
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
		float rTotal = rA[i] + rB[i] + rS[i];		
		float rSVelx = 0.0;
		float rSVely = 0.0;
		int partID = pID[i]; 
		if (partID >= 0) {
			rSVelx = rS[i]*p[partID].vx;
			rSVely = rS[i]*p[partID].vy;
		}
		u[i] = (uA + uB + rSVelx)/rTotal;
		v[i] = (vA + vB + rSVely)/rTotal;				
	}
}