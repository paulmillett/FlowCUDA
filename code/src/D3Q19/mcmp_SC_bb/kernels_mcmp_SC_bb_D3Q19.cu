
# include "kernels_mcmp_SC_bb_D3Q19.cuh"
# include "../../D2Q9/mcmp_SC/mcmp_pseudopotential.cuh"
# include <stdio.h>



// --------------------------------------------------------
// Zero particle forces:
// --------------------------------------------------------

__global__ void mcmp_zero_particle_forces_bb_D3Q19(particle3D_bb* pt,
							                       int nParts)
{
	// define particle:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nParts) {		
		pt[i].f = make_float3(0.0);
	}
}



// --------------------------------------------------------
// Update particle velocities and positions:
// --------------------------------------------------------

__global__ void mcmp_move_particles_bb_D3Q19(particle3D_bb* pt,
   								             int nParts)
{
	// define particle:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nParts) {		
		float3 a = pt[i].f/pt[i].mass;
		pt[i].r += pt[i].v + 0.5*a;  // assume dt = 1
		pt[i].v += a;
	}
}



// --------------------------------------------------------
// Fix particle velocity:
// --------------------------------------------------------

__global__ void mcmp_fix_particle_velocity_bb_D3Q19(particle3D_bb* pt,
                                                    float pvel,
   								                    int nParts)
{
	// define particle:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nParts) {	
		pt[i].f = make_float3(0.0);
		if (i == 0) {
			pt[i].v.x = -pvel;
			pt[i].v.y = 0.00;
			pt[i].v.z = 0.00;
		}
		if (i == 1) {
			pt[i].v.x = pvel;
			pt[i].v.y = 0.00;
			pt[i].v.z = 0.00;
		}		
	}
}



// --------------------------------------------------------
// D3Q19 kernel to update the particle fields on the lattice: 
// --------------------------------------------------------

__global__ void mcmp_map_particles_to_lattice_bb_D3Q19(particle3D_bb* pt,
													   int* x,
													   int* y,
													   int* z,
													   int* s,
													   int* sprev,
													   int* pIDgrid,
										               int nVoxels,
													   int nParts)
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nVoxels) {			
		// set previous "s" value:
		sprev[i] = s[i];
		// default values:				
		s[i] = 0;
		pIDgrid[i] = -1;		
		// loop over particles:		
		for (int j=0; j<nParts; j++) {			
			float dx = float(x[i]) - pt[j].r.x;
			float dy = float(y[i]) - pt[j].r.y;
			float dz = float(z[i]) - pt[j].r.z;
			float rr = sqrt(dx*dx + dy*dy + dz*dz);
			if (rr <= pt[j].rad) {
				s[i] = 1;
				pIDgrid[i] = j;	
			}			
		}
	}
}



// --------------------------------------------------------
// D3Q19 initialize populations to equilibrium values:
// --------------------------------------------------------

__global__ void mcmp_initial_equilibrium_bb_D3Q19(float* fA,
                                                  float* fB,
										          float* rA,
											      float* rB,
										          float* u,
										          float* v,
												  float* w,
										          int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {			
		int offst = 19*i;
		equilibrium_populations_bb_D3Q19(fA,fB,rA[i],rB[i],u[i],v[i],w[i],offst);
	}
}



// --------------------------------------------------------
// D3Q19 equilibirium populations: 
// --------------------------------------------------------

__device__ void equilibrium_populations_bb_D3Q19(float* fA,
                                                 float* fB,
										         float rA,
											     float rB,
										         float u,
										         float v,
												 float w,
												 int offst)
{
	const float w0 = 1.0/3.0;
	const float ws = 1.0/18.0;
	const float wd = 1.0/36.0;
	const float omusq = 1.0 - 1.5*(u*u + v*v + w*w);	
	// dir 0
	float feq = w0*omusq;
	fA[offst+0] = feq*rA;
	fB[offst+0] = feq*rB;	
	// dir 1
	float evel = u;
	feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+1] = feq*rA;
	fB[offst+1] = feq*rB;	
	// dir 2
	evel = -u; 
	feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+2] = feq*rA;
	fB[offst+2] = feq*rB;	
	// dir 3
	evel = v;
	feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+3] = feq*rA;
	fB[offst+3] = feq*rB;	
	// dir 4
	evel = -v;
	feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+4] = feq*rA;
	fB[offst+4] = feq*rB;	
	// dir 5
	evel = w;
	feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+5] = feq*rA;
	fB[offst+5] = feq*rB;	
	// dir 6
	evel = -w;
	feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+6] = feq*rA;
	fB[offst+6] = feq*rB;
	// dir 7
	evel = u + v;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+7] = feq*rA;
	fB[offst+7] = feq*rB;	
	// dir 8
	evel = -(u + v);
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+8] = feq*rA;
	fB[offst+8] = feq*rB;	
	// dir 9
	evel = u + w;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+9] = feq*rA;
	fB[offst+9] = feq*rB;	
	// dir 10
	evel = -(u + w);
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+10] = feq*rA;
	fB[offst+10] = feq*rB;	
	// dir 11
	evel = v + w;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+11] = feq*rA;
	fB[offst+11] = feq*rB;	
	// dir 12
	evel = -(v + w);
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+12] = feq*rA;
	fB[offst+12] = feq*rB;	
	// dir 13
	evel = u - v;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+13] = feq*rA;
	fB[offst+13] = feq*rB;	
	// dir 14
	evel = v - u;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+14] = feq*rA;
	fB[offst+14] = feq*rB;	
	// dir 15
	evel = u - w;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+15] = feq*rA;
	fB[offst+15] = feq*rB;	
	// dir 16
	evel = w - u;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+16] = feq*rA;
	fB[offst+16] = feq*rB;	
	// dir 17
	evel = v - w;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+17] = feq*rA;
	fB[offst+17] = feq*rB;	
	// dir 18
	evel = w - v;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+18] = feq*rA;
	fB[offst+18] = feq*rB;
}



// --------------------------------------------------------
// D3Q19 compute velocity (barycentric) for the system.
// Here, the fluid velocity is calculated as normal, but
// it is amended to match the particle velocity.
// --------------------------------------------------------

__global__ void mcmp_compute_velocity_bb_D3Q19(float* fA,
                                               float* fB,
										       float* rA,
										       float* rB,
										       float* FxA,
										       float* FxB,
										       float* FyA,
										       float* FyB,
										       float* FzA,
										       float* FzB,
										       float* u,
										       float* v,
											   float* w,
											   particle3D_bb* pt,	
											   int* s,
											   int* pIDgrid,
										       int nVoxels) 
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		
		// --------------------------------------------------	
		// if this is a fluid site:
		// --------------------------------------------------
		
		if (s[i] == 0) {				
			int offst = i*19;			
			float uA = fA[offst+1] + fA[offst+7] + fA[offst+9] + fA[offst+13] + fA[offst+15] - 
				       (fA[offst+2] + fA[offst+8] + fA[offst+10] + fA[offst+14] + fA[offst+16]) + 0.5*FxA[i];
			float uB = fB[offst+1] + fB[offst+7] + fB[offst+9] + fB[offst+13] + fB[offst+15] - 
				       (fB[offst+2] + fB[offst+8] + fB[offst+10] + fB[offst+14] + fB[offst+16]) + 0.5*FxB[i];		
			float vA = fA[offst+3] + fA[offst+7] + fA[offst+11] + fA[offst+14] + fA[offst+17] - 
				       (fA[offst+4] + fA[offst+8] + fA[offst+12] + fA[offst+13] + fA[offst+18]) + 0.5*FyA[i];
			float vB = fB[offst+3] + fB[offst+7] + fB[offst+11] + fB[offst+14] + fB[offst+17] - 
				       (fB[offst+4] + fB[offst+8] + fB[offst+12] + fB[offst+13] + fB[offst+18]) + 0.5*FyB[i];		
			float wA = fA[offst+5] + fA[offst+9] + fA[offst+11] + fA[offst+16] + fA[offst+18] - 
				       (fA[offst+6] + fA[offst+10] + fA[offst+12] + fA[offst+15] + fA[offst+17]) + 0.5*FzA[i];
			float wB = fB[offst+5] + fB[offst+9] + fB[offst+11] + fB[offst+16] + fB[offst+18] - 
				       (fB[offst+6] + fB[offst+10] + fB[offst+12] + fB[offst+15] + fB[offst+17]) + 0.5*FzB[i];
			float rTotal = rA[i] + rB[i];
			u[i] = (uA + uB)/rTotal;
			v[i] = (vA + vB)/rTotal;
			w[i] = (wA + wB)/rTotal;	
		}
		
		// --------------------------------------------------	
		// if this is a fluid site:
		// --------------------------------------------------
		
		if (s[i] == 1) {
				
		}
			
	}
}



// --------------------------------------------------------
// D3Q19 set velocity on the y=0 and y=Ny-1 boundaries: 
// --------------------------------------------------------

__global__ void mcmp_set_boundary_velocity_bb_D3Q19(float uBC,
                                                    float vBC,
												 	float wBC,
	                                                float* rA,
										            float* rB,
										            float* FxA,
										            float* FxB,
										            float* FyA,
										            float* FyB,
										            float* FzA,
										            float* FzB,
										            float* u,
										            float* v,
													float* w,
													int* y,											        
											        int Ny,
										            int nVoxels) 
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		if (y[i] == 0 || y[i] == Ny-1) {
			float rTotal = rA[i] + rB[i];
			float fxBC = (uBC - u[i])*2.0*rTotal;
			float fyBC = (vBC - v[i])*2.0*rTotal;
			float fzBC = (wBC - w[i])*2.0*rTotal;
			u[i] += 0.5*fxBC/rTotal;
			v[i] += 0.5*fyBC/rTotal;
			w[i] += 0.5*fzBC/rTotal;
			FxA[i] += fxBC*(rA[i]/rTotal);
			FxB[i] += fxBC*(rB[i]/rTotal);
			FyA[i] += fyBC*(rA[i]/rTotal);
			FyB[i] += fyBC*(rB[i]/rTotal);
			FzA[i] += fzBC*(rA[i]/rTotal);
			FzB[i] += fzBC*(rB[i]/rTotal);
		}		
	}
}



// --------------------------------------------------------
// D3Q19 compute density for each component: 
// --------------------------------------------------------

__global__ void mcmp_compute_density_bb_D3Q19(float* fA,
                                        	  float* fB,
										      float* rA,
										      float* rB,
										      int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		int offst = i*19;			
		rA[i] = fA[offst] + fA[offst+1] + fA[offst+2] + fA[offst+3] + fA[offst+4] + fA[offst+5] + fA[offst+6] +
		        fA[offst+7] + fA[offst+8] + fA[offst+9] + fA[offst+10] + fA[offst+11] + fA[offst+12] + fA[offst+13] +
				fA[offst+14] + fA[offst+15] + fA[offst+16] + fA[offst+17] + fA[offst+18];
		rB[i] = fB[offst] + fB[offst+1] + fB[offst+2] + fB[offst+3] + fB[offst+4] + fB[offst+5] + fB[offst+6] +
		        fB[offst+7] + fB[offst+8] + fB[offst+9] + fB[offst+10] + fB[offst+11] + fB[offst+12] + fB[offst+13] +
				fB[offst+14] + fB[offst+15] + fB[offst+16] + fB[offst+17] + fB[offst+18];
	}
}



// --------------------------------------------------------
// D3Q19 compute Shan-Chen forces for the components
// using pseudo-potential, psi = rho_0(1-exp(-rho/rho_o))
// --------------------------------------------------------

__global__ void mcmp_compute_SC_forces_bb_D3Q19(float* rA,
										        float* rB,
												float* rAvirt,
												float* rBvirt,
										        float* FxA,
										        float* FxB,
										        float* FyA,
										        float* FyB,
										        float* FzA,
										        float* FzB,
												particle3D_bb* pt,
											    int* nList,
												int* pIDgrid,	
												int* s,											
											    float gAB,													
										        int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
				
		int offst = i*19;
		
		float p0A = psi(rA[i]);				
		float p1A = psi(rA[nList[offst+1]]);
		float p2A = psi(rA[nList[offst+2]]);
		float p3A = psi(rA[nList[offst+3]]);
		float p4A = psi(rA[nList[offst+4]]);
		float p5A = psi(rA[nList[offst+5]]);
		float p6A = psi(rA[nList[offst+6]]);
		float p7A = psi(rA[nList[offst+7]]);
		float p8A = psi(rA[nList[offst+8]]);
		float p9A = psi(rA[nList[offst+9]]);
		float p10A = psi(rA[nList[offst+10]]);
		float p11A = psi(rA[nList[offst+11]]);
		float p12A = psi(rA[nList[offst+12]]);
		float p13A = psi(rA[nList[offst+13]]);
		float p14A = psi(rA[nList[offst+14]]);
		float p15A = psi(rA[nList[offst+15]]);
		float p16A = psi(rA[nList[offst+16]]);
		float p17A = psi(rA[nList[offst+17]]);
		float p18A = psi(rA[nList[offst+18]]);
		
		float p0B = psi(rB[i]);				
		float p1B = psi(rB[nList[offst+1]]);
		float p2B = psi(rB[nList[offst+2]]);
		float p3B = psi(rB[nList[offst+3]]);
		float p4B = psi(rB[nList[offst+4]]);
		float p5B = psi(rB[nList[offst+5]]);
		float p6B = psi(rB[nList[offst+6]]);
		float p7B = psi(rB[nList[offst+7]]);
		float p8B = psi(rB[nList[offst+8]]);
		float p9B = psi(rB[nList[offst+9]]);
		float p10B = psi(rB[nList[offst+10]]);
		float p11B = psi(rB[nList[offst+11]]);
		float p12B = psi(rB[nList[offst+12]]);
		float p13B = psi(rB[nList[offst+13]]);
		float p14B = psi(rB[nList[offst+14]]);
		float p15B = psi(rB[nList[offst+15]]);
		float p16B = psi(rB[nList[offst+16]]);
		float p17B = psi(rB[nList[offst+17]]);
		float p18B = psi(rB[nList[offst+18]]);
		
		float p0AV = psi(rAvirt[i]);	
		float p1AV = psi(rAvirt[nList[offst+1]]);
		float p2AV = psi(rAvirt[nList[offst+2]]);
		float p3AV = psi(rAvirt[nList[offst+3]]);
		float p4AV = psi(rAvirt[nList[offst+4]]);
		float p5AV = psi(rAvirt[nList[offst+5]]);
		float p6AV = psi(rAvirt[nList[offst+6]]);
		float p7AV = psi(rAvirt[nList[offst+7]]);
		float p8AV = psi(rAvirt[nList[offst+8]]);
		float p9AV = psi(rAvirt[nList[offst+9]]);
		float p10AV = psi(rAvirt[nList[offst+10]]);
		float p11AV = psi(rAvirt[nList[offst+11]]);
		float p12AV = psi(rAvirt[nList[offst+12]]);
		float p13AV = psi(rAvirt[nList[offst+13]]);
		float p14AV = psi(rAvirt[nList[offst+14]]);
		float p15AV = psi(rAvirt[nList[offst+15]]);
		float p16AV = psi(rAvirt[nList[offst+16]]);
		float p17AV = psi(rAvirt[nList[offst+17]]);
		float p18AV = psi(rAvirt[nList[offst+18]]);
	
		float p0BV = psi(rBvirt[i]);		
		float p1BV = psi(rBvirt[nList[offst+1]]);
		float p2BV = psi(rBvirt[nList[offst+2]]);
		float p3BV = psi(rBvirt[nList[offst+3]]);
		float p4BV = psi(rBvirt[nList[offst+4]]);
		float p5BV = psi(rBvirt[nList[offst+5]]);
		float p6BV = psi(rBvirt[nList[offst+6]]);
		float p7BV = psi(rBvirt[nList[offst+7]]);
		float p8BV = psi(rBvirt[nList[offst+8]]);
		float p9BV = psi(rBvirt[nList[offst+9]]);
		float p10BV = psi(rBvirt[nList[offst+10]]);
		float p11BV = psi(rBvirt[nList[offst+11]]);
		float p12BV = psi(rBvirt[nList[offst+12]]);
		float p13BV = psi(rBvirt[nList[offst+13]]);
		float p14BV = psi(rBvirt[nList[offst+14]]);
		float p15BV = psi(rBvirt[nList[offst+15]]);
		float p16BV = psi(rBvirt[nList[offst+16]]);
		float p17BV = psi(rBvirt[nList[offst+17]]);
		float p18BV = psi(rBvirt[nList[offst+18]]);		
				
		// sum neighbor psi values times wi times ei
		float ws = 1.0/18.0;
		float wd = 1.0/36.0;		
		float sumNbrPsiAx = ws*p1A + wd*p7A + wd*p9A  + wd*p13A + wd*p15A - (ws*p2A + wd*p8A  + wd*p10A + wd*p14A + wd*p16A);
		float sumNbrPsiAy = ws*p3A + wd*p7A + wd*p11A + wd*p14A + wd*p17A - (ws*p4A + wd*p8A  + wd*p12A + wd*p13A + wd*p18A);
		float sumNbrPsiAz = ws*p5A + wd*p9A + wd*p11A + wd*p16A + wd*p18A - (ws*p6A + wd*p10A + wd*p12A + wd*p15A + wd*p17A);		
		float sumNbrPsiBx = ws*p1B + wd*p7B + wd*p9B  + wd*p13B + wd*p15B - (ws*p2B + wd*p8B  + wd*p10B + wd*p14B + wd*p16B);
		float sumNbrPsiBy = ws*p3B + wd*p7B + wd*p11B + wd*p14B + wd*p17B - (ws*p4B + wd*p8B  + wd*p12B + wd*p13B + wd*p18B);
		float sumNbrPsiBz = ws*p5B + wd*p9B + wd*p11B + wd*p16B + wd*p18B - (ws*p6B + wd*p10B + wd*p12B + wd*p15B + wd*p17B);
		
		// --------------------------------------------	
		// if this is a fluid site:
		// --------------------------------------------
		
		if (s[i] == 0) {	
			float sumNbrPsiAVx = ws*p1AV + wd*p7AV + wd*p9AV  + wd*p13AV + wd*p15AV - (ws*p2AV + wd*p8AV  + wd*p10AV + wd*p14AV + wd*p16AV);
			float sumNbrPsiAVy = ws*p3AV + wd*p7AV + wd*p11AV + wd*p14AV + wd*p17AV - (ws*p4AV + wd*p8AV  + wd*p12AV + wd*p13AV + wd*p18AV);
			float sumNbrPsiAVz = ws*p5AV + wd*p9AV + wd*p11AV + wd*p16AV + wd*p18AV - (ws*p6AV + wd*p10AV + wd*p12AV + wd*p15AV + wd*p17AV);		
			float sumNbrPsiBVx = ws*p1BV + wd*p7BV + wd*p9BV  + wd*p13BV + wd*p15BV - (ws*p2BV + wd*p8BV  + wd*p10BV + wd*p14BV + wd*p16BV);
			float sumNbrPsiBVy = ws*p3BV + wd*p7BV + wd*p11BV + wd*p14BV + wd*p17BV - (ws*p4BV + wd*p8BV  + wd*p12BV + wd*p13BV + wd*p18BV);
			float sumNbrPsiBVz = ws*p5BV + wd*p9BV + wd*p11BV + wd*p16BV + wd*p18BV - (ws*p6BV + wd*p10BV + wd*p12BV + wd*p15BV + wd*p17BV);
			FxA[i] = -p0A*gAB*(sumNbrPsiBx + sumNbrPsiBVx);
			FxB[i] = -p0B*gAB*(sumNbrPsiAx + sumNbrPsiAVx);
			FyA[i] = -p0A*gAB*(sumNbrPsiBy + sumNbrPsiBVy);
			FyB[i] = -p0B*gAB*(sumNbrPsiAy + sumNbrPsiAVy);
			FzA[i] = -p0A*gAB*(sumNbrPsiBz + sumNbrPsiBVz);
			FzB[i] = -p0B*gAB*(sumNbrPsiAz + sumNbrPsiAVz);			
		}
		
		// --------------------------------------------	
		// if this is a solid site:
		// --------------------------------------------
		
		if (s[i] == 1) {
			float fxAV = -p0AV*gAB*(sumNbrPsiBx);
			float fxBV = -p0BV*gAB*(sumNbrPsiAx);
			float fyAV = -p0AV*gAB*(sumNbrPsiBy);
			float fyBV = -p0BV*gAB*(sumNbrPsiAy);
			float fzAV = -p0AV*gAB*(sumNbrPsiBz);
			float fzBV = -p0BV*gAB*(sumNbrPsiAz);		
			int pID = pIDgrid[i];
			atomicAdd(&pt[pID].f.x, fxAV + fxBV);
			atomicAdd(&pt[pID].f.y, fyAV + fyBV);
			atomicAdd(&pt[pID].f.z, fzAV + fzBV);							
		}
										
	}
}



// --------------------------------------------------------
// D3Q19 update kernel:
// --------------------------------------------------------

__global__ void mcmp_collide_stream_bb_D3Q19(float* f1A,
                                         	 float* f1B,
										 	 float* f2A,
										 	 float* f2B,
										 	 float* rA,
										 	 float* rB,
										 	 float* u,
										 	 float* v,
											 float* w,
										 	 float* FxA,
										 	 float* FxB,
										 	 float* FyA,
										 	 float* FyB,	
										 	 float* FzA,
										 	 float* FzB,										 
										 	 int* streamIndex,											 
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
		int offst = 19*i;
		const float w0 = 1.0/3.0;
		const float ws = 1.0/18.0;
		const float wd = 1.0/36.0;		
		const float omega = 2.0/(6.0*nu + 1.0);   // 1/tau
		const float omomega = 1.0 - omega;        // 1 - 1/tau
		const float omomega2 = 1.0 - 0.5*omega;   // 1 - 1/(2tau)
		const float ux = u[i];
		const float vy = v[i];
		const float wz = w[i];
		const float omusq = 1.0 - 1.5*(ux*ux + vy*vy + wz*wz);
										
		// direction 0
		float evel = 0.0;       // e dot velocity
		float emiu = 0.0-ux;    // e minus u
		float emiv = 0.0-vy;    // e minus v
		float emiw = 0.0-wz;    // e minus w
		float feq = w0*omusq;
		float frcA = w0*( FxA[i]*(3.0*emiu) + FyA[i]*(3.0*emiv)  + FzA[i]*(3.0*emiw) );
		float frcB = w0*( FxB[i]*(3.0*emiu) + FyB[i]*(3.0*emiv)  + FzB[i]*(3.0*emiw) );		
		f2A[streamIndex[offst+0]] = omomega*f1A[offst+0] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+0]] = omomega*f1B[offst+0] + omega*feq*rB[i] + omomega2*frcB;
				
		// direction 1
		evel = ux;
		emiu = 1.0-ux;
		emiv = 0.0-vy;
		emiw = 0.0-wz;
		feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = ws*( FxA[i]*(3.0*emiu + 9.0*evel) + FyA[i]*(3.0*emiv) + FzA[i]*(3.0*emiw) );
		frcB = ws*( FxB[i]*(3.0*emiu + 9.0*evel) + FyB[i]*(3.0*emiv) + FzB[i]*(3.0*emiw) );
		f2A[streamIndex[offst+1]] = omomega*f1A[offst+1] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+1]] = omomega*f1B[offst+1] + omega*feq*rB[i] + omomega2*frcB;
					
		// direction 2
		evel = -ux;
		emiu = -1.0-ux;
		emiv = 0.0-vy;
		emiw = 0.0-wz;
		feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = ws*(FxA[i]*(3.0*emiu - 9.0*evel) + FyA[i]*(3.0*emiv) + FzA[i]*(3.0*emiw));
		frcB = ws*(FxB[i]*(3.0*emiu - 9.0*evel) + FyB[i]*(3.0*emiv) + FzB[i]*(3.0*emiw));
		f2A[streamIndex[offst+2]] = omomega*f1A[offst+2] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+2]] = omomega*f1B[offst+2] + omega*feq*rB[i] + omomega2*frcB;
			
		// direction 3
		evel = vy;
		emiu = 0.0-ux;
		emiv = 1.0-vy;
		emiw = 0.0-wz;
		feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = ws*(FxA[i]*(3.0*emiu) + FyA[i]*(3.0*emiv + 9.0*evel) + FzA[i]*(3.0*emiw));	
		frcB = ws*(FxB[i]*(3.0*emiu) + FyB[i]*(3.0*emiv + 9.0*evel) + FzB[i]*(3.0*emiw));	
		f2A[streamIndex[offst+3]] = omomega*f1A[offst+3] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+3]] = omomega*f1B[offst+3] + omega*feq*rB[i] + omomega2*frcB;
		
		// direction 4
		evel = -vy;
		emiu = 0.0-ux;
		emiv = -1.0-vy;
		emiw = 0.0-wz;
		feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = ws*(FxA[i]*(3.0*emiu) + FyA[i]*(3.0*emiv - 9.0*evel) + FzA[i]*(3.0*emiw));
		frcB = ws*(FxB[i]*(3.0*emiu) + FyB[i]*(3.0*emiv - 9.0*evel) + FzB[i]*(3.0*emiw));
		f2A[streamIndex[offst+4]] = omomega*f1A[offst+4] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+4]] = omomega*f1B[offst+4] + omega*feq*rB[i] + omomega2*frcB;
		
		// direction 5
		evel = wz;
		emiu = 0.0-ux;
		emiv = 0.0-vy;
		emiw = 1.0-wz;
		feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = ws*(FxA[i]*(3.0*emiu) + FyA[i]*(3.0*emiv) + FzA[i]*(3.0*emiw + 9.0*evel));
		frcB = ws*(FxB[i]*(3.0*emiu) + FyB[i]*(3.0*emiv) + FzB[i]*(3.0*emiw + 9.0*evel));
		f2A[streamIndex[offst+5]] = omomega*f1A[offst+5] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+5]] = omomega*f1B[offst+5] + omega*feq*rB[i] + omomega2*frcB;
		
		// direction 6
		evel = -wz;
		emiu = 0.0-ux;
		emiv = 0.0-vy;
		emiw = -1.0-wz;
		feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = ws*(FxA[i]*(3.0*emiu) + FyA[i]*(3.0*emiv) + FzA[i]*(3.0*emiw - 9.0*evel));
		frcB = ws*(FxB[i]*(3.0*emiu) + FyB[i]*(3.0*emiv) + FzB[i]*(3.0*emiw - 9.0*evel));
		f2A[streamIndex[offst+6]] = omomega*f1A[offst+6] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+6]] = omomega*f1B[offst+6] + omega*feq*rB[i] + omomega2*frcB;
		
		// direction 7
		evel = ux+vy;
		emiu = 1.0-ux;
		emiv = 1.0-vy;
		emiw = 0.0-wz;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*(FxA[i]*(3.0*emiu + 9.0*evel) + FyA[i]*(3.0*emiv + 9.0*evel) + FzA[i]*(3.0*emiw));
		frcB = wd*(FxB[i]*(3.0*emiu + 9.0*evel) + FyB[i]*(3.0*emiv + 9.0*evel) + FzB[i]*(3.0*emiw));
		f2A[streamIndex[offst+7]] = omomega*f1A[offst+7] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+7]] = omomega*f1B[offst+7] + omega*feq*rB[i] + omomega2*frcB;
		
		// direction 8
		evel = -ux-vy;
		emiu = -1.0-ux;
		emiv = -1.0-vy;
		emiw = 0.0-wz;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*(FxA[i]*(3.0*emiu - 9.0*evel) + FyA[i]*(3.0*emiv - 9.0*evel) + FzA[i]*(3.0*emiw));
		frcB = wd*(FxB[i]*(3.0*emiu - 9.0*evel) + FyB[i]*(3.0*emiv - 9.0*evel) + FzB[i]*(3.0*emiw));
		f2A[streamIndex[offst+8]] = omomega*f1A[offst+8] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+8]] = omomega*f1B[offst+8] + omega*feq*rB[i] + omomega2*frcB;	
		
		// direction 9
		evel = ux+wz;
		emiu = 1.0-ux;
		emiv = 0.0-vy;
		emiw = 1.0-wz;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*(FxA[i]*(3.0*emiu + 9.0*evel) + FyA[i]*(3.0*emiv) + FzA[i]*(3.0*emiw + 9.0*evel));
		frcB = wd*(FxB[i]*(3.0*emiu + 9.0*evel) + FyB[i]*(3.0*emiv) + FzB[i]*(3.0*emiw + 9.0*evel));
		f2A[streamIndex[offst+9]] = omomega*f1A[offst+9] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+9]] = omomega*f1B[offst+9] + omega*feq*rB[i] + omomega2*frcB;	
		
		// direction 10
		evel = -ux-wz;
		emiu = -1.0-ux;
		emiv = 0.0-vy;
		emiw = -1.0-wz;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*(FxA[i]*(3.0*emiu - 9.0*evel) + FyA[i]*(3.0*emiv) + FzA[i]*(3.0*emiw - 9.0*evel));
		frcB = wd*(FxB[i]*(3.0*emiu - 9.0*evel) + FyB[i]*(3.0*emiv) + FzB[i]*(3.0*emiw - 9.0*evel));
		f2A[streamIndex[offst+10]] = omomega*f1A[offst+10] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+10]] = omomega*f1B[offst+10] + omega*feq*rB[i] + omomega2*frcB;
		
		// direction 11
		evel = vy+wz;
		emiu = 0.0-ux;
		emiv = 1.0-vy;
		emiw = 1.0-wz;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*(FxA[i]*(3.0*emiu) + FyA[i]*(3.0*emiv + 9.0*evel) + FzA[i]*(3.0*emiw + 9.0*evel));
		frcB = wd*(FxB[i]*(3.0*emiu) + FyB[i]*(3.0*emiv + 9.0*evel) + FzB[i]*(3.0*emiw + 9.0*evel));
		f2A[streamIndex[offst+11]] = omomega*f1A[offst+11] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+11]] = omomega*f1B[offst+11] + omega*feq*rB[i] + omomega2*frcB;	
		
		// direction 12
		evel = -vy-wz;
		emiu = 0.0-ux;
		emiv = -1.0-vy;
		emiw = -1.0-wz;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*(FxA[i]*(3.0*emiu) + FyA[i]*(3.0*emiv - 9.0*evel) + FzA[i]*(3.0*emiw - 9.0*evel));
		frcB = wd*(FxB[i]*(3.0*emiu) + FyB[i]*(3.0*emiv - 9.0*evel) + FzB[i]*(3.0*emiw - 9.0*evel));
		f2A[streamIndex[offst+12]] = omomega*f1A[offst+12] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+12]] = omomega*f1B[offst+12] + omega*feq*rB[i] + omomega2*frcB;	
		
		// direction 13
		evel = ux-vy;
		emiu = 1.0-ux;
		emiv = -1.0-vy;
		emiw = 0.0-wz;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*(FxA[i]*(3.0*emiu + 9.0*evel) + FyA[i]*(3.0*emiv - 9.0*evel) + FzA[i]*(3.0*emiw));
		frcB = wd*(FxB[i]*(3.0*emiu + 9.0*evel) + FyB[i]*(3.0*emiv - 9.0*evel) + FzB[i]*(3.0*emiw));
		f2A[streamIndex[offst+13]] = omomega*f1A[offst+13] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+13]] = omomega*f1B[offst+13] + omega*feq*rB[i] + omomega2*frcB;	
		
		// direction 14
		evel = -ux+vy;
		emiu = -1.0-ux;
		emiv = 1.0-vy;
		emiw = 0.0-wz;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*(FxA[i]*(3.0*emiu - 9.0*evel) + FyA[i]*(3.0*emiv + 9.0*evel) + FzA[i]*(3.0*emiw));
		frcB = wd*(FxB[i]*(3.0*emiu - 9.0*evel) + FyB[i]*(3.0*emiv + 9.0*evel) + FzB[i]*(3.0*emiw));
		f2A[streamIndex[offst+14]] = omomega*f1A[offst+14] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+14]] = omomega*f1B[offst+14] + omega*feq*rB[i] + omomega2*frcB;	
		
		// direction 15
		evel = ux-wz;
		emiu = 1.0-ux;
		emiv = 0.0-vy;
		emiw = -1.0-wz;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*(FxA[i]*(3.0*emiu + 9.0*evel) + FyA[i]*(3.0*emiv) + FzA[i]*(3.0*emiw - 9.0*evel));
		frcB = wd*(FxB[i]*(3.0*emiu + 9.0*evel) + FyB[i]*(3.0*emiv) + FzB[i]*(3.0*emiw - 9.0*evel));
		f2A[streamIndex[offst+15]] = omomega*f1A[offst+15] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+15]] = omomega*f1B[offst+15] + omega*feq*rB[i] + omomega2*frcB;	
		
		// direction 16
		evel = -ux+wz;
		emiu = -1.0-ux;
		emiv = 0.0-vy;
		emiw = 1.0-wz;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*(FxA[i]*(3.0*emiu - 9.0*evel) + FyA[i]*(3.0*emiv) + FzA[i]*(3.0*emiw + 9.0*evel));
		frcB = wd*(FxB[i]*(3.0*emiu - 9.0*evel) + FyB[i]*(3.0*emiv) + FzB[i]*(3.0*emiw + 9.0*evel));
		f2A[streamIndex[offst+16]] = omomega*f1A[offst+16] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+16]] = omomega*f1B[offst+16] + omega*feq*rB[i] + omomega2*frcB;	
		
		// direction 17
		evel = vy-wz;
		emiu = 0.0-ux;
		emiv = 1.0-vy;
		emiw = -1.0-wz;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*(FxA[i]*(3.0*emiu) + FyA[i]*(3.0*emiv + 9.0*evel) + FzA[i]*(3.0*emiw - 9.0*evel));
		frcB = wd*(FxB[i]*(3.0*emiu) + FyB[i]*(3.0*emiv + 9.0*evel) + FzB[i]*(3.0*emiw - 9.0*evel));
		f2A[streamIndex[offst+17]] = omomega*f1A[offst+17] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+17]] = omomega*f1B[offst+17] + omega*feq*rB[i] + omomega2*frcB;	
		
		// direction 18
		evel = -vy+wz;
		emiu = 0.0-ux;
		emiv = -1.0-vy;
		emiw = 1.0-wz;
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		frcA = wd*(FxA[i]*(3.0*emiu) + FyA[i]*(3.0*emiv - 9.0*evel) + FzA[i]*(3.0*emiw + 9.0*evel));
		frcB = wd*(FxB[i]*(3.0*emiu) + FyB[i]*(3.0*emiv - 9.0*evel) + FzB[i]*(3.0*emiw + 9.0*evel));
		f2A[streamIndex[offst+18]] = omomega*f1A[offst+18] + omega*feq*rA[i] + omomega2*frcA;
		f2B[streamIndex[offst+18]] = omomega*f1B[offst+18] + omega*feq*rB[i] + omomega2*frcB;		
		
	}
}



// --------------------------------------------------------
// D3Q19 implement bounce-back conditions:
// --------------------------------------------------------

__global__ void mcmp_bounce_back_D3Q19(float* f2A, 
									   float* f2B,
									   int* s,
									   int* nList,									  
									   int* streamIndex,
									   int nVoxels)
{

	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
								
		// --------------------------------------------------	
		// If the current voxel is solid, then bounce-back
		// the populations just received via streaming 
		// back to the neighboring voxel:
		// --------------------------------------------------
		
		if (s[i] == 1) {
			
			int offst = 19*i;
						
			// dir 1 bounce-back to nabor 2 as dir 2:
			if (s[nList[offst+2]] == 0) {
				f2A[streamIndex[offst+2]] = f2A[offst+1];
				f2B[streamIndex[offst+2]] = f2B[offst+1];
				f2A[offst+1] = 0.0;
				f2B[offst+1] = 0.0;
			}
			
			// dir 2 bounce-back to nabor 1 as dir 1:
			if (s[nList[offst+1]] == 0) {
				f2A[streamIndex[offst+1]] = f2A[offst+2];
				f2B[streamIndex[offst+1]] = f2B[offst+2];
				f2A[offst+2] = 0.0;
				f2B[offst+2] = 0.0;
			}
			
			// dir 3 bounce-back to nabor 4 as dir 4:
			if (s[nList[offst+4]] == 0) {
				f2A[streamIndex[offst+4]] = f2A[offst+3];
				f2B[streamIndex[offst+4]] = f2B[offst+3];
				f2A[offst+3] = 0.0;
				f2B[offst+3] = 0.0;
			}
			
			// dir 4 bounce-back to nabor 3 as dir 3:
			if (s[nList[offst+3]] == 0) {
				f2A[streamIndex[offst+3]] = f2A[offst+4];
				f2B[streamIndex[offst+3]] = f2B[offst+4];
				f2A[offst+4] = 0.0;
				f2B[offst+4] = 0.0;
			}
			
			// dir 5 bounce-back to nabor 6 as dir 6:
			if (s[nList[offst+6]] == 0) {
				f2A[streamIndex[offst+6]] = f2A[offst+5];
				f2B[streamIndex[offst+6]] = f2B[offst+5];
				f2A[offst+5] = 0.0;
				f2B[offst+5] = 0.0;
			}
			
			// dir 6 bounce-back to nabor 5 as dir 5:
			if (s[nList[offst+5]] == 0) {
				f2A[streamIndex[offst+5]] = f2A[offst+6];
				f2B[streamIndex[offst+5]] = f2B[offst+6];
				f2A[offst+6] = 0.0;
				f2B[offst+6] = 0.0;
			}
			
			// dir 7 bounce-back to nabor 8 as dir 8:
			if (s[nList[offst+8]] == 0) {
				f2A[streamIndex[offst+8]] = f2A[offst+7];
				f2B[streamIndex[offst+8]] = f2B[offst+7];
				f2A[offst+7] = 0.0;
				f2B[offst+7] = 0.0;
			}
			
			// dir 8 bounce-back to nabor 7 as dir 7:
			if (s[nList[offst+7]] == 0) {
				f2A[streamIndex[offst+7]] = f2A[offst+8];
				f2B[streamIndex[offst+7]] = f2B[offst+8];
				f2A[offst+8] = 0.0;
				f2B[offst+8] = 0.0;
			}
			
			// dir 9 bounce-back to nabor 10 as dir 10:
			if (s[nList[offst+10]] == 0) {
				f2A[streamIndex[offst+10]] = f2A[offst+9];
				f2B[streamIndex[offst+10]] = f2B[offst+9];
				f2A[offst+9] = 0.0;
				f2B[offst+9] = 0.0;
			}
			
			// dir 10 bounce-back to nabor 9 as dir 9:
			if (s[nList[offst+9]] == 0) {
				f2A[streamIndex[offst+9]] = f2A[offst+10];
				f2B[streamIndex[offst+9]] = f2B[offst+10];
				f2A[offst+10] = 0.0;
				f2B[offst+10] = 0.0;
			}
			
			// dir 11 bounce-back to nabor 12 as dir 12:
			if (s[nList[offst+12]] == 0) {
				f2A[streamIndex[offst+12]] = f2A[offst+11];
				f2B[streamIndex[offst+12]] = f2B[offst+11];
				f2A[offst+11] = 0.0;
				f2B[offst+11] = 0.0;
			}
			
			// dir 12 bounce-back to nabor 11 as dir 11:
			if (s[nList[offst+11]] == 0) {
				f2A[streamIndex[offst+11]] = f2A[offst+12];
				f2B[streamIndex[offst+11]] = f2B[offst+12];
				f2A[offst+12] = 0.0;
				f2B[offst+12] = 0.0;
			}
			
			// dir 13 bounce-back to nabor 14 as dir 14:
			if (s[nList[offst+14]] == 0) {
				f2A[streamIndex[offst+14]] = f2A[offst+13];
				f2B[streamIndex[offst+14]] = f2B[offst+13];
				f2A[offst+13] = 0.0;
				f2B[offst+13] = 0.0;
			}
			
			// dir 14 bounce-back to nabor 13 as dir 13:
			if (s[nList[offst+13]] == 0) {
				f2A[streamIndex[offst+13]] = f2A[offst+14];
				f2B[streamIndex[offst+13]] = f2B[offst+14];
				f2A[offst+14] = 0.0;
				f2B[offst+14] = 0.0;
			}
			
			// dir 15 bounce-back to nabor 16 as dir 16:
			if (s[nList[offst+16]] == 0) {
				f2A[streamIndex[offst+16]] = f2A[offst+15];
				f2B[streamIndex[offst+16]] = f2B[offst+15];
				f2A[offst+15] = 0.0;
				f2B[offst+15] = 0.0;
			}
			
			// dir 16 bounce-back to nabor 15 as dir 15:
			if (s[nList[offst+15]] == 0) {
				f2A[streamIndex[offst+15]] = f2A[offst+16];
				f2B[streamIndex[offst+15]] = f2B[offst+16];
				f2A[offst+16] = 0.0;
				f2B[offst+16] = 0.0;
			}
			
			// dir 17 bounce-back to nabor 18 as dir 18:
			if (s[nList[offst+18]] == 0) {
				f2A[streamIndex[offst+18]] = f2A[offst+17];
				f2B[streamIndex[offst+18]] = f2B[offst+17];
				f2A[offst+17] = 0.0;
				f2B[offst+17] = 0.0;
			}
			
			// dir 18 bounce-back to nabor 17 as dir 17:
			if (s[nList[offst+17]] == 0) {
				f2A[streamIndex[offst+17]] = f2A[offst+18];
				f2B[streamIndex[offst+17]] = f2B[offst+18];
				f2A[offst+18] = 0.0;
				f2B[offst+18] = 0.0;
			}
			
		}	
	}		
}



// --------------------------------------------------------
// D3Q19 implement bounce-back conditions for moving
// solids:
// --------------------------------------------------------

__global__ void mcmp_bounce_back_moving_D3Q19(float* f2A, 
									          float* f2B,
											  float* rA,
											  float* rB,
											  float* u,
											  float* v,
											  float* w,
											  particle3D_bb* pt, 
											  int* pIDgrid,
									          int* s,
									          int* nList,									  
									          int* streamIndex,
									          int nVoxels)
{

	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
								
		// --------------------------------------------------	
		// If the current voxel is solid, then bounce-back
		// the populations just received via streaming 
		// back to the neighboring voxel:
		// --------------------------------------------------
		
		if (s[i] == 1) {
						
			int offst = 19*i;
			int pID = pIDgrid[i];
			const float ws = 1.0/18.0;
			const float wd = 1.0/36.0;	
			float meF2S  = 0.0;  // momentum exchange fluid to solid
			float meF2Sx = 0.0;  // momentum exchange fluid to solid (x)
			float meF2Sy = 0.0;  // momentum exchange fluid to solid (y)
			float meF2Sz = 0.0;  // momentum exchange fluid to solid (z)
						
			// dir 1 bounce-back to nabor 3 as dir 3:
			if (s[nList[offst+3]] == 0) {
				// bounce-back
				float evel = u[i];
				f2A[streamIndex[offst+3]] = f2A[offst+1] - 6.0*ws*rA[nList[offst+3]]*evel;
				f2B[streamIndex[offst+3]] = f2B[offst+1] - 6.0*ws*rB[nList[offst+3]]*evel;
				// momentum exchange to particle
				meF2S = 2.0*f2A[offst+1] - 6.0*ws*rA[nList[offst+3]]*evel + 
					    2.0*f2B[offst+1] - 6.0*ws*rB[nList[offst+3]]*evel;
				meF2Sx = meF2S;
				meF2Sy = 0.0;
				atomicAdd(&pt[pID].f.x, meF2Sx);
				atomicAdd(&pt[pID].f.y, meF2Sy);
				// zero populations inside particle				
				f2A[offst+1] = 0.0;
				f2B[offst+1] = 0.0;				
			}
			
			// dir 2 bounce-back to nabor 4 as dir 4:
			if (s[nList[offst+4]] == 0) {
				// bounce-back
				float evel = v[i];
				f2A[streamIndex[offst+4]] = f2A[offst+2] - 6.0*ws*rA[nList[offst+4]]*evel;
				f2B[streamIndex[offst+4]] = f2B[offst+2] - 6.0*ws*rB[nList[offst+4]]*evel;
				// momentum exchange to particle
				meF2S = 2.0*f2A[offst+2] - 6.0*ws*rA[nList[offst+4]]*evel + 
					    2.0*f2B[offst+2] - 6.0*ws*rB[nList[offst+4]]*evel;
				meF2Sx = 0.0;
				meF2Sy = meF2S;
				atomicAdd(&pt[pID].f.x, meF2Sx);
				atomicAdd(&pt[pID].f.y, meF2Sy);
				// zero populations inside particle	
				f2A[offst+2] = 0.0;
				f2B[offst+2] = 0.0;
			}
			
			// dir 3 bounce-back to nabor 1 as dir 1:
			if (s[nList[offst+1]] == 0) {
				// bounce-back
				float evel = -u[i];
				f2A[streamIndex[offst+1]] = f2A[offst+3] - 6.0*ws*rA[nList[offst+1]]*evel;
				f2B[streamIndex[offst+1]] = f2B[offst+3] - 6.0*ws*rB[nList[offst+1]]*evel;
				// momentum exchange to particle
				meF2S = 2.0*f2A[offst+3] - 6.0*ws*rA[nList[offst+1]]*evel + 
					    2.0*f2B[offst+3] - 6.0*ws*rB[nList[offst+1]]*evel;
				meF2Sx = -meF2S;
				meF2Sy = 0.0;
				atomicAdd(&pt[pID].f.x, meF2Sx);
				atomicAdd(&pt[pID].f.y, meF2Sy);	
				// zero populations inside particle
				f2A[offst+3] = 0.0;
				f2B[offst+3] = 0.0;
			}
			
			// dir 4 bounce-back to nabor 2 as dir 2:
			if (s[nList[offst+2]] == 0) {
				// bounce-back
				float evel = -v[i];
				f2A[streamIndex[offst+2]] = f2A[offst+4] - 6.0*ws*rA[nList[offst+2]]*evel;
				f2B[streamIndex[offst+2]] = f2B[offst+4] - 6.0*ws*rB[nList[offst+2]]*evel;
				// momentum exchange to particle
				meF2S = 2.0*f2A[offst+4] - 6.0*ws*rA[nList[offst+2]]*evel + 
					    2.0*f2B[offst+4] - 6.0*ws*rB[nList[offst+2]]*evel;
				meF2Sx = 0.0;  
				meF2Sy = -meF2S;
				atomicAdd(&pt[pID].f.x, meF2Sx);
				atomicAdd(&pt[pID].f.y, meF2Sy);	
				// zero populations inside particle
				f2A[offst+4] = 0.0;
				f2B[offst+4] = 0.0;
			}
			
			// dir 5 bounce-back to nabor 7 as dir 7:
			if (s[nList[offst+7]] == 0) {
				// bounce-back
				float evel = u[i] + v[i];
				f2A[streamIndex[offst+7]] = f2A[offst+5] - 6.0*wd*rA[nList[offst+7]]*evel;
				f2B[streamIndex[offst+7]] = f2B[offst+5] - 6.0*wd*rB[nList[offst+7]]*evel;
				// momentum exchange to particle
				meF2S = 2.0*f2A[offst+5] - 6.0*wd*rA[nList[offst+7]]*evel + 
					    2.0*f2B[offst+5] - 6.0*wd*rB[nList[offst+7]]*evel;  
				meF2Sx = meF2S;  
				meF2Sy = meF2S;
				atomicAdd(&pt[pID].f.x, meF2Sx);
				atomicAdd(&pt[pID].f.y, meF2Sy);	
				// zero populations inside particle
				f2A[offst+5] = 0.0;
				f2B[offst+5] = 0.0;
			}
			
			// dir 6 bounce-back to nabor 8 as dir 8:
			if (s[nList[offst+8]] == 0) {
				// bounce-back
				float evel = -u[i] + v[i];
				f2A[streamIndex[offst+8]] = f2A[offst+6] - 6.0*wd*rA[nList[offst+8]]*evel;
				f2B[streamIndex[offst+8]] = f2B[offst+6] - 6.0*wd*rB[nList[offst+8]]*evel;
				// momentum exchange to particle
				meF2S = 2.0*f2A[offst+6] - 6.0*wd*rA[nList[offst+8]]*evel + 
					    2.0*f2B[offst+6] - 6.0*wd*rB[nList[offst+8]]*evel;  
				meF2Sx = -meF2S;  
				meF2Sy = meF2S;
				atomicAdd(&pt[pID].f.x, meF2Sx);
				atomicAdd(&pt[pID].f.y, meF2Sy);	
				// zero populations inside particle
				f2A[offst+6] = 0.0;
				f2B[offst+6] = 0.0;
			}
			
			// dir 7 bounce-back to nabor 5 as dir 5:
			if (s[nList[offst+5]] == 0) {
				// bounce-back
				float evel = -u[i] - v[i];
				f2A[streamIndex[offst+5]] = f2A[offst+7] - 6.0*wd*rA[nList[offst+5]]*evel;
				f2B[streamIndex[offst+5]] = f2B[offst+7] - 6.0*wd*rB[nList[offst+5]]*evel;
				// momentum exchange to particle
				meF2S = 2.0*f2A[offst+7] - 6.0*wd*rA[nList[offst+5]]*evel + 
					    2.0*f2B[offst+7] - 6.0*wd*rB[nList[offst+5]]*evel;  
				meF2Sx = -meF2S;  
				meF2Sy = -meF2S;
				atomicAdd(&pt[pID].f.x, meF2Sx);
				atomicAdd(&pt[pID].f.y, meF2Sy);
				// zero populations inside particle	
				f2A[offst+7] = 0.0;
				f2B[offst+7] = 0.0;
			}
			
			// dir 8 bounce-back to nabor 6 as dir 6:
			if (s[nList[offst+6]] == 0) {
				// bounce-back
				float evel = u[i] - v[i];
				f2A[streamIndex[offst+6]] = f2A[offst+8] - 6.0*wd*rA[nList[offst+6]]*evel;
				f2B[streamIndex[offst+6]] = f2B[offst+8] - 6.0*wd*rB[nList[offst+6]]*evel;
				// momentum exchange to particle
				meF2S = 2.0*f2A[offst+8] - 6.0*wd*rA[nList[offst+6]]*evel + 
					    2.0*f2B[offst+8] - 6.0*wd*rB[nList[offst+6]]*evel;  
				meF2Sx = meF2S;  
				meF2Sy = -meF2S;
				atomicAdd(&pt[pID].f.x, meF2Sx);
				atomicAdd(&pt[pID].f.y, meF2Sy);
				// zero populations inside particle	
				f2A[offst+8] = 0.0;
				f2B[offst+8] = 0.0;
			}
			
		}	
	}		
}