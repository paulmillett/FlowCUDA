
# include "kernels_mcmp_SC_dip_D3Q19.cuh"
# include "../../D2Q9/mcmp_SC/mcmp_pseudopotential.cuh"
# include <stdio.h>



// --------------------------------------------------------
// Zero particle forces:
// --------------------------------------------------------

__global__ void mcmp_zero_particle_forces_dip_D3Q19(particle3D_dip* pt,
							                        int nParts)
{
	// define particle:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nParts) {		
		pt[i].f = make_float3(0.0);
		pt[i].mass = 0.0;  // this gets added up when re-mapping particles on lattice
	}
}



// --------------------------------------------------------
// Update particle velocities and positions:
// --------------------------------------------------------

__global__ void mcmp_move_particles_dip_D3Q19(particle3D_dip* pt,
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

__global__ void mcmp_fix_particle_velocity_dip_D3Q19(particle3D_dip* pt,
                                                     float pvel,
   								                     int nParts)
{
	// define particle:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nParts) {	
		//printf("%f \n",pt[0].f.x);	
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

__global__ void mcmp_map_particles_to_lattice_dip_D3Q19(float* rS,			                  
                                                        particle3D_dip* pt,
													    int* x,
													    int* y,
													    int* z,
													    int* pIDgrid,
										                int nVoxels,
													    int nParts)
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
				
		// --------------------------------------------------	
		// default values:
		// --------------------------------------------------
				
		rS[i] = 0.0;
		pIDgrid[i] = -1;
		
		// --------------------------------------------------	
		// loop over particles:
		// --------------------------------------------------
		
		for (int j=0; j<nParts; j++) {
			
			// ---------------------------	
			// distance to particle c.o.m:
			// ---------------------------
			
			float dx = float(x[i]) - pt[j].r.x;
			float dy = float(y[i]) - pt[j].r.y;
			float dz = float(z[i]) - pt[j].r.z;
			float r = sqrt(dx*dx + dy*dy + dz*dz);
						
			// ---------------------------	
			// assign values:
			// ---------------------------
			
			float rI = pt[j].rInner;
			float rO = pt[j].rOuter;			
			if (r <= rO) {
				if (r < rI) {
					rS[i] = 1.0;
				}
				else {
					float rr = r - rI;
					rS[i] = 1.0 - rr/(rO-rI);					
				}
				pIDgrid[i] = j;	
				atomicAdd(&pt[j].mass,rS[i]);		
			}			
		}
	}
}



// --------------------------------------------------------
// D3Q19 set velocity on the y=0 and y=Ny-1 boundaries: 
// --------------------------------------------------------

__global__ void mcmp_set_boundary_velocity_dip_D3Q19(float uBC,
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
// D3Q19 initialize kernel: 
// --------------------------------------------------------

__global__ void mcmp_initial_equilibrium_dip_D3Q19(float* fA,
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
	
	// initialize populations to equilibrium values:
	if (i < nVoxels) {	
		
		int offst = 19*i;
		const float w0 = 1.0/3.0;
		const float ws = 1.0/18.0;
		const float wd = 1.0/36.0;
		const float omusq = 1.0 - 1.5*(u[i]*u[i] + v[i]*v[i] + w[i]*w[i]);
		
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
		evel = -u[i]; 
		feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+2] = feq*rA[i];
		fB[offst+2] = feq*rB[i];
		
		// dir 3
		evel = v[i];
		feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+3] = feq*rA[i];
		fB[offst+3] = feq*rB[i];
		
		// dir 4
		evel = -v[i];
		feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+4] = feq*rA[i];
		fB[offst+4] = feq*rB[i];
		
		// dir 5
		evel = w[i];
		feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+5] = feq*rA[i];
		fB[offst+5] = feq*rB[i];
		
		// dir 6
		evel = -w[i];
		feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+6] = feq*rA[i];
		fB[offst+6] = feq*rB[i];
		
		// dir 7
		evel = u[i] + v[i];
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+7] = feq*rA[i];
		fB[offst+7] = feq*rB[i];
		
		// dir 8
		evel = -(u[i] + v[i]);
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+8] = feq*rA[i];
		fB[offst+8] = feq*rB[i];
		
		// dir 9
		evel = u[i] + w[i];
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+9] = feq*rA[i];
		fB[offst+9] = feq*rB[i];
		
		// dir 10
		evel = -(u[i] + w[i]);
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+10] = feq*rA[i];
		fB[offst+10] = feq*rB[i];
		
		// dir 11
		evel = v[i] + w[i];
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+11] = feq*rA[i];
		fB[offst+11] = feq*rB[i];
		
		// dir 12
		evel = -(v[i] + w[i]);
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+12] = feq*rA[i];
		fB[offst+12] = feq*rB[i];
		
		// dir 13
		evel = u[i] - v[i];
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+13] = feq*rA[i];
		fB[offst+13] = feq*rB[i];
		
		// dir 14
		evel = v[i] - u[i];
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+14] = feq*rA[i];
		fB[offst+14] = feq*rB[i];
		
		// dir 15
		evel = u[i] - w[i];
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+15] = feq*rA[i];
		fB[offst+15] = feq*rB[i];
		
		// dir 16
		evel = w[i] - u[i];
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+16] = feq*rA[i];
		fB[offst+16] = feq*rB[i];
		
		// dir 17
		evel = v[i] - w[i];
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+17] = feq*rA[i];
		fB[offst+17] = feq*rB[i];
		
		// dir 18
		evel = w[i] - v[i];
		feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
		fA[offst+18] = feq*rA[i];
		fB[offst+18] = feq*rB[i];
			
	}
}



// --------------------------------------------------------
// D3Q19 compute velocity (barycentric) for the system.
// Here, the fluid velocity is calculated as normal, but
// it is amended to match the particle velocity.
// --------------------------------------------------------

__global__ void mcmp_compute_velocity_dip_D3Q19(float* fA,
                                                float* fB,
										        float* rA,
										        float* rB,
											    float* rS,
										        float* FxA,
										        float* FxB,
										        float* FyA,
										        float* FyB,
										        float* FzA,
										        float* FzB,
										        float* u,
										        float* v,
												float* w,											   
											    particle3D_dip* pt,											   
											    int* pIDgrid,
										        int nVoxels) 
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		// barycentric velocity		
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
		// modification due to particles
		int pID = pIDgrid[i];
		if (pID > -1) {
			float partvx = pt[pID].v.x;
			float partvy = pt[pID].v.y;
			float partvz = pt[pID].v.z;
			float partfx = (partvx - u[i])*2.0*rTotal*rS[i];
			float partfy = (partvy - v[i])*2.0*rTotal*rS[i];
			float partfz = (partvz - w[i])*2.0*rTotal*rS[i];
			// ammend fluid velocity
			u[i] += 0.5*partfx/rTotal;
			v[i] += 0.5*partfy/rTotal;
			w[i] += 0.5*partfz/rTotal;
			// ammend fluid forces
			FxA[i] += partfx*(rA[i]/rTotal); 
			FxB[i] += partfx*(rB[i]/rTotal);
			FyA[i] += partfy*(rA[i]/rTotal);
			FyB[i] += partfy*(rB[i]/rTotal);
			FzA[i] += partfz*(rA[i]/rTotal);
			FzB[i] += partfz*(rB[i]/rTotal);
			// ammend particle forces  (AtomicAdd!)
			atomicAdd(&pt[pID].f.x, -partfx);
			atomicAdd(&pt[pID].f.y, -partfy);
			atomicAdd(&pt[pID].f.z, -partfz);			
		}							
	}
}



// --------------------------------------------------------
// D3Q19 compute velocity (barycentric) for the system.
// Here, the fluid velocity is calculated by incorporating
// the particle velocity in the weighted sum.   
// -------------------------------------------------------- 

__global__ void mcmp_compute_velocity_dip_2_D3Q19(float* fA,
                                                  float* fB,
										          float* rA,
										          float* rB,
											      float* rS,
										          float* FxA,
										          float* FxB,
										          float* FyA,
										          float* FyB,
										          float* FzA,
										          float* FzB,
										          float* u,
										          float* v,	
												  float* w,										   
											      particle3D_dip* pt,											   
											      int* pIDgrid,
										          int nVoxels) 
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		// barycentric velocity		
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
		float rTotal = rA[i] + rB[i] + rS[i];
		// include contribution from particles:
		float rSVx = 0.0;
		float rSVy = 0.0;
		float rSVz = 0.0;
		int pID = pIDgrid[i]; 
		if (pID >= 0) {
			rSVx = rS[i]*pt[pID].v.x;
			rSVy = rS[i]*pt[pID].v.y;
			rSVz = rS[i]*pt[pID].v.z;
		}
		u[i] = (uA + uB + rSVx)/rTotal;
		v[i] = (vA + vB + rSVy)/rTotal;		
		w[i] = (wA + wB + rSVz)/rTotal;	
		// add force to particles:
		if (pID > -1) {
			float pFx = 2.0*rS[i]*(u[i] - pt[pID].v.x);
			float pFy = 2.0*rS[i]*(v[i] - pt[pID].v.y);
			float pFz = 2.0*rS[i]*(w[i] - pt[pID].v.z);
			atomicAdd(&pt[pID].f.x, pFx);
			atomicAdd(&pt[pID].f.y, pFy);
			atomicAdd(&pt[pID].f.z, pFz);
		}
	}
}



// --------------------------------------------------------
// D3Q19 compute density for each component: 
// --------------------------------------------------------

__global__ void mcmp_compute_density_dip_D3Q19(float* fA,
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

__global__ void mcmp_compute_SC_forces_dip_D3Q19(float* rA,
										         float* rB,
												 float* rS,
										         float* FxA,
										         float* FxB,
										         float* FyA,
										         float* FyB,
										         float* FzA,
										         float* FzB,
												 particle3D_dip* pt,
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
		int offst = i*19;
		
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
		
		float r0S = rS[i];
		float r1S = rS[nList[offst+1]];
		float r2S = rS[nList[offst+2]];
		float r3S = rS[nList[offst+3]];
		float r4S = rS[nList[offst+4]];
		float r5S = rS[nList[offst+5]];
		float r6S = rS[nList[offst+6]];
		float r7S = rS[nList[offst+7]];
		float r8S = rS[nList[offst+8]];
		float r9S = rS[nList[offst+9]];
		float r10S = rS[nList[offst+10]];
		float r11S = rS[nList[offst+11]];
		float r12S = rS[nList[offst+12]];
		float r13S = rS[nList[offst+13]];
		float r14S = rS[nList[offst+14]];
		float r15S = rS[nList[offst+15]];
		float r16S = rS[nList[offst+16]];
		float r17S = rS[nList[offst+17]];
		float r18S = rS[nList[offst+18]];
				
		float p0SA = r0S + omega*r0S*(1.0-r0S);		
		float p1SA = r1S + omega*r1S*(1.0-r1S);
		float p2SA = r2S + omega*r2S*(1.0-r2S);
		float p3SA = r3S + omega*r3S*(1.0-r3S);
		float p4SA = r4S + omega*r4S*(1.0-r4S);
		float p5SA = r5S + omega*r5S*(1.0-r5S);
		float p6SA = r6S + omega*r6S*(1.0-r6S);
		float p7SA = r7S + omega*r7S*(1.0-r7S);
		float p8SA = r8S + omega*r8S*(1.0-r8S);
		float p9SA = r9S + omega*r9S*(1.0-r9S);
		float p10SA = r10S + omega*r10S*(1.0-r10S);
		float p11SA = r11S + omega*r11S*(1.0-r11S);
		float p12SA = r12S + omega*r12S*(1.0-r12S);
		float p13SA = r13S + omega*r13S*(1.0-r13S);
		float p14SA = r14S + omega*r14S*(1.0-r14S);
		float p15SA = r15S + omega*r15S*(1.0-r15S);
		float p16SA = r16S + omega*r16S*(1.0-r16S);
		float p17SA = r17S + omega*r17S*(1.0-r17S);
		float p18SA = r18S + omega*r18S*(1.0-r18S);
		
		float p0SB = r0S - omega*r0S*(1.0-r0S);		
		float p1SB = r1S - omega*r1S*(1.0-r1S);
		float p2SB = r2S - omega*r2S*(1.0-r2S);
		float p3SB = r3S - omega*r3S*(1.0-r3S);
		float p4SB = r4S - omega*r4S*(1.0-r4S);
		float p5SB = r5S - omega*r5S*(1.0-r5S);
		float p6SB = r6S - omega*r6S*(1.0-r6S);
		float p7SB = r7S - omega*r7S*(1.0-r7S);
		float p8SB = r8S - omega*r8S*(1.0-r8S);
		float p9SB = r9S - omega*r9S*(1.0-r9S);
		float p10SB = r10S - omega*r10S*(1.0-r10S);
		float p11SB = r11S - omega*r11S*(1.0-r11S);
		float p12SB = r12S - omega*r12S*(1.0-r12S);
		float p13SB = r13S - omega*r13S*(1.0-r13S);
		float p14SB = r14S - omega*r14S*(1.0-r14S);
		float p15SB = r15S - omega*r15S*(1.0-r15S);
		float p16SB = r16S - omega*r16S*(1.0-r16S);
		float p17SB = r17S - omega*r17S*(1.0-r17S);
		float p18SB = r18S - omega*r18S*(1.0-r18S);
				
		// sum neighbor psi values times wi times ei
		float ws = 1.0/18.0;
		float wd = 1.0/36.0;		
		float sumNbrPsiAx = ws*p1A + wd*p7A + wd*p9A  + wd*p13A + wd*p15A - (ws*p2A + wd*p8A  + wd*p10A + wd*p14A + wd*p16A);
		float sumNbrPsiAy = ws*p3A + wd*p7A + wd*p11A + wd*p14A + wd*p17A - (ws*p4A + wd*p8A  + wd*p12A + wd*p13A + wd*p18A);
		float sumNbrPsiAz = ws*p5A + wd*p9A + wd*p11A + wd*p16A + wd*p18A - (ws*p6A + wd*p10A + wd*p12A + wd*p15A + wd*p17A);
		
		float sumNbrPsiBx = ws*p1B + wd*p7B + wd*p9B  + wd*p13B + wd*p15B - (ws*p2B + wd*p8B  + wd*p10B + wd*p14B + wd*p16B);
		float sumNbrPsiBy = ws*p3B + wd*p7B + wd*p11B + wd*p14B + wd*p17B - (ws*p4B + wd*p8B  + wd*p12B + wd*p13B + wd*p18B);
		float sumNbrPsiBz = ws*p5B + wd*p9B + wd*p11B + wd*p16B + wd*p18B - (ws*p6B + wd*p10B + wd*p12B + wd*p15B + wd*p17B);
		
		float sumNbrPsiSAx = ws*p1SA + wd*p7SA + wd*p9SA  + wd*p13SA + wd*p15SA - (ws*p2SA + wd*p8SA  + wd*p10SA + wd*p14SA + wd*p16SA);
		float sumNbrPsiSAy = ws*p3SA + wd*p7SA + wd*p11SA + wd*p14SA + wd*p17SA - (ws*p4SA + wd*p8SA  + wd*p12SA + wd*p13SA + wd*p18SA);
		float sumNbrPsiSAz = ws*p5SA + wd*p9SA + wd*p11SA + wd*p16SA + wd*p18SA - (ws*p6SA + wd*p10SA + wd*p12SA + wd*p15SA + wd*p17SA);
		
		float sumNbrPsiSBx = ws*p1SB + wd*p7SB + wd*p9SB  + wd*p13SB + wd*p15SB - (ws*p2SB + wd*p8SB  + wd*p10SB + wd*p14SB + wd*p16SB);
		float sumNbrPsiSBy = ws*p3SB + wd*p7SB + wd*p11SB + wd*p14SB + wd*p17SB - (ws*p4SB + wd*p8SB  + wd*p12SB + wd*p13SB + wd*p18SB);
		float sumNbrPsiSBz = ws*p5SB + wd*p9SB + wd*p11SB + wd*p16SB + wd*p18SB - (ws*p6SB + wd*p10SB + wd*p12SB + wd*p15SB + wd*p17SB);
				
		// fluid forces
		FxA[i] = -p0A*(gAB*sumNbrPsiBx + gAS*sumNbrPsiSAx);
		FxB[i] = -p0B*(gAB*sumNbrPsiAx + gBS*sumNbrPsiSBx);
		FyA[i] = -p0A*(gAB*sumNbrPsiBy + gAS*sumNbrPsiSAy);
		FyB[i] = -p0B*(gAB*sumNbrPsiAy + gBS*sumNbrPsiSBy);
		FzA[i] = -p0A*(gAB*sumNbrPsiBz + gAS*sumNbrPsiSAz);
		FzB[i] = -p0B*(gAB*sumNbrPsiAz + gBS*sumNbrPsiSBz);
		
		// particle forces
		int pID = pIDgrid[i];
		if (pID > -1) {
			float FxS = -(p0SA*gAS*sumNbrPsiAx + p0SB*gBS*sumNbrPsiBx);
			float FyS = -(p0SA*gAS*sumNbrPsiAy + p0SB*gBS*sumNbrPsiBy);
			float FzS = -(p0SA*gAS*sumNbrPsiAz + p0SB*gBS*sumNbrPsiBz);
			atomicAdd(&pt[pID].f.x, FxS);
			atomicAdd(&pt[pID].f.y, FyS);
			atomicAdd(&pt[pID].f.z, FzS);
		}
								
	}
}



// --------------------------------------------------------
// D3Q19 update kernel:
// --------------------------------------------------------

__global__ void mcmp_collide_stream_dip_D3Q19(float* f1A,
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



