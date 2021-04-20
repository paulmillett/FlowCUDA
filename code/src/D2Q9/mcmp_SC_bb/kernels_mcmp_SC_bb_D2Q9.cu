
# include "kernels_mcmp_SC_bb_D2Q9.cuh"
# include "../mcmp_SC/mcmp_pseudopotential.cuh"
# include <stdio.h>



// --------------------------------------------------------
// Zero particle forces:
// --------------------------------------------------------

__global__ void mcmp_zero_particle_forces_bb_D2Q9(particle2D_bb* pt,
							                      int nParts)
{
	// define particle:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nParts) {		
		pt[i].f = make_float2(0.0);
	}
}



// --------------------------------------------------------
// Update particle velocities and positions:
// --------------------------------------------------------

__global__ void mcmp_move_particles_bb_D2Q9(particle2D_bb* pt,
   								            int nParts)
{
	// define particle:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nParts) {	
		printf("particle force-x = %f \n",pt[i].f.x); 
		float2 a = pt[i].f/pt[i].mass;
		pt[i].r += pt[i].v + 0.5*a;  // assume dt = 1
		pt[i].v += a;
	}
}



// --------------------------------------------------------
// Fix particle velocity:
// --------------------------------------------------------

__global__ void mcmp_fix_particle_velocity_bb_D2Q9(particle2D_bb* pt,
                                                   float pvel,
   								                   int nParts)
{
	// define particle:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nParts) {	
		pt[i].f = make_float2(0.0);
		if (i == 0) {
			pt[i].v.x = -pvel;
			pt[i].v.y = 0.00;
		}
		if (i == 1) {
			pt[i].v.x = pvel;
			pt[i].v.y = 0.00;
		}		
	}
}



// --------------------------------------------------------
// Calculate particle-particle forces:
// --------------------------------------------------------

__global__ void mcmp_particle_particle_forces_bb_D2Q9(particle2D_bb* pt,
                                                      float K,
													  float halo,
   								                      int nParts)
{
	// define particle:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nParts) {	
		for (int j=0; j<nParts; j++) {
			if (i==j) continue;
			float2 rij = pt[i].r - pt[j].r;
			float rr = length(rij);
			// Hertz contact force:
			float twoRadii = pt[i].rad + pt[j].rad + halo;
			if (rr < twoRadii) {
				float fmag = 2.5*K*pow(twoRadii - rr,1.5);
				pt[i].f += fmag*(rij/rr);
			}
		}		
	}
}



// --------------------------------------------------------
// D2Q9 initialize populations to equilibrium values: 
// --------------------------------------------------------

__global__ void mcmp_initial_equilibrium_bb_D2Q9(float* fA,
                                                 float* fB,
										         float* rA,
											     float* rB,
										         float* u,
										         float* v,
										         int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {			
		int offst = 9*i;
		equilibrium_populations_bb_D2Q9(fA,fB,rA[i],rB[i],u[i],v[i],offst);		
	}
}



// --------------------------------------------------------
// D2Q9 equilibirium populations: 
// --------------------------------------------------------

__device__ void equilibrium_populations_bb_D2Q9(float* fA,
                                                float* fB,
										        float rA,
											    float rB,
										        float u,
										        float v,
												int offst)
{
	const float w0 = 4.0/9.0;
	const float ws = 1.0/9.0;
	const float wd = 1.0/36.0;
	const float omusq = 1.0 - 1.5*(u*u + v*v);	
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
	evel = v;
	feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+2] = feq*rA;
	fB[offst+2] = feq*rB;	
	// dir 3
	evel = -u;
	feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+3] = feq*rA;
	fB[offst+3] = feq*rB;	
	// dir 4
	evel = -v;
	feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+4] = feq*rA;
	fB[offst+4] = feq*rB;	
	// dir 5
	evel = u + v;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+5] = feq*rA;
	fB[offst+5] = feq*rB;	
	// dir 6
	evel = -u + v;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+6] = feq*rA;
	fB[offst+6] = feq*rB;	
	// dir 7
	evel = -u - v;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+7] = feq*rA;
	fB[offst+7] = feq*rB;	
	// dir 8
	evel = u - v;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	fA[offst+8] = feq*rA;
	fB[offst+8] = feq*rB;
}



// --------------------------------------------------------
// Map particles to grid by updating s[] and pIDgrid[]:
// --------------------------------------------------------

__global__ void mcmp_map_particles_on_lattice_bb_D2Q9(particle2D_bb* pt,
                                                      int* x,
						    		                  int* y,
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
			float rr = sqrt(dx*dx + dy*dy);
			if (rr <= pt[j].rad) {
				s[i] = 1;
				pIDgrid[i] = j;					
			}		
		}							
	}
}



// --------------------------------------------------------
// D2Q9 kernel to cover/uncover voxels as particles move: 
// --------------------------------------------------------

__global__ void mcmp_cover_uncover_bb_D2Q9(int* s,
                                           int* sprev,
										   int* nList,
										   float* u,
										   float* v,
										   float* rA,
										   float* rB,
										   float* fA,
										   float* fB,
										   int nVoxels)
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nVoxels) {
		if (sprev[i] == 1 && s[i] == 1) stay_covered_D2Q9(i,fA,fB);		
		if (sprev[i] == 0 && s[i] == 1) cover_voxel_D2Q9(i,s,sprev,nList,u,v,rA,rB,fA,fB);
		if (sprev[i] == 1 && s[i] == 0) uncover_voxel_D2Q9(i,s,sprev,nList,u,v,rA,rB,fA,fB);			
	}
}



// --------------------------------------------------------
// D2Q9 kernel to keep solid site populations zero: 
// --------------------------------------------------------
	
__device__ void stay_covered_D2Q9(int i,
                                  float* fA,
								  float* fB)
{
	int offst = 9*i;		
	for (int n=0; n<9; n++) {
		fA[offst+n] = 0.0;
		fB[offst+n] = 0.0;
	}
}



// --------------------------------------------------------
// D2Q9 kernel to cover lattice site: 
// --------------------------------------------------------

__device__ void cover_voxel_D2Q9(int i,
                                 int* s,
								 int* sprev,
								 int* nList,
								 float* u,
								 float* v,
								 float* rA,
								 float* rB,
								 float* fA,
								 float* fB)
{
	
	// --------------------------------------------	
	// sum up all the neighbors that are fluid:
	// --------------------------------------------
	
	int offst = 9*i;
	
	/*
	int nfn = 0;	
	for (int n=1; n<9; n++) {
		int nabor = nList[offst+n];
		if (sprev[nabor] == 0 && s[nabor] == 0) nfn++;
	}
	
	// --------------------------------------------	
	// determine the density to distribute to each
	// neighbor:
	// --------------------------------------------
	
	float rAdist = 0.0;
	float rBdist = 0.0;
	if (nfn > 0) {
		rAdist = rA[i]/float(nfn);
		rBdist = rB[i]/float(nfn);
	}
	
	// --------------------------------------------	
	// add current voxel's density to neighboring
	// fluid voxel densities:
	// --------------------------------------------
		
	for (int n=1; n<9; n++) {
		int nabor = nList[offst+n];
		if (sprev[nabor] == 0 && s[nabor] == 0) {
			add_density_to_populations_D2Q9(nabor,rAdist,rBdist,u[nabor],v[nabor],fA,fB);
		}
	}
	*/
	
	// --------------------------------------------	
	// zero the populations for this voxel:
	// --------------------------------------------
	
	for (int n=0; n<9; n++) {
		fA[offst+n] = 0.0;
		fB[offst+n] = 0.0;
	}
		
}



// --------------------------------------------------------
// D2Q9 kernel to cover lattice site: 
// --------------------------------------------------------

__device__ void uncover_voxel_D2Q9(int i,
                                   int* s,
								   int* sprev,
								   int* nList,
								   float* u,
								   float* v,
								   float* rA,
								   float* rB,
								   float* fA,
								   float* fB)
{
	
	// --------------------------------------------	
	// sum up all the neighbors that are fluid:
	// --------------------------------------------
	
	int nfn = 0;
	int offst = 9*i;
	float avenbrRA = 0.0;
	float avenbrRB = 0.0;
	for (int n=1; n<9; n++) {
		int nabor = nList[offst+n];
		if (sprev[nabor] == 0 && s[nabor] == 0) {
			nfn++;
			avenbrRA += rA[nabor];
			avenbrRB += rB[nabor];
		}
	}
	avenbrRA /= float(nfn);
	avenbrRB /= float(nfn);
	
	// --------------------------------------------	
	// assign the equilibrium populations:
	// --------------------------------------------
	
	equilibrium_populations_bb_D2Q9(fA,fB,avenbrRA,avenbrRB,u[i],v[i],offst);
	
	// --------------------------------------------	
	// reduce neighboring fluid densities to
	// conserve mass:
	// --------------------------------------------
	
	/*
	float rAdist = -avenbrRA/float(nfn);
	float rBdist = -avenbrRB/float(nfn);
	for (int n=1; n<9; n++) {
		int nabor = nList[offst+n];
		if (sprev[nabor] == 0 && s[nabor] == 0) {
			add_density_to_populations_D2Q9(nabor,rAdist,rBdist,u[nabor],v[nabor],fA,fB);
		}
	}
	*/
	
}



// --------------------------------------------------------
// D2Q9 kernel to add density to populations: 
// --------------------------------------------------------

__device__ void add_density_to_populations_D2Q9(int i,
                                                float rAdist,
												float rBdist,
												float u,
												float v,
												float* fA,
												float* fB)
{
	const int offst = 9*i;
	const float w0 = 4.0/9.0;
	const float ws = 1.0/9.0;
	const float wd = 1.0/36.0;	
	const float omusq = 1.0 - 1.5*(u*u + v*v);
	// dir 0
	float feq = w0*omusq;
	atomicAdd(&fA[offst+0], feq*rAdist);
	atomicAdd(&fB[offst+0], feq*rBdist);
	// dir 1
	float evel = u;
	feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
	atomicAdd(&fA[offst+1], feq*rAdist);
	atomicAdd(&fB[offst+1], feq*rBdist);
	// dir 2
	evel = v;
	feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
	atomicAdd(&fA[offst+2], feq*rAdist);
	atomicAdd(&fB[offst+2], feq*rBdist);
	// dir 3
	evel = -u;
	feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
	atomicAdd(&fA[offst+3], feq*rAdist);
	atomicAdd(&fB[offst+3], feq*rBdist);
	// dir 4
	evel = -v;
	feq = ws*(omusq + 3.0*evel + 4.5*evel*evel);
	atomicAdd(&fA[offst+4], feq*rAdist);
	atomicAdd(&fB[offst+4], feq*rBdist);
	// dir 5
	evel = u + v;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	atomicAdd(&fA[offst+5], feq*rAdist);
	atomicAdd(&fB[offst+5], feq*rBdist);
	// dir 6
	evel = -u + v;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	atomicAdd(&fA[offst+6], feq*rAdist);
	atomicAdd(&fB[offst+6], feq*rBdist);
	// dir 7
	evel = -u - v;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	atomicAdd(&fA[offst+7], feq*rAdist);
	atomicAdd(&fB[offst+7], feq*rBdist);
	// dir 8
	evel = u - v;
	feq = wd*(omusq + 3.0*evel + 4.5*evel*evel);
	atomicAdd(&fA[offst+8], feq*rAdist);	
	atomicAdd(&fB[offst+8], feq*rBdist);
}



// --------------------------------------------------------
// D2Q9 kernel to update the particle fields on the lattice: 
// --------------------------------------------------------

__global__ void mcmp_update_particles_on_lattice_D2Q9(float* fA,
                                                      float* fB,
										              float* rA,
											          float* rB,
										              float* u,
										              float* v,
													  particle2D_bb* pt,
													  int* x,
													  int* y,
													  int* s,
													  int* pIDgrid,
													  int* nList,													  
										              int nVoxels,
													  int nParts)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {	
		
		int offst = 9*i;
		
		// --------------------------------------------	
		// get the current state of voxel:
		// --------------------------------------------
		
		int s0 = s[i];
		
		// --------------------------------------------	
		// get the new state of voxel by seeing if any
		// particles overlay it:
		// --------------------------------------------
				
		int s1 = 0;
		int pID = -1;
		float partvx = 0.0;
		float partvy = 0.0;
		for (int p=0; p<nParts; p++) {
			float dx = float(x[i]) - pt[p].r.x;
			float dy = float(y[i]) - pt[p].r.y;
			float rp = sqrt(dx*dx + dy*dy);
			if (rp <= pt[p].rad) {
				s1 = 1;
				pID = p;
				partvx = pt[p].v.x;
				partvy = pt[p].v.y;
			}		
		}
		s[i] = s1;
		
		// --------------------------------------------	
		// decide course of action:
		// --------------------------------------------
		
		// fluid site STAYS fluid site
		if (s0 == 0 && s1 == 0) {
			pIDgrid[i] = -1;  // this is redundant, but that's OK
		}
		
		// particle site STAYS particle site
		else if (s0 == 1 && s1 == 1) {			
			u[i] = partvx;
			v[i] = partvy;
			pIDgrid[i] = pID;
			// zero all the populations			
			for (int n=0; n<9; n++) {
				fA[offst+n] = 0.0;
				fB[offst+n] = 0.0;
			}
		}
		
		// fluid site becomes particle site (COVERING)
		else if (s0 == 0 && s1 == 1) {						
			// update velocity to particle's velocity
			u[i] = partvx;
			v[i] = partvy;
			pIDgrid[i] = pID;
			// zero all the populations			
			for (int n=0; n<9; n++) {
				fA[offst+n] = 0.0;
				fB[offst+n] = 0.0;
			}
		}
		
		// particle site becomes fluid site (UNCOVERING)
		else if (s0 == 1 && s1 == 0) {				
			// assign voxel velocity with particle velocity
			int pID = pIDgrid[i];
			u[i] = pt[pID].v.x; 
			v[i] = pt[pID].v.y;
			pIDgrid[i] = -1;
			// get average density of surrounding fluid sites:
			int num_fluid_nabors = 0;
			float aver_rA_nabors = 0.0;
			float aver_rB_nabors = 0.0;			
			for (int n=1; n<9; n++) {  // do not include self, n=0
				int nID = nList[offst+n];
				if (s[nID] == 0) {
					num_fluid_nabors++;
					aver_rA_nabors += rA[nID];
					aver_rB_nabors += rB[nID];
				}
			}			
			if (num_fluid_nabors > 0) {
				rA[i] = aver_rA_nabors/num_fluid_nabors;
				rB[i] = aver_rB_nabors/num_fluid_nabors;
			}
			// set populations to the equilibrium for the given
			// velocity and density:
			equilibrium_populations_bb_D2Q9(fA,fB,rA[i],rB[i],u[i],v[i],offst);
			
		}				
	}
}



// --------------------------------------------------------
// D2Q9 compute velocity (barycentric) for the system: 
// --------------------------------------------------------

__global__ void mcmp_compute_velocity_bb_D2Q9(float* fA,
                                              float* fB,
										      float* rA,
										      float* rB,
										      float* FxA,
										      float* FxB,
										      float* FyA,
										      float* FyB,
										      float* u,
										      float* v,
											  int* s,
											  int* pIDgrid,
											  particle2D_bb* pt,
										      int nVoxels) 
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		
		// --------------------------------------------------	
		// if this is a fluid site:
		// --------------------------------------------------
		
		if (s[i] == 0) {
			int offst = i*9;			
			float uA = fA[offst+1] + fA[offst+5] + fA[offst+8] - (fA[offst+3] + fA[offst+6] + fA[offst+7]) + 0.5*FxA[i];
			float uB = fB[offst+1] + fB[offst+5] + fB[offst+8] - (fB[offst+3] + fB[offst+6] + fB[offst+7]) + 0.5*FxB[i];
			float vA = fA[offst+2] + fA[offst+5] + fA[offst+6] - (fA[offst+4] + fA[offst+7] + fA[offst+8]) + 0.5*FyA[i];
			float vB = fB[offst+2] + fB[offst+5] + fB[offst+6] - (fB[offst+4] + fB[offst+7] + fB[offst+8]) + 0.5*FyB[i];
			float rTotal = rA[i] + rB[i];
			u[i] = (uA + uB)/rTotal;
			v[i] = (vA + vB)/rTotal;	
		}
		
		// --------------------------------------------------	
		// if this is a solid site:
		// --------------------------------------------------
		
		else if (s[i] == 1) {
			int pID = pIDgrid[i];
			u[i] = pt[pID].v.x;
			v[i] = pt[pID].v.y;	
		}
					
	}
}



// --------------------------------------------------------
// D2Q9 set velocity on the y=0 and y=Ny-1 boundaries: 
// --------------------------------------------------------

__global__ void mcmp_set_boundary_velocity_bb_D2Q9(float uBC,
                                                   float vBC,
	                                               float* rA,
										           float* rB,
										           float* FxA,
										           float* FxB,
										           float* FyA,
										           float* FyB,
										           float* u,
										           float* v,
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
			u[i] += 0.5*fxBC/rTotal;
			v[i] += 0.5*fyBC/rTotal;
			FxA[i] += fxBC*(rA[i]/rTotal);
			FxB[i] += fxBC*(rB[i]/rTotal);
			FyA[i] += fyBC*(rA[i]/rTotal);
			FyB[i] += fyBC*(rB[i]/rTotal);
		}		
	}
}



// --------------------------------------------------------
// D2Q9 set shear velocity on the y=0 and y=Ny-1 boundaries: 
// --------------------------------------------------------

__global__ void mcmp_set_boundary_shear_velocity_bb_D2Q9(float uBot,
                                                         float uTop,
	                                                     float* rA,
										                 float* rB,
										                 float* FxA,
										                 float* FxB,
										                 float* FyA,
										                 float* FyB,
										                 float* u,
										                 float* v,
												         int* y,											        
											             int Ny,
										                 int nVoxels) 
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		if (y[i] == 0) {
			float rTotal = rA[i] + rB[i];
			float fxBC = (uBot - u[i])*2.0*rTotal;
			float fyBC = (0.0  - v[i])*2.0*rTotal;
			u[i] += 0.5*fxBC/rTotal;
			v[i] += 0.5*fyBC/rTotal;
			FxA[i] += fxBC*(rA[i]/rTotal);
			FxB[i] += fxBC*(rB[i]/rTotal);
			FyA[i] += fyBC*(rA[i]/rTotal);
			FyB[i] += fyBC*(rB[i]/rTotal);
		} 
		if (y[i] == Ny-1) {
			float rTotal = rA[i] + rB[i];
			float fxBC = (uTop - u[i])*2.0*rTotal;
			float fyBC = (0.0  - v[i])*2.0*rTotal;
			u[i] += 0.5*fxBC/rTotal;
			v[i] += 0.5*fyBC/rTotal;
			FxA[i] += fxBC*(rA[i]/rTotal);
			FxB[i] += fxBC*(rB[i]/rTotal);
			FyA[i] += fyBC*(rA[i]/rTotal);
			FyB[i] += fyBC*(rB[i]/rTotal);
		}		
	}
}



// --------------------------------------------------------
// D2Q9 compute density for each component: 
// --------------------------------------------------------

__global__ void mcmp_compute_density_bb_D2Q9(float* fA,
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
// D2Q9 compute virtual density for each component: 
// --------------------------------------------------------

__global__ void mcmp_compute_virtual_density_bb_D2Q9(float* rAvirt,
                                        	         float* rBvirt,
										             float* rA,
										             float* rB,
													 int* s,
													 int* nList,
													 float omega,
										             int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
		
		int offst = i*9;
		
		// fluid node:
		if (s[i] == 0) {
			rAvirt[i] = 0.0;
			rBvirt[i] = 0.0;
		}
		
		// solid node:
		if (s[i] == 1) {
			const float ws = 1.0/9.0;
			const float wd = 1.0/36.0;			
			float r1A = rA[nList[offst+1]];
			float r2A = rA[nList[offst+2]];
			float r3A = rA[nList[offst+3]];
			float r4A = rA[nList[offst+4]];
			float r5A = rA[nList[offst+5]];
			float r6A = rA[nList[offst+6]];
			float r7A = rA[nList[offst+7]];
			float r8A = rA[nList[offst+8]];	
			float r1B = rB[nList[offst+1]];
			float r2B = rB[nList[offst+2]];
			float r3B = rB[nList[offst+3]];
			float r4B = rB[nList[offst+4]];
			float r5B = rB[nList[offst+5]];
			float r6B = rB[nList[offst+6]];
			float r7B = rB[nList[offst+7]];
			float r8B = rB[nList[offst+8]];				
			float s1 = ws*(1 - s[nList[offst+1]]);
			float s2 = ws*(1 - s[nList[offst+2]]);
			float s3 = ws*(1 - s[nList[offst+3]]);
			float s4 = ws*(1 - s[nList[offst+4]]);
			float s5 = wd*(1 - s[nList[offst+5]]);
			float s6 = wd*(1 - s[nList[offst+6]]);
			float s7 = wd*(1 - s[nList[offst+7]]);
			float s8 = wd*(1 - s[nList[offst+8]]);			
			float sumRA = s1*r1A + s2*r2A + s3*r3A + s4*r4A + 
				          s5*r5A + s6*r6A + s7*r7A + s8*r8A;
			float sumRB = s1*r1B + s2*r2B + s3*r3B + s4*r4B + 
				          s5*r5B + s6*r6B + s7*r7B + s8*r8B;
			float sumWS = s1+s2+s3+s4+s5+s6+s7+s8;		
			if (sumWS > 0.0) {
				rAvirt[i] = sumRA/sumWS*(1.0+omega);
				rBvirt[i] = sumRB/sumWS*(1.0-omega);	
			}	
			else {
				rAvirt[i] = 0.0;
				rBvirt[i] = 0.0;	
			}
					
		}		
	}
}



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
											   particle2D_bb* pt,
											   int* pIDgrid,
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
	
		const float ws = 1.0/9.0;
		const float wd = 1.0/36.0;		
		float sumNbrRhoAx = ws*r1A + wd*r5A + wd*r8A - (ws*r3A + wd*r6A + wd*r7A);
		float sumNbrRhoAy = ws*r2A + wd*r5A + wd*r6A - (ws*r4A + wd*r7A + wd*r8A);
		float sumNbrRhoBx = ws*r1B + wd*r5B + wd*r8B - (ws*r3B + wd*r6B + wd*r7B);
		float sumNbrRhoBy = ws*r2B + wd*r5B + wd*r6B - (ws*r4B + wd*r7B + wd*r8B);
		float sumNbrSx = ws*s1 + wd*s5 + wd*s8 - (ws*s3 + wd*s6 + wd*s7);
		float sumNbrSy = ws*s2 + wd*s5 + wd*s6 - (ws*s4 + wd*s7 + wd*s8);
		
		// --------------------------------------------	
		// if this is a fluid site:
		// --------------------------------------------
		
		if (s[i] == 0) {				
			FxA[i] = -r0A*(gAB*sumNbrRhoBx + gAS*sumNbrSx);
			FxB[i] = -r0B*(gAB*sumNbrRhoAx + gBS*sumNbrSx);
			FyA[i] = -r0A*(gAB*sumNbrRhoBy + gAS*sumNbrSy);
			FyB[i] = -r0B*(gAB*sumNbrRhoAy + gBS*sumNbrSy);						
		}
		
		// --------------------------------------------	
		// if this is a solid site:
		// --------------------------------------------
		
		if (s[i] == 1) {			
			float fxFS = -(gAS*sumNbrRhoAx + gBS*sumNbrRhoBx);
			float fyFS = -(gAS*sumNbrRhoAy + gBS*sumNbrRhoBy);
			int pID = pIDgrid[i];
			atomicAdd(&pt[pID].f.x, fxFS);
			atomicAdd(&pt[pID].f.y, fyFS);							
		}
		
	}
}



// --------------------------------------------------------
// D2Q9 compute Shan-Chen forces for the components
// using pseudo-potential, psi = rho_0(1-exp(-rho/rho_o))
//
// Note: here we use the virtual fluid as described in
//       Jansen & Harting, PRE, 83, 046707 (2011).
//
// --------------------------------------------------------

__global__ void mcmp_compute_SC_forces_bb_2_D2Q9(float* rA,
										         float* rB,	
												 float* rAvirt,
												 float* rBvirt,										   
										         float* FxA,
										         float* FxB,
										         float* FyA,
										         float* FyB,
											     particle2D_bb* pt,
											     int* pIDgrid,
											     int* s,
											     int* nList,
											     float gAB,											     
										         int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {
							
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
		
		float r0AV = psi(rAvirt[i]);	
		float r1AV = psi(rAvirt[nList[offst+1]]);
		float r2AV = psi(rAvirt[nList[offst+2]]);
		float r3AV = psi(rAvirt[nList[offst+3]]);
		float r4AV = psi(rAvirt[nList[offst+4]]);
		float r5AV = psi(rAvirt[nList[offst+5]]);
		float r6AV = psi(rAvirt[nList[offst+6]]);
		float r7AV = psi(rAvirt[nList[offst+7]]);
		float r8AV = psi(rAvirt[nList[offst+8]]);
	
		float r0BV = psi(rBvirt[i]);		
		float r1BV = psi(rBvirt[nList[offst+1]]);
		float r2BV = psi(rBvirt[nList[offst+2]]);
		float r3BV = psi(rBvirt[nList[offst+3]]);
		float r4BV = psi(rBvirt[nList[offst+4]]);
		float r5BV = psi(rBvirt[nList[offst+5]]);
		float r6BV = psi(rBvirt[nList[offst+6]]);
		float r7BV = psi(rBvirt[nList[offst+7]]);
		float r8BV = psi(rBvirt[nList[offst+8]]);
	
		const float ws = 1.0/9.0;
		const float wd = 1.0/36.0;		
		float sumNbrRhoAx = ws*r1A + wd*r5A + wd*r8A - (ws*r3A + wd*r6A + wd*r7A);
		float sumNbrRhoAy = ws*r2A + wd*r5A + wd*r6A - (ws*r4A + wd*r7A + wd*r8A);
		float sumNbrRhoBx = ws*r1B + wd*r5B + wd*r8B - (ws*r3B + wd*r6B + wd*r7B);
		float sumNbrRhoBy = ws*r2B + wd*r5B + wd*r6B - (ws*r4B + wd*r7B + wd*r8B);		
		
		// --------------------------------------------	
		// if this is a fluid site:
		// --------------------------------------------
		
		if (s[i] == 0) {	
			float sumNbrRhoAVx = ws*r1AV + wd*r5AV + wd*r8AV - (ws*r3AV + wd*r6AV + wd*r7AV);
			float sumNbrRhoAVy = ws*r2AV + wd*r5AV + wd*r6AV - (ws*r4AV + wd*r7AV + wd*r8AV);
			float sumNbrRhoBVx = ws*r1BV + wd*r5BV + wd*r8BV - (ws*r3BV + wd*r6BV + wd*r7BV);
			float sumNbrRhoBVy = ws*r2BV + wd*r5BV + wd*r6BV - (ws*r4BV + wd*r7BV + wd*r8BV);				
			FxA[i] = -r0A*gAB*(sumNbrRhoBx + sumNbrRhoBVx);
			FxB[i] = -r0B*gAB*(sumNbrRhoAx + sumNbrRhoAVx);
			FyA[i] = -r0A*gAB*(sumNbrRhoBy + sumNbrRhoBVy);
			FyB[i] = -r0B*gAB*(sumNbrRhoAy + sumNbrRhoAVy);						
		}
		
		// --------------------------------------------	
		// if this is a solid site:
		// --------------------------------------------
		
		if (s[i] == 1) {
			float fxAV = -r0AV*gAB*(sumNbrRhoBx);
			float fxBV = -r0BV*gAB*(sumNbrRhoAx);
			float fyAV = -r0AV*gAB*(sumNbrRhoBy);
			float fyBV = -r0BV*gAB*(sumNbrRhoAy);		
			int pID = pIDgrid[i];
			atomicAdd(&pt[pID].f.x, fxAV + fxBV);
			atomicAdd(&pt[pID].f.y, fyAV + fyBV);							
		}
		
	}
}



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
}



// --------------------------------------------------------
// D2Q9 implement bounce-back conditions:
// --------------------------------------------------------

__global__ void mcmp_bounce_back_D2Q9(float* f2A, 
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
			
			int offst = 9*i;
						
			// dir 1 bounce-back to nabor 3 as dir 3:
			if (s[nList[offst+3]] == 0) {
				f2A[streamIndex[offst+3]] = f2A[offst+1];
				f2B[streamIndex[offst+3]] = f2B[offst+1];
				f2A[offst+1] = 0.0;
				f2B[offst+1] = 0.0;
			}
			
			// dir 2 bounce-back to nabor 4 as dir 4:
			if (s[nList[offst+4]] == 0) {
				f2A[streamIndex[offst+4]] = f2A[offst+2];
				f2B[streamIndex[offst+4]] = f2B[offst+2];
				f2A[offst+2] = 0.0;
				f2B[offst+2] = 0.0;
			}
			
			// dir 3 bounce-back to nabor 1 as dir 1:
			if (s[nList[offst+1]] == 0) {
				f2A[streamIndex[offst+1]] = f2A[offst+3];
				f2B[streamIndex[offst+1]] = f2B[offst+3];
				f2A[offst+3] = 0.0;
				f2B[offst+3] = 0.0;
			}
			
			// dir 4 bounce-back to nabor 2 as dir 2:
			if (s[nList[offst+2]] == 0) {
				f2A[streamIndex[offst+2]] = f2A[offst+4];
				f2B[streamIndex[offst+2]] = f2B[offst+4];
				f2A[offst+4] = 0.0;
				f2B[offst+4] = 0.0;
			}
			
			// dir 5 bounce-back to nabor 7 as dir 7:
			if (s[nList[offst+7]] == 0) {
				f2A[streamIndex[offst+7]] = f2A[offst+5];
				f2B[streamIndex[offst+7]] = f2B[offst+5];
				f2A[offst+5] = 0.0;
				f2B[offst+5] = 0.0;
			}
			
			// dir 6 bounce-back to nabor 8 as dir 8:
			if (s[nList[offst+8]] == 0) {
				f2A[streamIndex[offst+8]] = f2A[offst+6];
				f2B[streamIndex[offst+8]] = f2B[offst+6];
				f2A[offst+6] = 0.0;
				f2B[offst+6] = 0.0;
			}
			
			// dir 7 bounce-back to nabor 5 as dir 5:
			if (s[nList[offst+5]] == 0) {
				f2A[streamIndex[offst+5]] = f2A[offst+7];
				f2B[streamIndex[offst+5]] = f2B[offst+7];
				f2A[offst+7] = 0.0;
				f2B[offst+7] = 0.0;
			}
			
			// dir 8 bounce-back to nabor 6 as dir 6:
			if (s[nList[offst+6]] == 0) {
				f2A[streamIndex[offst+6]] = f2A[offst+8];
				f2B[streamIndex[offst+6]] = f2B[offst+8];
				f2A[offst+8] = 0.0;
				f2B[offst+8] = 0.0;
			}
			
		}	
	}		
}



// --------------------------------------------------------
// D2Q9 implement bounce-back conditions for moving
// solids:
// --------------------------------------------------------

__global__ void mcmp_bounce_back_moving_D2Q9(float* f2A, 
									         float* f2B,
											 float* rA,
											 float* rB,
											 float* u,
											 float* v,
											 particle2D_bb* pt,
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
						
			int offst = 9*i;
			int pID = pIDgrid[i];
			const float ws = 1.0/9.0;
			const float wd = 1.0/36.0;
			float meF2S  = 0.0;  // momentum exchange fluid to solid
			float meF2Sx = 0.0;  // momentum exchange fluid to solid (x)
			float meF2Sy = 0.0;  // momentum exchange fluid to solid (y)
						
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



