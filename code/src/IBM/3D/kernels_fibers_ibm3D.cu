
# include "kernels_fibers_ibm3D.cuh"
# include <stdio.h>


// --------------------------------------------------------
//
// These kernels implement the implicit finite-difference
// model for a flexible filament given by:
//
// Huang WX, Shin SJ, Sung HJ.  Simulation of flexible 
// filaments in a uniform flow by the immersed boundary
// method.  Journal of Computational Physics 226 (2007)
// 2206-2228.
// 
// --------------------------------------------------------






// --------------------------------------------------------
// IBM3D kernel to zero bead forces:
// --------------------------------------------------------

__global__ void zero_bead_forces_fibers_IBM3D(
	beadfiber* beads,	
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		beads[i].f = make_float3(0.0f,0.0f,0.0f);
	}
}



// --------------------------------------------------------
// IBM3D kernel to zero bead forces:
// --------------------------------------------------------

__global__ void calculate_bead_velocity_fibers_IBM3D(
	beadfiber* beads,	
	float dt,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		beads[i].v = (beads[i].r - beads[i].rm1)/dt;
	}
}



// --------------------------------------------------------
// IBM3D kernel to update rstar
// --------------------------------------------------------

__global__ void update_rstar_fibers_IBM3D(
	beadfiber* beads,
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		beads[i].rstar = 2*beads[i].r - beads[i].rm1;			
	}
}



// --------------------------------------------------------
// IBM3D kernel to update bead positions
// --------------------------------------------------------

__global__ void update_bead_positions_fibers_IBM3D(
	beadfiber* beads,
	float* xp1,
	float* yp1,
	float* zp1,
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		beads[i].rm1 = beads[i].r;
		beads[i].r = make_float3(xp1[i], yp1[i], zp1[i]);		
	}
}


	
// --------------------------------------------------------
// IBM3D kernel to compute Laplacian (d2r/dS2) for beads
// needed to calculate the bending force
// --------------------------------------------------------

__global__ void compute_Laplacian_fibers_IBM3D(
	beadfiber* beads,
	float dS,
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		// internal bead:
		if (beads[i].posID == 0) {
			beads[i].d2r = (beads[i+1].rstar - 2.0*beads[i].rstar + beads[i-1].rstar)/(dS*dS);
		} 
		// end bead:
		else {			
			beads[i].d2r = make_float3(0.0f,0.0f,0.0f);;
		}	
	}	
}
		
		

// --------------------------------------------------------
// IBM3D kernel to compute bending force on beads 
// --------------------------------------------------------

__global__ void compute_bending_force_fibers_IBM3D(
	beadfiber* beads,
	float dS,
	float gam,
	int nBeads)
{		
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nBeads) {
		// internal bead:
		if (beads[i].posID == 0) {			
			beads[i].f -= gam*(beads[i+1].d2r - 2.0*beads[i].d2r + beads[i-1].d2r)/(dS*dS);
		} 
		// left-most bead:
		else if (beads[i].posID == 1) {			
			beads[i].f -= gam*(beads[i+2].d2r - beads[i+1].d2r)/(dS*dS);
		} 
		// right-most bead:
		else if (beads[i].posID == 2) {
			beads[i].f -= gam*(beads[i-2].d2r - beads[i-1].d2r)/(dS*dS);
		}		
	}	
}



// --------------------------------------------------------
// IBM3D kernel to compute RHS of tension equation, done
// using the edges
// --------------------------------------------------------

__global__ void compute_tension_RHS_fibers_IBM3D(
	beadfiber* beads,
	edgefiber* edges,
	float* B,
	float dS,
	float dt,
	int nEdges)
{	
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nEdges) {
		
		// left and right beads:
		int b0 = edges[i].b0;
		int b1 = edges[i].b1;
		
		// first term:
		float3 d1r = (beads[b1].r - beads[b0].r)/dS;
		float3 d1rm1 = (beads[b1].rm1 - beads[b0].rm1)/dS;
		float d1rdot = dot(d1r,d1r);
		float d1rm1dot = dot(d1rm1,d1rm1);
		float RHS1 = (1.0 - 2.0*d1rdot + d1rm1dot)/(2*dt*dt);
		
		// second term:
		float3 d1v = (beads[b1].v - beads[b0].v)/dS;
		float RHS2 = dot(d1v,d1v);
		
		// third term:
		float3 d1f = (beads[b1].f - beads[b0].f)/dS;
		float3 d1rstar = (beads[b1].rstar - beads[b0].rstar)/dS;
		float RHS3 = dot(d1rstar,d1f);
		
		// assign values to RHS vector for tension:
		B[i] = RHS1 - RHS2 - RHS3;		
	}
}



// --------------------------------------------------------
// IBM3D kernel to compute tridiagonal [A] matrix of
// tension equation, done using the edges
// --------------------------------------------------------

__global__ void compute_tension_tridiag_fibers_IBM3D(
	beadfiber* beads,
	edgefiber* edges,
	float* Au,
	float* Ac,
	float* Al,
	float dS,
	float dt,
	int nEdges)
{	
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nEdges) {
		
		// left and right beads:
		int b0 = edges[i].b0;
		int b1 = edges[i].b1;
		
		// internal edges:
		if (edges[i].posID == 0) {
			float3 r10 = beads[b1].r - beads[b0].r;
			float r10_r10 = dot(r10,r10);
			
			// center diagonal term:
			Ac[i] = -2.0*r10_r10/(dS*dS*dS*dS);
			
			// upper diagonal term:
			int b2 = b1 + 1;  // bead to the right of b1
			float3 r21 = beads[b2].r - beads[b1].r;
			float r21_r10 = dot(r21,r10);
			Au[i] = r21_r10/(dS*dS*dS*dS);
			
			// lower diagonal term:
			int b0m1 = b0 - 1;  // bead to the left of b0
			float3 r00m1 = beads[b0].r - beads[b0m1].r;
			float r10_r00m1 = dot(r10,r00m1);
			Al[i] = r10_r00m1/(dS*dS*dS*dS);
		}
		
		// left-most edge:
		else if (edges[i].posID == 1) {
			float3 r10 = beads[b1].r - beads[b0].r;
			float r10_r10 = dot(r10,r10);
			
			// center diagonal term:
			Ac[i] = -3.0*r10_r10/(dS*dS*dS*dS);
			
			// upper diagonal term:
			int b2 = b1 + 1;  // bead to the right of b1
			float3 r21 = beads[b2].r - beads[b1].r;
			float r21_r10 = dot(r21,r10);
			Au[i] = r21_r10/(dS*dS*dS*dS);			
			
			// lower diagonal term:
			Al[i] = 0.0f;
		}
					
		// right-most edge:
		else if (edges[i].posID == 2) {
			float3 r10 = beads[b1].r - beads[b0].r;
			float r10_r10 = dot(r10,r10);
			
			// center diagonal term:
			Ac[i] = -3.0*r10_r10/(dS*dS*dS*dS);
			
			// upper diagonal term:
			Au[i] = 0.0f;
			
			// lower diagonal term:
			int b0m1 = b0 - 1;  // bead to the left of b0
			float3 r00m1 = beads[b0].r - beads[b0m1].r;
			float r10_r00m1 = dot(r10,r00m1);
			Al[i] = r10_r00m1/(dS*dS*dS*dS);
		}

	}
}



// --------------------------------------------------------
// IBM3D kernel to compute bead update matrices 
// --------------------------------------------------------

__global__ void compute_bead_update_matrices_fibers_IBM3D(
	beadfiber* beads,
	float* T,
	float* Bx,
	float* By,
	float* Bz,
	float* Au,
	float* Ac,
	float* Al,
	float dS,
	float dt,
	int nBeads)
{		
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nBeads) {
		
		// right-hand sides:
		Bx[i] = beads[i].rstar.x + dt*dt*(beads[i].f.x);
		By[i] = beads[i].rstar.y + dt*dt*(beads[i].f.y);
		Bz[i] = beads[i].rstar.z + dt*dt*(beads[i].f.z);
		
		// tridiagonal A-matrix:
		float dtdS2 = dt*dt/dS/dS;
			
		// internal bead:
		if (beads[i].posID == 0) {
			// e0 = left edge, e1 = right edge
			int e0 = i - beads[i].fiberID - 1;
			int e1 = i - beads[i].fiberID;
			
			// center diagonal term:
			Ac[i] = 1.0 + (T[e1] + T[e0])*dtdS2;
			
			// upper diagonal term:
			Au[i] = -T[e1]*dtdS2;
			
			// lower diagonal term:
			Al[i] = -T[e0]*dtdS2;
		}
				
		// left-most bead:
		else if (beads[i].posID == 1) {
			// right edge
			int e1 = i - beads[i].fiberID;
			
			// center diagonal term:
			Ac[i] = 1.0 + 2.0*T[e1]*dtdS2;
			
			// upper diagonal term:
			Au[i] = -2.0*T[e1]*dtdS2;
					
			// lower diagonal term:
			Al[i] = 0.0f;
		}
				
		// right-most bead:
		else if (beads[i].posID == 2) {
			// left edge
			int e0 = i - beads[i].fiberID - 1;
			
			// center diagonal term:			
			Ac[i] = 1.0 + 2.0*T[e0]*dtdS2;
			
			// upper diagonal term:
			Au[i] = 0.0f;
			
			// lower diagonal term:
			Al[i] = -2.0*T[e0]*dtdS2;
		}
		
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate hydrodynamic force between
// IBM fiber bead and LBM fluid
// --------------------------------------------------------

__global__ void hydrodynamic_force_bead_fluid_IBM3D(
	beadfiber* beads,
	float* fxLBM,
	float* fyLBM,
	float* fzLBM,
	float* uLBM,
	float* vLBM,
	float* wLBM,
	float mob,
	int Nx,
	int Ny,
	int Nz,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nBeads) {
				
		// --------------------------------------
		// find nearest LBM voxel (rounded down)
		// --------------------------------------
		
		int i0 = int(floor(beads[i].r.x));
		int j0 = int(floor(beads[i].r.y));
		int k0 = int(floor(beads[i].r.z));
		
		// --------------------------------------
		// loop over footprint to get 
		// interpolated LBM velocity:
		// --------------------------------------
				
		float vxLBMi = 0.0;
		float vyLBMi = 0.0;
		float vzLBMi = 0.0;		
		for (int kk=k0; kk<=k0+1; kk++) {
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {				
					int ndx = bead_fiber_voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = beads[i].r.x - float(ii);
					float ry = beads[i].r.y - float(jj);
					float rz = beads[i].r.z - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					vxLBMi += del*uLBM[ndx];
					vyLBMi += del*vLBM[ndx];
					vzLBMi += del*wLBM[ndx];				
				}
			}
		}
		
		// --------------------------------------
		// calculate hydrodynamic forces & add them
		// to IBM bead forces:
		// --------------------------------------
				
		float vfx = mob*(vxLBMi - beads[i].v.x);
		float vfy = mob*(vyLBMi - beads[i].v.y);
		float vfz = mob*(vzLBMi - beads[i].v.z);
		beads[i].f.x += vfx;
		beads[i].f.y += vfy;
		beads[i].f.z += vfz;
		
		// --------------------------------------
		// distribute the !negative! of the 
		// hydrodynamic bead force to the LBM
		// fluid (it is assumed that the thermal and
		// the propulsion forces have already been 
		// calculated):
		// --------------------------------------
		
		for (int kk=k0; kk<=k0+1; kk++) {
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {				
					int ndx = bead_fiber_voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = beads[i].r.x - float(ii);
					float ry = beads[i].r.y - float(jj);
					float rz = beads[i].r.z - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					atomicAdd(&fxLBM[ndx],-del*vfx);
					atomicAdd(&fyLBM[ndx],-del*vfy);
					atomicAdd(&fzLBM[ndx],-del*vfz);				
				}
			}
		}
				
	}	
}



// --------------------------------------------------------
// IBM3D kernel to determine 1D index from 3D indices:
// --------------------------------------------------------

__device__ inline int bead_fiber_voxel_ndx(
	int i,
	int j,
	int k,
	int Nx,
	int Ny,
	int Nz)
{
    if (i < 0) i += Nx;
    if (i >= Nx) i -= Nx;
    if (j < 0) j += Ny;
    if (j >= Ny) j -= Ny;
    if (k < 0) k += Nz;
    if (k >= Nz) k -= Nz;
    return k*Nx*Ny + j*Nx + i;	
}



// --------------------------------------------------------
// IBM3D kernel to unwrap bead coordinates.  Here, the
// beads of a filament are brought back close to the bead's 
// headBead.  This is done to avoid complications with
// PBCs:
// --------------------------------------------------------

__global__ void unwrap_bead_coordinates_IBM3D(
	beadfiber* beads,
	fiber* fibers,
	float3 Box,
	int3 pbcFlag,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		int f = beads[i].fiberID;
		int j = fibers[f].headBead;
		float3 rij = beads[j].r - beads[i].r;
		float3 adjust = roundf(rij/Box)*Box*pbcFlag;
		beads[i].r = beads[i].r +  adjust;    // PBC's
		beads[i].rm1 = beads[i].rm1 + adjust; // PBC's
	}
}



// --------------------------------------------------------
// IBM3D kernel to wrap bead coordinates for PBCs:
// --------------------------------------------------------

__global__ void wrap_bead_coordinates_IBM3D(
	beadfiber* beads,
	float3 Box,
	int3 pbcFlag,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		float3 adjust = floorf(beads[i].r/Box)*Box*pbcFlag;
		beads[i].r = beads[i].r - adjust;      // PBC's
		beads[i].rm1 = beads[i].rm1 - adjust;  // PBC's 
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void bead_wall_forces_ydir_IBM3D(
	beadfiber* beads,
	float3 Box,
	float repA,
	float repD,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		const float d = repD;
		const float A = repA;
		const float yi = beads[i].r.y;
		// bottom wall
		if (yi < d) {
			const float force = A/pow(yi,2) - A/pow(d,2);
			beads[i].f.y += force;
			if (yi < 0.0001) beads[i].r.y = 0.0001;
		}
		// top wall
		else if (yi > (Box.y-1.0)-d) {
			const float bmyi = (Box.y-1.0) - yi;
			const float force = A/pow(bmyi,2) - A/pow(d,2);
			beads[i].f.y -= force;
			if (yi > Box.y-1.0001) beads[i].r.y = Box.y-1.0001;
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void bead_wall_forces_zdir_IBM3D(
	beadfiber* beads,
	float3 Box,
	float repA,
	float repD,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		const float d = repD;
		const float A = repA;
		const float zi = beads[i].r.z;
		// bottom wall
		if (zi < d) {
			const float force = A/pow(zi,2) - A/pow(d,2);
			beads[i].f.z += force;
			if (zi < 0.0001) beads[i].r.z = 0.0001;
		}
		// top wall
		else if (zi > (Box.z-1.0)-d) {
			const float bmzi = (Box.z-1.0) - zi;
			const float force = A/pow(bmzi,2) - A/pow(d,2);
			beads[i].f.z -= force;
			if (zi > Box.z-1.0001) beads[i].r.z = Box.z-1.0001;
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void bead_wall_forces_ydir_zdir_IBM3D(
	beadfiber* beads,
	float3 Box,
	float repA,
	float repD,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		const float d = repD;
		const float A = repA;
		const float yi = beads[i].r.y;
		const float zi = beads[i].r.z;
		// bottom wall
		if (yi < d) {
			const float force = A/pow(yi,2) - A/pow(d,2);
			beads[i].f.y += force;
			if (yi < 0.0001) beads[i].r.y = 0.0001;
		}
		// top wall
		else if (yi > (Box.y-1.0)-d) {
			const float bmyi = (Box.y-1.0) - yi;
			const float force = A/pow(bmyi,2) - A/pow(d,2);
			beads[i].f.y -= force;
			if (yi > Box.y-1.0001) beads[i].r.y = Box.y-1.0001;
		}
		// back wall
		if (zi < d) {
			const float force = A/pow(zi,2) - A/pow(d,2);
			beads[i].f.z += force;
			if (zi < 0.0001) beads[i].r.z = 0.0001;
		}
		// front wall
		else if (zi > (Box.z-1.0)-d) {
			const float bmzi = (Box.z-1.0) - zi;
			const float force = A/pow(bmzi,2) - A/pow(d,2);
			beads[i].f.z -= force;
			if (zi > Box.z-1.0001) beads[i].r.z = Box.z-1.0001;
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to build the binMap array:
// --------------------------------------------------------

__global__ void build_binMap_for_beads_fibers_IBM3D(
	bindata bins)
{
	// define bin:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < bins.nBins) {
	
		// -------------------------------
		// calculate bin's x,y,z coordinates:
		// -------------------------------
				
		int binx = i/(bins.numBins.y*bins.numBins.z);
		int biny = (i/bins.numBins.z)%bins.numBins.y;
		int binz = i%bins.numBins.z;
		
		// -------------------------------
		// determine neighboring bins:
		// -------------------------------
		
		int cnt = 0;
		int offst = i*bins.nnbins;
		
		for (int bx = binx-1; bx < binx+2; bx++) {
			for (int by = biny-1; by < biny+2; by++) {
				for (int bz = binz-1; bz < binz+2; bz++) {
					// do not include current bin
					if (bx==binx && by==biny && bz==binz) continue;
					// bin index of neighbor
					bins.binMap[offst+cnt] = bin_index_for_beads_fibers(bx,by,bz,bins.numBins);
					// update counter
					cnt++;
				}
			}
		}		
		
	}	
}



// --------------------------------------------------------
// IBM3D kernel to reset bin arrays:
// --------------------------------------------------------

__global__ void reset_bin_lists_for_beads_fibers_IBM3D(
	bindata bins)
{
	// define bin:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < bins.nBins) {
		
		// -------------------------------
		// reset binOccupancy[] to zero,
		// and binMembers[] array to -1:
		// -------------------------------
		
		bins.binOccupancy[i] = 0;
		int offst = i*bins.binMax;
		for (int k=offst; k<offst+bins.binMax; k++) {
			bins.binMembers[k] = -1;
		}
		
	}	
}



// --------------------------------------------------------
// IBM3D kernel to assign beads to bins:
// --------------------------------------------------------

__global__ void build_bin_lists_for_beads_fibers_IBM3D(
	beadfiber* beads,
	bindata bins,
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {		
		
		// -------------------------------
		// calculate bin ID:
		// -------------------------------
		
		int binID = int(floor(beads[i].r.x/bins.sizeBins))*bins.numBins.z*bins.numBins.y +  
			        int(floor(beads[i].r.y/bins.sizeBins))*bins.numBins.z +
		            int(floor(beads[i].r.z/bins.sizeBins));		
						
		// -------------------------------
		// update the lists:
		// -------------------------------
		
		if (binID >= 0 && binID < bins.nBins) {
			atomicAdd(&bins.binOccupancy[binID],1);
			int offst = binID*bins.binMax;
			for (int k=offst; k<offst+bins.binMax; k++) {
				int flag = atomicCAS(&bins.binMembers[k],-1,i); 
				if (flag == -1) break;  
			}
		}
		
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate nonbonded bead interactions
// using the bin lists:
// --------------------------------------------------------

__global__ void nonbonded_bead_interactions_IBM3D(
	beadfiber* beads,
	bindata bins,
	float repA,
	float repD,
	int nBeads,
	float3 Box,	
	int3 pbcFlag)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {		
		
		// -------------------------------
		// calculate bin ID:
		// -------------------------------
		
		int binID = int(floor(beads[i].r.x/bins.sizeBins))*bins.numBins.z*bins.numBins.y +  
			        int(floor(beads[i].r.y/bins.sizeBins))*bins.numBins.z +
		            int(floor(beads[i].r.z/bins.sizeBins));		
		
		// -------------------------------
		// loop over beads in the same bin:
		// -------------------------------
				
		int offst = binID*bins.binMax;
		int occup = bins.binOccupancy[binID];
		if (occup > bins.binMax) {
			printf("occup = %i, binID = %i \n", occup, binID);
			occup = bins.binMax;
		}
								
		for (int k=offst; k<offst+occup; k++) {
			int j = bins.binMembers[k];
			if (i==j) continue;
			if (beads[i].fiberID == beads[j].fiberID) continue;
			//pairwise_bead_interaction_forces_WCA(i,j,repA,repD,beads,Box,pbcFlag);
			pairwise_bead_interaction_forces(i,j,repA,repD,beads,Box,pbcFlag);		
		}
		
		// -------------------------------
		// loop over neighboring bins:
		// -------------------------------
		
        for (int b=0; b<bins.nnbins; b++) {
            // get neighboring bin ID
			int naborbinID = bins.binMap[binID*bins.nnbins + b];
			offst = naborbinID*bins.binMax;
			occup = bins.binOccupancy[naborbinID];
			if (occup > bins.binMax) occup = bins.binMax;
			// loop over beads in this bin:
			for (int k=offst; k<offst+occup; k++) {
				int j = bins.binMembers[k];
				if (beads[i].fiberID == beads[j].fiberID) continue;				
				//pairwise_bead_interaction_forces_WCA(i,j,repA,repD,beads,Box,pbcFlag);
				pairwise_bead_interaction_forces(i,j,repA,repD,beads,Box,pbcFlag);		
			}
		}
				
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate i-j force:
// --------------------------------------------------------

__device__ inline void pairwise_bead_interaction_forces(
	const int i, 
	const int j,
	const float repA,
	const float repD,
	beadfiber* beads,
	float3 Box,
	int3 pbcFlag)
{
	float3 rij = beads[i].r - beads[j].r;
	rij -= roundf(rij/Box)*Box*pbcFlag;  // PBC's	
	const float r = length(rij);
	if (r < repD) {
		// linear force repulsion
		float force = repA - (repA/repD)*r;
		float3 fij = force*(rij/r);
		beads[i].f += fij;
	} 	
}



// --------------------------------------------------------
// IBM3D kernel to calculate i-j force:
// Weeks-Chandler-Anderson potential
// --------------------------------------------------------

__device__ inline void pairwise_bead_interaction_forces_WCA(
	const int i, 
	const int j,
	const float repA,
	const float repD,
	beadfiber* beads,
	float3 Box,
	int3 pbcFlag)
{
	float3 rij = beads[i].r - beads[j].r;
	rij -= roundf(rij/Box)*Box*pbcFlag;  // PBC's	
	const float r = length(rij);
	if (r < repD) {
		float sig = 0.8909*repD;  // this ensures F=0 is at cutoff
		float eps = 0.001;
		float sigor = sig/r;
		float sigor6 = sigor*sigor*sigor*sigor*sigor*sigor;
		float sigor12 = sigor6*sigor6;
		float force = 24.0*eps*(2*sigor12 - sigor6)/r/r;
		beads[i].f += force*rij;
	} 	
}



// --------------------------------------------------------
// IBM3D kernel to calculate i-j force:
// --------------------------------------------------------

__device__ inline int bin_index_for_beads_fibers(
	int i, 
	int j,
	int k, 
	const int3 size)
{
    if (i < 0) i += size.x;
    if (i >= size.x) i -= size.x;
    if (j < 0) j += size.y;
    if (j >= size.y) j -= size.y;
    if (k < 0) k += size.z;
    if (k >= size.z) k -= size.z;
    return i*size.z*size.y + j*size.z + k;
}








