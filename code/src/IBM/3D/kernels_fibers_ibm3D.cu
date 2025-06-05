
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
	fiber* filams,
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

__global__ void compute_bead_update_matrices_IBM3D(
	beadfiber* beads,
	fiber* filams,
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
			Au[i] = -2.0*T[e1]*dtdS2;
			
			// lower diagonal term:
			Al[i] = -2.0*T[e0]*dtdS2;
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





