
# include "kernels_scsp_active_D2Q9.cuh"
# include <stdio.h>



// --------------------------------------------------------
// D2Q9 kernel to re-set the fluid forces to zero: 
// --------------------------------------------------------

__global__ void scsp_active_zero_forces_D2Q9(
	float2* F,
	int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {			
		F[i].x = 0.0;
		F[i].y = 0.0;
	}
}



// --------------------------------------------------------
// D2Q9 initialize kernel: 
// --------------------------------------------------------

__global__ void scsp_active_initial_equilibrium_D2Q9(
	float* f1,
	float* r,
	float2* u,
	int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {			
		// useful constants: 
		const float rho = r[i];
		const float ux = u[i].x;
		const float vy = u[i].y;
		const float w0r = rho*4.0/9.0;
		const float wsr = rho*1.0/9.0;
		const float wdr = rho*1.0/36.0;
		const float omusq = 1.0 - 1.5*(ux*ux + vy*vy);	
		const float tux = 3.0*ux;
		const float tvy = 3.0*vy;			
		// equilibrium populations:
		f1[9*i+0] = w0r*(omusq);		
		float cidot3u = tux;
		f1[9*i+1] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy;
		f1[9*i+2] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -tux;
		f1[9*i+3] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -tvy;
		f1[9*i+4] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));	
		cidot3u = tux+tvy;
		f1[9*i+5] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy-tux;
		f1[9*i+6] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -(tux+tvy);
		f1[9*i+7] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tux-tvy;
		f1[9*i+8] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	}
}



// --------------------------------------------------------
// D2Q9 update kernel.
// This algorithm is based on the optimized "stream-collide-
// save" algorithm recommended by T. Kruger in the 
// textbook: "The Lattice Boltzmann Method: Principles
// and Practice".
// --------------------------------------------------------

__global__ void scsp_active_stream_collide_save_D2Q9(
	float* f1,
    float* f2,
	float* r,
	float2* u,
	int* streamIndex,
	int* voxelType,
	float nu,
	int nVoxels)
{

	// -----------------------------------------------
	// define voxel:
	// -----------------------------------------------
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
		
		// --------------------------------------------------		
		// voxel-specific parameters:
		// --------------------------------------------------
		
		int offst = 9*i;	
		float ft[9];
		
		// --------------------------------------------------		
		// STREAMING - load populations from adjacent voxels,
		//             note	that streamIndex[] accounts for
		//             halfway bounceback conditions.
		// --------------------------------------------------
		
		ft[0] = f1[streamIndex[offst+0]];                   
		ft[1] = f1[streamIndex[offst+1]]; 
		ft[2] = f1[streamIndex[offst+2]];  
		ft[3] = f1[streamIndex[offst+3]];  
		ft[4] = f1[streamIndex[offst+4]];  
		ft[5] = f1[streamIndex[offst+5]]; 
		ft[6] = f1[streamIndex[offst+6]];  
		ft[7] = f1[streamIndex[offst+7]];  
		ft[8] = f1[streamIndex[offst+8]]; 		
						
		// --------------------------------------------------
		// MACROS - calculate the velocity and density.
		// --------------------------------------------------
		
		float rho = ft[0] + ft[1] + ft[2] + ft[3] + ft[4] + ft[5] + ft[6] + ft[7] + ft[8];
		float rhoinv = 1.0/rho;
		float ux = rhoinv*(ft[1] + ft[5] + ft[8] - (ft[3] + ft[6] + ft[7]));
		float vy = rhoinv*(ft[2] + ft[5] + ft[6] - (ft[4] + ft[7] + ft[8]));
		
		// --------------------------------------------------
		// COLLISION - perform the BGK collision operator.
		// --------------------------------------------------
			
		// useful constants:
		const float tauinv = 2.0/(6.0*nu + 1.0);   // 1/tau
		const float omtauinv = 1.0 - tauinv;       // 1 - 1/tau
		const float tw0r = (4.0/9.0)*rho*tauinv;   // w[0]*rho/tau
		const float twsr = (1.0/9.0)*rho*tauinv;   // w[1-4]*rho/tau
		const float twdr = (1.0/36.0)*rho*tauinv;  // w[5-8]*rho/tau
		const float omusq = 1.0 - 1.5*(ux*ux + vy*vy);
		const float tux = 3.0*ux;
		const float tvy = 3.0*vy;	
		
		// collision calculations:	
		f2[offst+0] = omtauinv*ft[0] + tw0r*(omusq);		                      // dir-0
		float cidot3u = tux;
		f2[offst+1] = omtauinv*ft[1] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-1
		cidot3u = tvy;
		f2[offst+2] = omtauinv*ft[2] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-2
		cidot3u = -tux;
		f2[offst+3] = omtauinv*ft[3] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-3
		cidot3u = -tvy;
		f2[offst+4] = omtauinv*ft[4] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-4
		cidot3u = tux+tvy;
		f2[offst+5] = omtauinv*ft[5] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-5
		cidot3u = tvy-tux;
		f2[offst+6] = omtauinv*ft[6] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-6
		cidot3u = -(tux+tvy);
		f2[offst+7] = omtauinv*ft[7] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-7
		cidot3u = tux-tvy;
		f2[offst+8] = omtauinv*ft[8] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-8			
	
		// --------------------------------------------------		
		// SAVE - write macros to arrays 
		// --------------------------------------------------
		
		r[i] = rho;
		u[i].x = ux;
		u[i].y = vy;
							
	}
}



// --------------------------------------------------------
// D2Q9 update kernel.
// This algorithm is based on the optimized "stream-collide-
// save" algorithm recommended by T. Kruger in the 
// textbook: "The Lattice Boltzmann Method: Principles
// and Practice".
// --------------------------------------------------------

__global__ void scsp_active_stream_collide_save_forcing_D2Q9(
	float* f1,
    float* f2,
	float* r,
	float2* u,
	float2* F,
	int* streamIndex,
	int* voxelType,
	float nu,
	int nVoxels)
{

	// -----------------------------------------------
	// define voxel:
	// -----------------------------------------------
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
		
		// --------------------------------------------------		
		// voxel-specific parameters:
		// --------------------------------------------------
		
		int offst = 9*i;	
		float ft[9];
		
		// --------------------------------------------------		
		// STREAMING - load populations from adjacent voxels,
		//             note	that streamIndex[] accounts for
		//             halfway bounceback conditions.
		// --------------------------------------------------
		
		ft[0] = f1[streamIndex[offst+0]];                   
		ft[1] = f1[streamIndex[offst+1]]; 
		ft[2] = f1[streamIndex[offst+2]];  
		ft[3] = f1[streamIndex[offst+3]];  
		ft[4] = f1[streamIndex[offst+4]];  
		ft[5] = f1[streamIndex[offst+5]]; 
		ft[6] = f1[streamIndex[offst+6]];  
		ft[7] = f1[streamIndex[offst+7]];  
		ft[8] = f1[streamIndex[offst+8]]; 	
				
		// --------------------------------------------------
		// MACROS - calculate the velocity and density (force
		//          corrected).
		// --------------------------------------------------	
				
		float rho = ft[0] + ft[1] + ft[2] + ft[3] + ft[4] + ft[5] + ft[6] + ft[7] + ft[8];
		float rhoinv = 1.0/rho;
		float ux = rhoinv*(ft[1] + ft[5] + ft[8] - (ft[3] + ft[6] + ft[7]) + 0.5*F[i].x);
		float vy = rhoinv*(ft[2] + ft[5] + ft[6] - (ft[4] + ft[7] + ft[8]) + 0.5*F[i].y);
		
		// --------------------------------------------------
		// COLLISION - perform the BGK collision operator
		//             with Guo forcing.
		// --------------------------------------------------
					
		// useful constants:
		const float w0 = 4.0/9.0;
		const float ws = 1.0/9.0;
		const float wd = 1.0/36.0;			
		const float omega = 2.0/(6.0*nu + 1.0);   // 1/tau
		const float omomega = 1.0 - omega;        // 1 - 1/tau
		const float omomega2 = 1.0 - 0.5*omega;   // 1 - 1/(2tau)
		const float omusq = 1.0 - 1.5*(ux*ux + vy*vy);
					
		// direction 0
		float evel = 0.0;       // e dot velocity
		float emiu = 0.0-ux;    // e minus u
		float emiv = 0.0-vy;    // e minus v
		float feq = w0*rho*omusq;
		float frc = w0*(F[i].x*(3.0*emiu) + F[i].y*(3.0*emiv));
		f2[offst+0] = omomega*ft[0] + omega*feq + omomega2*frc;
		
		// direction 1
		evel = ux;
		emiu = 1.0-ux;
		emiv = 0.0-vy;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(F[i].x*(3.0*emiu + 9.0*evel) + F[i].y*(3.0*emiv));
		f2[offst+1] = omomega*ft[1] + omega*feq + omomega2*frc;
		
		// direction 2
		evel = vy; 
		emiu = 0.0-ux;
		emiv = 1.0-vy;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(F[i].x*(3.0*emiu) + F[i].y*(3.0*emiv + 9.0*evel));
		f2[offst+2] = omomega*ft[2] + omega*feq + omomega2*frc;
		
		// direction 3
		evel = -ux;
		emiu = -1.0-ux;
		emiv =  0.0-vy;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(F[i].x*(3.0*emiu - 9.0*evel) + F[i].y*(3.0*emiv));
		f2[offst+3] = omomega*ft[3] + omega*feq + omomega2*frc;
		
		// direction 4
		evel = -vy;
		emiu =  0.0-ux;
		emiv = -1.0-vy;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(F[i].x*(3.0*emiu) + F[i].y*(3.0*emiv - 9.0*evel));
		f2[offst+4] = omomega*ft[4] + omega*feq + omomega2*frc;
		
		// direction 5
		evel = ux + vy;
		emiu = 1.0-ux;
		emiv = 1.0-vy;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(F[i].x*(3.0*emiu + 9.0*evel) + F[i].y*(3.0*emiv + 9.0*evel));
		f2[offst+5] = omomega*ft[5] + omega*feq + omomega2*frc;
		
		// direction 6
		evel = -ux + vy;
		emiu = -1.0-ux;
		emiv =  1.0-vy;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(F[i].x*(3.0*emiu - 9.0*evel) + F[i].y*(3.0*emiv + 9.0*evel));
		f2[offst+6] = omomega*ft[6] + omega*feq + omomega2*frc;
		
		// direction 7
		evel = -ux - vy;
		emiu = -1.0-ux;
		emiv = -1.0-vy;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(F[i].x*(3.0*emiu - 9.0*evel) + F[i].y*(3.0*emiv - 9.0*evel));
		f2[offst+7] = omomega*ft[7] + omega*feq + omomega2*frc;
		
		// direction 8
		evel = ux - vy;
		emiu =  1.0-ux;
		emiv = -1.0-vy;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(F[i].x*(3.0*emiu + 9.0*evel) + F[i].y*(3.0*emiv - 9.0*evel));
		f2[offst+8] = omomega*ft[8] + omega*feq + omomega2*frc;		
	
		// --------------------------------------------------		
		// SAVE - write macros to arrays 
		// --------------------------------------------------
		
		r[i] = rho;
		u[i].x = ux;
		u[i].y = vy;
					
	}
}



// --------------------------------------------------------
// D2Q9 update kernel for the orientation field.
// See: Tjhung et al. Soft Matter (2011) 7:7453
// --------------------------------------------------------

__global__ void scsp_active_update_orientation_D2Q9(
	float2* u,
	float2* p,
	int* nList,
	float sf,
	float fricR,
	int nVoxels)
{
	
	// -----------------------------------------------
	// define voxel:
	// -----------------------------------------------
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
		
		int offst = 9*i;	
		
		// calculate gradient of velocity field		
		tensor2D W;
		W.xx = (u[nList[offst+1]].x - u[nList[offst+3]].x) / 2.0;  // assume dx=1
		W.xy = (u[nList[offst+2]].x - u[nList[offst+4]].x) / 2.0;  // assume dx=1
		W.yx = (u[nList[offst+1]].y - u[nList[offst+3]].y) / 2.0;  // assume dx=1
		W.yy = (u[nList[offst+2]].y - u[nList[offst+4]].y) / 2.0;  // assume dx=1
				
		// calculate symmetric and anti-symmetric flow field contribution to dpdt:
		tensor2D D = 0.5*(W + transpose(W));
		tensor2D O = 0.5*(W - transpose(W));
		float2 dpdt1 = (sf*D - O)*p[i];
		
		// advection contribution to dpdt (see Wikipedia page on 'material derivative'):
		float px  = p[i].x;
		float py  = p[i].y;
		float pxE = p[nList[offst+1]].x;  // east
		float pxN = p[nList[offst+2]].x;  // north
		float pxW = p[nList[offst+3]].x;  // west
		float pxS = p[nList[offst+4]].x;  // south
		float pyE = p[nList[offst+1]].y;  // east
		float pyN = p[nList[offst+2]].y;  // north
		float pyW = p[nList[offst+3]].y;  // west
		float pyS = p[nList[offst+4]].y;  // south		
		float dpxdx = (pxE - pxW) / 2.0;  // assume dx=1
		float dpxdy = (pxN - pxS) / 2.0;  // assume dx=1
		float dpydx = (pyE - pyW) / 2.0;  // assume dx=1
		float dpydy = (pyN - pyS) / 2.0;  // assume dx=1
		float2 dpdt2 = make_float2(u[i].x*dpxdx + u[i].y*dpxdy, u[i].x*dpydx + u[i].y*dpydy);
		dpdt2 *= -1.0f;
		
		// molecular field contribution to dpdt:
		float pmag = sqrt(px*px + py*py);
		float dfdpmag = pmag*pmag*pmag - pmag;
		float laplpx = (pxE + pxW + pxN + pxS - 4.0*px);   // assume dx=1
		float laplpy = (pyE + pyW + pyN + pyS - 4.0*py);   // assume dx=1;
		float2 dpdt3 = make_float2(dfdpmag*px - laplpx, dfdpmag*py - laplpy)/fricR;
		dpdt3 *= -1.0f;
		
		// update orientation field:
		p[i] += dpdt1 + dpdt2 + dpdt3;   // assume dt=1.0
		
	}
}



// --------------------------------------------------------
// Kernel to calculate the active stress to be applied
// to the fluid.
// --------------------------------------------------------

__global__ void scsp_active_fluid_stress_D2Q9(
	float2* p,
	tensor2D* stress,
	float activity,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {		
		// calculate active stress tensor:
		stress[i] = -activity*dyadic(p[i]);
	}
}



// --------------------------------------------------------
// Kernel to calculate the active forces to be applied
// to the fluid.
// --------------------------------------------------------

__global__ void scsp_active_fluid_forces_D2Q9(
	float2* F,
	tensor2D* stress,
	int* nList,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {
		int offst = 9*i;	
		// divergence of stress tensor:
		float dsxxdx = (stress[nList[offst+1]].xx - stress[nList[offst+3]].xx) / 2.0;  // assume dx=1
		float dsxydy = (stress[nList[offst+2]].xy - stress[nList[offst+4]].xy) / 2.0;  // assume dx=1
		float dsyxdx = (stress[nList[offst+1]].yx - stress[nList[offst+3]].yx) / 2.0;  // assume dx=1
		float dsyydy = (stress[nList[offst+2]].yy - stress[nList[offst+4]].yy) / 2.0;  // assume dx=1
		float2 force = make_float2(dsxxdx + dsxydy, dsyxdx + dsyydy);
		F[i] += force;
	}
	
}


