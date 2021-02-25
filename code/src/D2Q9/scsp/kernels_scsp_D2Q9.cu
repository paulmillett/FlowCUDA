
# include "kernels_scsp_D2Q9.cuh"
# include "../iolets/zou_he_BC_D2Q9.cuh"
# include <stdio.h>



// --------------------------------------------------------
// D2Q9 kernel to re-set the fluid forces to zero: 
// --------------------------------------------------------

__global__ void scsp_zero_forces_D2Q9(
	float* fx,
	float* fy,
	int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {			
		fx[i] = 0.0;
		fy[i] = 0.0;
	}
}



// --------------------------------------------------------
// D2Q9 kernel to re-set the fluid forces (and extrapolated
// IB velocities and weights) to zero:
// --------------------------------------------------------

__global__ void scsp_zero_forces_D2Q9(
	float* fx,
	float* fy,
	float* uIBvox,
	float* vIBvox,
	float* weights,
	int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {			
		fx[i] = 0.0;
		fy[i] = 0.0;
		uIBvox[i] = 0.0;
		vIBvox[i] = 0.0;
		weights[i] = 0.0;
	}
}



// --------------------------------------------------------
// D2Q9 initialize kernel: 
// --------------------------------------------------------

__global__ void scsp_initial_equilibrium_D2Q9(float* f1,
										      float* r,
										      float* u,
										      float* v,
										      int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {			
		// useful constants: 
		const float rho = r[i];
		const float ux = u[i];
		const float vy = v[i];
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
// D2Q9 kernel to calculate the necessary force to 
// adjust the velocity from (u,v) to (uIBvox,vIBvox): 
// --------------------------------------------------------

__global__ void scsp_force_velocity_match_D2Q9(
	float* fx,
	float* fy,
	float* u,
	float* v,
	float* uIBvox,
	float* vIBvox,
	float* weights,
	float* rho,		
	int nVoxels)
{
	
	// -----------------------------------------------
	// define voxel:
	// -----------------------------------------------
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	
	if (i < nVoxels) {			
		
		// --------------------------------------------------		
		// Only do this if weights[i] is > 0:
		// --------------------------------------------------
		
		if (weights[i] > 0.0) {			
			uIBvox[i] /= weights[i];  // weighted average
			vIBvox[i] /= weights[i];  // "              "		
			fx[i] = (uIBvox[i] - u[i])*2.0*rho[i];
			fy[i] = (vIBvox[i] - v[i])*2.0*rho[i];			
		}
	}
}



// --------------------------------------------------------
// D2Q9 update kernel.
// This algorithm is based on the optimized "stream-collide-
// save" algorithm recommended by T. Kruger in the 
// textbook: "The Lattice Boltzmann Method: Principles
// and Practice".
// --------------------------------------------------------

__global__ void scsp_stream_collide_save_D2Q9(
	float* f1,
    float* f2,
	float* r,
	float* u,
	float* v,
	int* streamIndex,
	int* voxelType,
	iolet2D* iolets,
	float nu,
	int nVoxels,
	bool save)
{

	// -----------------------------------------------
	// define voxel:
	// -----------------------------------------------
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
		
		// --------------------------------------------------		
		// voxel-specific parameters:
		// --------------------------------------------------
		
		int vtype = voxelType[i];
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
		// INLETS/OUTLETS - correct the ft[] values at inlets
		//                  and outlets using Zou-He BC's.
		// --------------------------------------------------
		
		if (vtype > 0) {
			zou_he_BC_D2Q9(vtype,ft,iolets);
		}
				
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
		
		if (save) {
			r[i] = rho;
			u[i] = ux;
			v[i] = vy;
		}
							
	}
}



// --------------------------------------------------------
// D2Q9 update kernel.
// This algorithm is based on the optimized "stream-collide-
// save" algorithm recommended by T. Kruger in the 
// textbook: "The Lattice Boltzmann Method: Principles
// and Practice".
// --------------------------------------------------------

__global__ void scsp_stream_collide_save_forcing_D2Q9(
	float* f1,
    float* f2,
	float* r,
	float* u,
	float* v,
	float* Fx,
	float* Fy,
	int* streamIndex,
	int* voxelType,
	iolet2D* iolets,
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
		
		int vtype = voxelType[i];
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
		// INLETS/OUTLETS - correct the ft[] values at inlets
		//                  and outlets using Zou-He BC's.
		// --------------------------------------------------
		
		if (vtype > 0) {
			zou_he_BC_D2Q9(vtype,ft,iolets);
		}
		
		// --------------------------------------------------
		// MACROS - calculate the velocity and density (force
		//          corrected).
		// --------------------------------------------------	
				
		float rho = ft[0] + ft[1] + ft[2] + ft[3] + ft[4] + ft[5] + ft[6] + ft[7] + ft[8];
		float rhoinv = 1.0/rho;
		float ux = rhoinv*(ft[1] + ft[5] + ft[8] - (ft[3] + ft[6] + ft[7]) + 0.5*Fx[i]);
		float vy = rhoinv*(ft[2] + ft[5] + ft[6] - (ft[4] + ft[7] + ft[8]) + 0.5*Fy[i]);
		
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
		float frc = w0*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv));
		f2[offst+0] = omomega*ft[0] + omega*feq + omomega2*frc;
		
		// direction 1
		evel = ux;
		emiu = 1.0-ux;
		emiv = 0.0-vy;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv));
		f2[offst+1] = omomega*ft[1] + omega*feq + omomega2*frc;
		
		// direction 2
		evel = vy; 
		emiu = 0.0-ux;
		emiv = 1.0-vy;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv + 9.0*evel));
		f2[offst+2] = omomega*ft[2] + omega*feq + omomega2*frc;
		
		// direction 3
		evel = -ux;
		emiu = -1.0-ux;
		emiv =  0.0-vy;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv));
		f2[offst+3] = omomega*ft[3] + omega*feq + omomega2*frc;
		
		// direction 4
		evel = -vy;
		emiu =  0.0-ux;
		emiv = -1.0-vy;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv - 9.0*evel));
		f2[offst+4] = omomega*ft[4] + omega*feq + omomega2*frc;
		
		// direction 5
		evel = ux + vy;
		emiu = 1.0-ux;
		emiv = 1.0-vy;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv + 9.0*evel));
		f2[offst+5] = omomega*ft[5] + omega*feq + omomega2*frc;
		
		// direction 6
		evel = -ux + vy;
		emiu = -1.0-ux;
		emiv =  1.0-vy;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv + 9.0*evel));
		f2[offst+6] = omomega*ft[6] + omega*feq + omomega2*frc;
		
		// direction 7
		evel = -ux - vy;
		emiu = -1.0-ux;
		emiv = -1.0-vy;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv - 9.0*evel));
		f2[offst+7] = omomega*ft[7] + omega*feq + omomega2*frc;
		
		// direction 8
		evel = ux - vy;
		emiu =  1.0-ux;
		emiv = -1.0-vy;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv - 9.0*evel));
		f2[offst+8] = omomega*ft[8] + omega*feq + omomega2*frc;		
	
		// --------------------------------------------------		
		// SAVE - write macros to arrays 
		// --------------------------------------------------
		
		r[i] = rho;
		u[i] = ux;
		v[i] = vy;
					
	}
}



// --------------------------------------------------------
// D2Q9 update kernel.
// This algorithm is based on the optimized "stream-collide-
// save" algorithm recommended by T. Kruger in the 
// textbook: "The Lattice Boltzmann Method: Principles
// and Practice".
// --------------------------------------------------------

__global__ void scsp_stream_collide_save_IBforcing_D2Q9(
	float* f1,
    float* f2,
	float* r,
	float* u,
	float* v,
	float* uIBvox,
	float* vIBvox,
	float* weights,
	int* streamIndex,
	int* voxelType,
	iolet2D* iolets,
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
		
		int vtype = voxelType[i];
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
		// INLETS/OUTLETS - correct the ft[] values at inlets
		//                  and outlets using Zou-He BC's.
		// --------------------------------------------------
		
		if (vtype > 0) {
			zou_he_BC_D2Q9(vtype,ft,iolets);
		}
		
		// --------------------------------------------------
		// MACROS - calculate the velocity and density.
		// --------------------------------------------------	
				
		float rho = ft[0] + ft[1] + ft[2] + ft[3] + ft[4] + ft[5] + ft[6] + ft[7] + ft[8];
		float rhoinv = 1.0/rho;
		float ux = rhoinv*(ft[1] + ft[5] + ft[8] - (ft[3] + ft[6] + ft[7]));
		float vy = rhoinv*(ft[2] + ft[5] + ft[6] - (ft[4] + ft[7] + ft[8]));
		
		// --------------------------------------------------
		// IB FORCE CORRECTION - calculate and add the force
		//                       needed to match the fluid
		//                       velocity with the IB velocity
		// --------------------------------------------------	
		
		float Fx = 0.0;
		float Fy = 0.0;
		if (weights[i] > 0.0){
			float uxIB = uIBvox[i]/weights[i];  // weighted average
			float vyIB = vIBvox[i]/weights[i];  // "              "		
			Fx = (uxIB - ux)*2.0*rho;
			Fy = (vyIB - vy)*2.0*rho;
			ux += 0.5*Fx*rhoinv;
			vy += 0.5*Fy*rhoinv;
		}		
		
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
		float frc = w0*(Fx*(3.0*emiu) + Fy*(3.0*emiv));
		f2[offst+0] = omomega*ft[0] + omega*feq + omomega2*frc;
		
		// direction 1
		evel = ux;
		emiu = 1.0-ux;
		emiv = 0.0-vy;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx*(3.0*emiu + 9.0*evel) + Fy*(3.0*emiv));
		f2[offst+1] = omomega*ft[1] + omega*feq + omomega2*frc;
		
		// direction 2
		evel = vy; 
		emiu = 0.0-ux;
		emiv = 1.0-vy;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx*(3.0*emiu) + Fy*(3.0*emiv + 9.0*evel));
		f2[offst+2] = omomega*ft[2] + omega*feq + omomega2*frc;
		
		// direction 3
		evel = -ux;
		emiu = -1.0-ux;
		emiv =  0.0-vy;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx*(3.0*emiu - 9.0*evel) + Fy*(3.0*emiv));
		f2[offst+3] = omomega*ft[3] + omega*feq + omomega2*frc;
		
		// direction 4
		evel = -vy;
		emiu =  0.0-ux;
		emiv = -1.0-vy;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx*(3.0*emiu) + Fy*(3.0*emiv - 9.0*evel));
		f2[offst+4] = omomega*ft[4] + omega*feq + omomega2*frc;
		
		// direction 5
		evel = ux + vy;
		emiu = 1.0-ux;
		emiv = 1.0-vy;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu + 9.0*evel) + Fy*(3.0*emiv + 9.0*evel));
		f2[offst+5] = omomega*ft[5] + omega*feq + omomega2*frc;
		
		// direction 6
		evel = -ux + vy;
		emiu = -1.0-ux;
		emiv =  1.0-vy;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu - 9.0*evel) + Fy*(3.0*emiv + 9.0*evel));
		f2[offst+6] = omomega*ft[6] + omega*feq + omomega2*frc;
		
		// direction 7
		evel = -ux - vy;
		emiu = -1.0-ux;
		emiv = -1.0-vy;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu - 9.0*evel) + Fy*(3.0*emiv - 9.0*evel));
		f2[offst+7] = omomega*ft[7] + omega*feq + omomega2*frc;
		
		// direction 8
		evel = ux - vy;
		emiu =  1.0-ux;
		emiv = -1.0-vy;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu + 9.0*evel) + Fy*(3.0*emiv - 9.0*evel));
		f2[offst+8] = omomega*ft[8] + omega*feq + omomega2*frc;
			
		// --------------------------------------------------		
		// SAVE - write macros to arrays 
		// --------------------------------------------------
		
		r[i] = rho;
		u[i] = ux;
		v[i] = vy;
					
	}
}

