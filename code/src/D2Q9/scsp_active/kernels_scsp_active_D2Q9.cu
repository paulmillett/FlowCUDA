
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
	float2* h,
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
		
		// calculate gradient of velocity field (at the end, enforce tracelessness)
		tensor2D W = grad_vector_field_D2Q9(i,u,nList);
		float tr = (1.0/2.0)*(W.xx + W.yy);
		W.xx -= tr;
		W.yy -= tr;
			
		// calculate symmetric and anti-symmetric flow field contribution to dpdt:
		tensor2D D = 0.5*(W + transpose(W));
		tensor2D O = 0.5*(W - transpose(W));
		float2 dpdt1 = -sf*D*p[i] + O*p[i];
		
		// advection contribution to dpdt (see Wikipedia page on 'material derivative'):
		tensor2D dp = grad_vector_field_D2Q9(i,p,nList);
		float2 dpdt2 = make_float2(u[i].x*dp.xx + u[i].y*dp.xy, u[i].x*dp.yx + u[i].y*dp.yy);
		dpdt2 *= -1.0f;
		
		// molecular field contribution to dpdt:
		float2 dpdt3 = -h[i]/fricR;
				
		// update orientation field (assuming dt=1.0):
		p[i] += dpdt1 + dpdt2 + dpdt3;
		
	}
}



// --------------------------------------------------------
// D2Q9 update kernel for the orientation field assuming
// only diffusive transport.  See: Marth & Voigt 
// Interface Focus 6 (2016) 20160037.
// --------------------------------------------------------

__global__ void scsp_active_update_orientation_diffusive_D2Q9(
	float2* p,
	float2* h,
	int* nList,
	float fricR,
	int nVoxels)
{
	
	// -----------------------------------------------
	// define voxel:
	// -----------------------------------------------
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
		
		// advection contribution to dpdt (see Wikipedia page on 'material derivative'):
		float v0 = 0.0001;
		float2 u0 = v0*p[i];
		tensor2D dp = grad_vector_field_D2Q9(i,p,nList);
		float2 dpdt2 = make_float2(u0.x*dp.xx + u0.y*dp.xy, u0.x*dp.yx + u0.y*dp.yy);
		dpdt2 *= -1.0f;
		
		// molecular field contribution to dpdt:
		float2 dpdt = -h[i]/fricR;
		p[i] += dpdt + dpdt2;   // assume dt=1.0	
			
	}
}



// --------------------------------------------------------
// Kernel to calculate the active stress to be applied
// to the fluid.
// --------------------------------------------------------

__global__ void scsp_active_fluid_stress_D2Q9(
	float2* p,
	float2* h,
	tensor2D* stress,
	int* nList,
	float sf,
	float kapp,
	float activity,	
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {
				
		// calculate the elastic (passive) stress tensor:
		tensor2D ph = dyadic(p[i],h[i]);
		tensor2D phT = transpose(ph);
		tensor2D symph = 0.5*(ph - phT);
		tensor2D asymph = 0.5*sf*(ph + phT);
		tensor2D dp = grad_vector_field_D2Q9(i,p,nList);;
		dp = transpose(dp);
		stress[i] = symph - asymph - kapp*dp*transpose(dp);
						
		// calculate active stress tensor:
		stress[i] += -activity*(dyadic(p[i]) - 0.5*identity2D());
		
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
		F[i] += divergence_tensor_field_D2Q9(i,stress,nList);
	}	
}



// --------------------------------------------------------
// Kernel to calculate the molecular field "h" = dFdp
// --------------------------------------------------------

__global__ void scsp_active_fluid_molecular_field_D2Q9(
	float2* h,
	float2* p,
	tensor2D* stress,
	int* nList,
	float alpha,
	float kapp,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {		
		// molecular field:
		float p2 = length2(p[i]);
		float2 dfdp = alpha*p[i]*(p2 - 1.0);
		h[i] = dfdp - kapp*laplacian_vector_field_D2Q9(i,p,nList);
	}
}



// --------------------------------------------------------
// Kernel to calculate the molecular field "h" = dFdp
// here including the order parameter 'phi'
// --------------------------------------------------------

__global__ void scsp_active_fluid_molecular_field_with_phi_D2Q9(
	float* phi,
	float2* h,
	float2* p,
	tensor2D* stress,
	int* nList,
	float alpha,
	float kapp,
	float beta,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {		
		// molecular field:
		float p2 = length2(p[i]);
		float phicoeff = (phi[i] - 0.5)/0.5;   // phi_critical = 0.5
		float2 dfdp = alpha*p[i]*(p2 - phicoeff);
		h[i] = dfdp - kapp*laplacian_vector_field_D2Q9(i,p,nList) + beta*grad_scalar_field_D2Q9(i,phi,nList);	
	}
}



// --------------------------------------------------------
// Kernel to calculate the chemical potential of the 
// order parameter:
// --------------------------------------------------------

__global__ void scsp_active_fluid_chemical_potential_D2Q9(
	float* phi,
	float* chempot,
	float2* p,
	int* nList,
	float a,
	float alpha,
	float kapphi,
	float beta,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {		
		float laplphi = laplacian_scalar_field_D2Q9(i,phi,nList);
		float divp = divergence_vector_field_D2Q9(i,p,nList);
		float phii = phi[i];
		chempot[i] = a*(4.0*phii*phii*phii - 6.0*phii*phii + 2.0*phii) - kapphi*laplphi - beta*divp;
		//float p2 = length2(p[i]);
		//chempot[i] -= 0.25*alpha*p2;
	}		
}



// --------------------------------------------------------
// Kernel to calculate the chemical potentials of the 
// order parameters (used when there are 2 phi's):
// --------------------------------------------------------

__global__ void scsp_active_fluid_chemical_potential_2phi_D2Q9(
	float* phi1,
	float* phi2,
	float* chempot1,
	float* chempot2,
	float2* p,
	int* nList,
	float a,
	float alpha,
	float kapphi,
	float beta,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {		
		float laplphi1 = laplacian_scalar_field_D2Q9(i,phi1,nList);
		float laplphi2 = laplacian_scalar_field_D2Q9(i,phi2,nList);
		float divp = divergence_vector_field_D2Q9(i,p,nList);
		float p1 = phi1[i];
		float p2 = phi2[i];
		// chemical potentials for phi1,phi2:
		chempot1[i] = a*12.0*(p1*p1*p1 - p1*p1 + p1*p2*p2) - kapphi*laplphi1 - beta*divp;
		chempot2[i] = a*12.0*(p2*p2*p2 - p2*p2 + p2*p1*p1) - kapphi*laplphi2;
	}		
}



// --------------------------------------------------------
// Kernel to calculate the chemical potentials of the 
// order parameters (used when there are 3 phi's):
// --------------------------------------------------------

__global__ void scsp_active_fluid_chemical_potential_3phi_D2Q9(
	float* phi1,
	float* phi2,
	float* phi3,
	float* chempot1,
	float* chempot2,
	float* chempot3,
	float2* p,
	int* nList,
	float a,
	float alpha,
	float kapphi,
	float beta,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {		
		float laplphi1 = laplacian_scalar_field_D2Q9(i,phi1,nList);
		float laplphi2 = laplacian_scalar_field_D2Q9(i,phi2,nList);
		float laplphi3 = laplacian_scalar_field_D2Q9(i,phi3,nList);
		float divp = divergence_vector_field_D2Q9(i,p,nList);
		float p1 = phi1[i];
		float p2 = phi2[i];
		float p3 = phi3[i];
		// chemical potentials for phi1,phi2,phi3:
		chempot1[i] = a*12.0*(p1*p1*p1 - p1*p1 + p1*(p2*p2 + p3*p3)) - kapphi*laplphi1 - beta*divp;
		chempot2[i] = a*12.0*(p2*p2*p2 - p2*p2 + p2*(p1*p1 + p3*p3)) - kapphi*laplphi2;
		chempot3[i] = a*12.0*(p3*p3*p3 - p3*p3 + p3*(p1*p1 + p2*p2)) - kapphi*laplphi3;
	}		
}



// --------------------------------------------------------
// Kernel to calculate the interfacial capillary force
// the fluid:
// --------------------------------------------------------

__global__ void scsp_active_fluid_capillary_force_D2Q9(
	float* phi,
	float* chempot,
	float2* F,
	int* nList,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {		
		F[i] += chempot[i]*grad_scalar_field_D2Q9(i,phi,nList);
	}		
}



// --------------------------------------------------------
// Kernel to calculate the interfacial capillary force
// the fluid (used when there are 2 phi's):
// --------------------------------------------------------

__global__ void scsp_active_fluid_capillary_force_2phi_D2Q9(
	float* phi1,
	float* phi2,
	float* chempot1,
	float* chempot2,
	float2* F,
	int* nList,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {		
		F[i] += chempot1[i]*grad_scalar_field_D2Q9(i,phi1,nList) + 
			    chempot2[i]*grad_scalar_field_D2Q9(i,phi2,nList);
	}		
}



// --------------------------------------------------------
// Kernel to calculate the interfacial capillary force
// the fluid (used when there are 3 phi's):
// --------------------------------------------------------

__global__ void scsp_active_fluid_capillary_force_3phi_D2Q9(
	float* phi1,
	float* phi2,
	float* phi3,
	float* chempot1,
	float* chempot2,
	float* chempot3,
	float2* F,
	int* nList,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {		
		F[i] += chempot1[i]*grad_scalar_field_D2Q9(i,phi1,nList) + 
			    chempot2[i]*grad_scalar_field_D2Q9(i,phi2,nList) + 
				chempot3[i]*grad_scalar_field_D2Q9(i,phi3,nList);
	}		
}



// --------------------------------------------------------
// Kernel to update the order parameter phi:
// --------------------------------------------------------

__global__ void scsp_active_fluid_update_phi_D2Q9(
	float* phi,
	float* chempot,
	float2* u,
	int* nList,
	float mob,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {		
		float laplcp = laplacian_scalar_field_D2Q9(i,chempot,nList);
		float2 gradphi = grad_scalar_field_D2Q9(i,phi,nList);
		phi[i] += mob*laplcp - dot(u[i],gradphi);   // assume dt=1
	}		
}



// --------------------------------------------------------
// Kernel to update the order parameter phi (used when
// there are 2 phi's):
// --------------------------------------------------------

__global__ void scsp_active_fluid_update_phi_2phi_D2Q9(
	float* phi1,
	float* phi2,
	float* chempot1,
	float* chempot2,
	float2* u,
	int* nList,
	float mob,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {		
		float laplcp1 = laplacian_scalar_field_D2Q9(i,chempot1,nList);
		float laplcp2 = laplacian_scalar_field_D2Q9(i,chempot2,nList);
		float2 gradphi1 = grad_scalar_field_D2Q9(i,phi1,nList);
		float2 gradphi2 = grad_scalar_field_D2Q9(i,phi2,nList);
		phi1[i] += mob*laplcp1 - dot(u[i],gradphi1);   // assume dt=1
		phi2[i] += mob*laplcp2 - dot(u[i],gradphi2);   // assume dt=1
	}		
}



// --------------------------------------------------------
// Kernel to update the order parameter phi (used when
// there are 3 phi's):
// --------------------------------------------------------

__global__ void scsp_active_fluid_update_phi_3phi_D2Q9(
	float* phi1,
	float* phi2,
	float* phi3,
	float* chempot1,
	float* chempot2,
	float* chempot3,
	float2* u,
	int* nList,
	float mob,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {		
		float laplcp1 = laplacian_scalar_field_D2Q9(i,chempot1,nList);
		float laplcp2 = laplacian_scalar_field_D2Q9(i,chempot2,nList);
		float laplcp3 = laplacian_scalar_field_D2Q9(i,chempot3,nList);
		float2 gradphi1 = grad_scalar_field_D2Q9(i,phi1,nList);
		float2 gradphi2 = grad_scalar_field_D2Q9(i,phi2,nList);
		float2 gradphi3 = grad_scalar_field_D2Q9(i,phi3,nList);
		phi1[i] += mob*laplcp1 - dot(u[i],gradphi1);   // assume dt=1
		phi2[i] += mob*laplcp2 - dot(u[i],gradphi2);   // assume dt=1
		phi3[i] += mob*laplcp3 - dot(u[i],gradphi3);   // assume dt=1
	}		
}



// --------------------------------------------------------
// Kernel to update the order parameter phi assuming only
// diffusive transport:
// --------------------------------------------------------

__global__ void scsp_active_fluid_update_phi_diffusive_D2Q9(
	float* phi,
	float* chempot,
	float2* u,
	int* nList,
	float mob,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {		
		float laplcp = laplacian_scalar_field_D2Q9(i,chempot,nList);
		float divuphi = divergence_vector_scalar_field_D2Q9(i,u,phi,nList);		
		phi[i] += mob*laplcp - divuphi;   // assume dt=1
	}		
}



// --------------------------------------------------------
// Kernel to set the velocity field:
// --------------------------------------------------------

__global__ void scsp_active_fluid_set_velocity_field_D2Q9(
	float2* u,
	float2* p,
	float v0,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nVoxels) {		
		u[i] = v0*p[i];
	}
}











// ************************************************************************************************
//      Kernels that perform vector calculus operations
// ************************************************************************************************












// --------------------------------------------------------
// Kernel to compute gradient of vector field:
// --------------------------------------------------------

__device__ tensor2D grad_vector_field_D2Q9(
	int i,
	float2* v,
	int* nList)
{
	int offst = 9*i;
	tensor2D dv;
	// assuming dx=dy=1:
	dv.xx = (v[nList[offst+1]].x - v[nList[offst+3]].x)/2.0;  // d(v_x)/dx
	dv.xy = (v[nList[offst+2]].x - v[nList[offst+4]].x)/2.0;  // d(v_x)/dy
	dv.yx = (v[nList[offst+1]].y - v[nList[offst+3]].y)/2.0;  // d(v_y)/dy
	dv.yy = (v[nList[offst+2]].y - v[nList[offst+4]].y)/2.0;  // d(v_y)/dy
	return dv; 
}



// --------------------------------------------------------
// Kernel to compute gradient of scalar field:
// --------------------------------------------------------

__device__ float2 grad_scalar_field_D2Q9(
	int i,
	float* a,
	int* nList)
{
	int offst = 9*i;
	// assuming dx=dy=1:
	float gradx = (a[nList[offst+1]] - a[nList[offst+3]])/2.0;  // da/dx
	float grady = (a[nList[offst+2]] - a[nList[offst+4]])/2.0;  // da/dy
	return make_float2(gradx,grady);
}



// --------------------------------------------------------
// Kernel to compute laplacian of vector field.  This
// returns grad^2(vx) and grad^2(vy) as a float2:
// --------------------------------------------------------

__device__ float2 laplacian_vector_field_D2Q9(
	int i,
	float2* v,
	int* nList)
{
	int offst = 9*i;	
	float vx0 = v[i].x;
	float vy0 = v[i].y;
	float vxE = v[nList[offst+1]].x;  // east
	float vxN = v[nList[offst+2]].x;  // north
	float vxW = v[nList[offst+3]].x;  // west
	float vxS = v[nList[offst+4]].x;  // south
	float vyE = v[nList[offst+1]].y;  // east
	float vyN = v[nList[offst+2]].y;  // north
	float vyW = v[nList[offst+3]].y;  // west
	float vyS = v[nList[offst+4]].y;  // south	
	// assuming dx=dy=1:
	return make_float2(vxE + vxW + vxN + vxS - 4.0*vx0,
	                   vyE + vyW + vyN + vyS - 4.0*vy0);
}



// --------------------------------------------------------
// Kernel to compute laplacian of scalar field.
// --------------------------------------------------------

__device__ float laplacian_scalar_field_D2Q9(
	int i,
	float* a,
	int* nList)
{
	int offst = 9*i;	
	float a0 = a[i];
	float aE = a[nList[offst+1]];  // east
	float aN = a[nList[offst+2]];  // north
	float aW = a[nList[offst+3]];  // west
	float aS = a[nList[offst+4]];  // south
	// assuming dx=dy=1:
	return (aE + aW + aN + aS - 4.0*a0);
}



// --------------------------------------------------------
// Kernel to compute divergence of vector field:
// --------------------------------------------------------

__device__ float divergence_vector_field_D2Q9(
	int i,
	float2* v,
	int* nList)
{
	int offst = 9*i;	
	// assuming dx=dy=1:
	float dvxdx = (v[nList[offst+1]].x - v[nList[offst+3]].x)/2.0;
	float dvydy = (v[nList[offst+2]].y - v[nList[offst+4]].y)/2.0;
	return dvxdx + dvydy;
}



// --------------------------------------------------------
// Kernel to compute divergence of vector field times
// scalar field:
// --------------------------------------------------------

__device__ float divergence_vector_scalar_field_D2Q9(
	int i,
	float2* v,
	float* a,
	int* nList)
{
	int offst = 9*i;	
	// assuming dx=dy=1:	
	float aE = a[nList[offst+1]];  // east
	float aN = a[nList[offst+2]];  // north
	float aW = a[nList[offst+3]];  // west
	float aS = a[nList[offst+4]];  // south		
	float vxE = v[nList[offst+1]].x;  // east
	float vyN = v[nList[offst+2]].y;  // north
	float vxW = v[nList[offst+3]].x;  // west
	float vyS = v[nList[offst+4]].y;  // south
	float gradvxax = (vxE*aE - vxW*aW)/2.0;
	float gradvyay = (vyN*aN - vyS*aS)/2.0;
	return gradvxax + gradvyay;
}



// --------------------------------------------------------
// Kernel to compute divergence of tensor field:
// --------------------------------------------------------

__device__ float2 divergence_tensor_field_D2Q9(
	int i,
	tensor2D* t,
	int* nList)
{
	int offst = 9*i;	
	// divergence of tensor (see Wiki page on 'divergence'):
	float dtxxdx = (t[nList[offst+1]].xx - t[nList[offst+3]].xx) / 2.0;  // assume dx=1
	float dtxydy = (t[nList[offst+2]].xy - t[nList[offst+4]].xy) / 2.0;  // assume dx=1
	float dtyxdx = (t[nList[offst+1]].yx - t[nList[offst+3]].yx) / 2.0;  // assume dx=1
	float dtyydy = (t[nList[offst+2]].yy - t[nList[offst+4]].yy) / 2.0;  // assume dx=1
	return make_float2(dtxxdx + dtxydy, dtyxdx + dtyydy);
}











