# include "kernels_scsp_D3Q19.cuh"
# include "../iolets/zou_he_BC_D3Q19.cuh"
# include <stdio.h>



// --------------------------------------------------------
// D3Q19 kernel to reset forces to zero: 
// --------------------------------------------------------

__global__ void scsp_zero_forces_D3Q19(
	float* fx,
	float* fy,
	float* fz,
	int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {			
		fx[i] = 0.0;
		fy[i] = 0.0;
		fz[i] = 0.0;
	}
}



// --------------------------------------------------------
// D2Q9 kernel to re-set the fluid forces (and extrapolated
// IB velocities and weights) to zero:
// --------------------------------------------------------

__global__ void scsp_zero_forces_D3Q19(
	float* fx,
	float* fy,
	float* fz,
	float* uIBvox,
	float* vIBvox,
	float* wIBvox,
	float* weights,
	int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {			
		fx[i] = 0.0;
		fy[i] = 0.0;
		fz[i] = 0.0;
		uIBvox[i] = 0.0;
		vIBvox[i] = 0.0;
		wIBvox[i] = 0.0;
		weights[i] = 0.0;
	}
}



// --------------------------------------------------------
// D3Q19 initialize kernel:
// --------------------------------------------------------

__global__ void scsp_initial_equilibrium_D3Q19(float* f1,
										       float* r,
										       float* u,
										       float* v,
										       float* w,
										       int nVoxels)
{
	
	// -----------------------------------------------
	// define current voxel:
	// -----------------------------------------------
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	// -----------------------------------------------
	// assign equilibrium populations:
	// -----------------------------------------------
	
	if (i < nVoxels) {			
		// useful constants: 
		const int offst = 19*i;
		equilibrium_populations_bb_D3Q19(f1,r[i],u[i],v[i],w[i],offst);
	}		
}



// --------------------------------------------------------
// D3Q19 equilibrium populations:
// --------------------------------------------------------

__device__ void equilibrium_populations_bb_D3Q19(float* f1,
										         const float r,
										         const float u,
										         const float v,
												 const float w,
												 const int offst)
{
	const float w0r = r*1.0/3.0;
	const float wsr = r*1.0/18.0;
	const float wdr = r*1.0/36.0;
	const float omusq = 1.0 - 1.5*(u*u + v*v + w*w);	
	const float tux = 3.0*u;
	const float tvy = 3.0*v;
	const float twz = 3.0*w;
	// equilibrium populations:
	f1[offst+0] = w0r*(omusq);				
	float cidot3u = tux;
	f1[offst+1] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = -tux;
	f1[offst+2] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tvy;
	f1[offst+3] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = -tvy;
	f1[offst+4] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));	
	cidot3u = twz;
	f1[offst+5] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = -twz;
	f1[offst+6] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));		
	cidot3u = tux+tvy;
	f1[offst+7] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = -(tux+tvy);
	f1[offst+8] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tux+twz;
	f1[offst+9] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = -(tux+twz);
	f1[offst+10] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tvy+twz;
	f1[offst+11] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = -(tvy+twz);
	f1[offst+12] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tux-tvy;
	f1[offst+13] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tvy-tux;
	f1[offst+14] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tux-twz;
	f1[offst+15] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = twz-tux;
	f1[offst+16] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tvy-twz;
	f1[offst+17] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = twz-tvy;
	f1[offst+18] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
}



// --------------------------------------------------------
// D3Q19 add body force to fluid nodes:
// --------------------------------------------------------

__global__ void scsp_add_body_force_D3Q19(
	float bx,
	float by,
	float bz,
	float* fx,
	float* fy,
	float* fz,
	int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {	
		fx[i] += bx;
		fy[i] += by;
		fz[i] += bz;
	}
}



// --------------------------------------------------------
// D3Q19 add body force to fluid nodes.  Nodes above and
// below 'zdivide' get different body forces
// --------------------------------------------------------

__global__ void scsp_add_body_force_divided_D3Q19(
	float bxL,
	float bxU,
	float* fx,
	float* fy,
	float* fz,
	int nVoxels,
	int Nx,
	int Ny,
	int Nz,
	int zdivide)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {	
		float bx = 0.0;
		int zi = i/(Nx*Ny);
		if (zi < zdivide) bx = bxL;  // lower body force
		if (zi > zdivide) bx = bxU;  // upper body force
		fx[i] += bx;
		fy[i] += 0.0;
		fz[i] += 0.0;
	}
}



// --------------------------------------------------------
// D3Q19 add body force to fluid nodes:
// --------------------------------------------------------

__global__ void scsp_add_body_force_solid_D3Q19(
	float bx,
	float by,
	float bz,
	float* fx,
	float* fy,
	float* fz,
	int* solid,
	int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nVoxels) {	
		if (solid[i] == 0) {
			fx[i] += bx;
			fy[i] += by;
			fz[i] += bz;
		}		
	}
}



// --------------------------------------------------------
// D3Q19 kernel to set shear velocities at the y=0 and
// y=Ny-1 boundaries.  The shear direction is the x-dir.
// NOTE: This should be called AFTER the collide-streaming
//       step.  It should be the last calculation for the 
//       fluid update.  
// --------------------------------------------------------

__global__ void scsp_set_boundary_shear_velocity_D3Q19(float uBot,
                                                       float uTop,
													   float* f1,													   
													   float* u,
													   float* v,
													   float* w,
													   float* r,
													   int Nx,
													   int Ny,
													   int Nz,
													   int nVoxels)
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {		
		int zi = i/(Nx*Ny);    // z-index assuming data is ordered first x, then y, then z
		if (zi == 0) {
			int offst = 19*i;
			u[i] = uBot;
			v[i] = 0.0;
			w[i] = 0.0;
			equilibrium_populations_bb_D3Q19(f1,r[i],u[i],v[i],w[i],offst);
		} 
		if (zi == Nz-1) {
			int offst = 19*i;
			u[i] = uTop;
			v[i] = 0.0;
			w[i] = 0.0;
			equilibrium_populations_bb_D3Q19(f1,r[i],u[i],v[i],w[i],offst);
		}		
	}		
}



// --------------------------------------------------------
// D3Q19 kernel to set wall velocities at the z=0 and
// z=Nz-1 boundaries.  The flow direction is the x-dir.
// NOTE: This should be called AFTER the collide-streaming
//       step.  It should be the last calculation for the 
//       fluid update.  
// --------------------------------------------------------

__global__ void scsp_set_boundary_slit_velocity_D3Q19(float uWall,
                                                      float* f1,													   
													  float* u,
													  float* v,
													  float* w,
													  float* r,
													  int Nx,
													  int Ny,
													  int Nz,
													  int nVoxels)
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {		
		// 3D indices assuming data is ordered first x, then y, then z
		int zi = i/(Nx*Ny); 
		if (zi == 0) {
			int offst = 19*i;
			u[i] = uWall;
			v[i] = 0.0;
			w[i] = 0.0;
			r[i] = 1.0;
			equilibrium_populations_bb_D3Q19(f1,r[i],u[i],v[i],w[i],offst);
		} 
		if (zi == Nz-1) {
			int offst = 19*i;
			u[i] = uWall;
			v[i] = 0.0;
			w[i] = 0.0;
			r[i] = 1.0;
			equilibrium_populations_bb_D3Q19(f1,r[i],u[i],v[i],w[i],offst);
		}		
	}		
}



// --------------------------------------------------------
// D3Q19 kernel to set the channel velocities at the y=0,
// y=Ny-1, z=0, and z=Nz-1 boundaries.  The velocity 
// direction is the x-dir.
// NOTE: This should be called AFTER the collide-streaming
//       step.  It should be the last calculation for the 
//       fluid update.  
// --------------------------------------------------------

__global__ void scsp_set_channel_wall_velocity_D3Q19(float uWall,
													 float* f1,													   
													 float* u,
													 float* v,
													 float* w,
													 float* r,
													 int Nx,
													 int Ny,
													 int Nz,
													 int nVoxels)
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {		
		// 3D indices assuming data is ordered first x, then y, then z
		//int xi = i%Nx;
		int yi = (i/Nx)%Ny;
		int zi = i/(Nx*Ny);  
		if (yi == 0) {
			int offst = 19*i;
			u[i] = uWall;
			v[i] = 0.0;
			w[i] = 0.0;
			equilibrium_populations_bb_D3Q19(f1,r[i],u[i],v[i],w[i],offst);
		} 
		if (yi == Ny-1) {
			int offst = 19*i;
			u[i] = uWall;
			v[i] = 0.0;
			w[i] = 0.0;
			equilibrium_populations_bb_D3Q19(f1,r[i],u[i],v[i],w[i],offst);
		}		
		if (zi == 0) {
			int offst = 19*i;
			u[i] = uWall;
			v[i] = 0.0;
			w[i] = 0.0;
			equilibrium_populations_bb_D3Q19(f1,r[i],u[i],v[i],w[i],offst);
		} 
		if (zi == Nz-1) {
			int offst = 19*i;
			u[i] = uWall;
			v[i] = 0.0;
			w[i] = 0.0;
			equilibrium_populations_bb_D3Q19(f1,r[i],u[i],v[i],w[i],offst);
		}		
	}		
}



// --------------------------------------------------------
// D3Q19 kernel to set wall densities at the z=0 and
// z=Nz-1 boundaries.  The flow direction is the x-dir.
// --------------------------------------------------------

__global__ void scsp_set_boundary_slit_density_D3Q19(float* f1,													   
												     int Nx,
													 int Ny,
													 int Nz,
													 int nVoxels)
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {		
		// 3D indices assuming data is ordered first x, then y, then z
		int zi = i/(Nx*Ny); 
		const float w0 = 1.0/3.0;
		const float ws = 1.0/18.0;
		const float wd = 1.0/36.0;
		// only top and bottom boundaries
		if (zi == 0 || zi == Nz-1) {
			int offst = 19*i;			
			float rho = f1[offst+0]+f1[offst+1]+f1[offst+2]+f1[offst+3]+f1[offst+4]+
			            f1[offst+5]+f1[offst+6]+f1[offst+7]+f1[offst+8]+f1[offst+9]+
			            f1[offst+10]+f1[offst+11]+f1[offst+12]+f1[offst+13]+f1[offst+14]+
			            f1[offst+15]+f1[offst+16]+f1[offst+17]+f1[offst+18];
			float rAdd = 1.0 - rho;
			f1[offst+0] += w0*rAdd;
			f1[offst+1] += ws*rAdd;
			f1[offst+2] += ws*rAdd;
			f1[offst+3] += ws*rAdd;
			f1[offst+4] += ws*rAdd;
			f1[offst+5] += ws*rAdd;
			f1[offst+6] += ws*rAdd;
			f1[offst+7] += wd*rAdd;
			f1[offst+8] += wd*rAdd;
			f1[offst+9] += wd*rAdd;
			f1[offst+10] += wd*rAdd;
			f1[offst+11] += wd*rAdd;
			f1[offst+12] += wd*rAdd;
			f1[offst+13] += wd*rAdd;
			f1[offst+14] += wd*rAdd;
			f1[offst+15] += wd*rAdd;
			f1[offst+16] += wd*rAdd;
			f1[offst+17] += wd*rAdd;
			f1[offst+18] += wd*rAdd;
		} 
	}		
}



// --------------------------------------------------------
// D3Q19 kernel to set wall densities at the z=0, 
// z=Nz-1, y=0, and y=Ny-1 boundaries.
// The flow direction is the x-dir.
// --------------------------------------------------------

__global__ void scsp_set_boundary_duct_density_D3Q19(float* f1,													   
												     int Nx,
													 int Ny,
													 int Nz,
													 int nVoxels)
{
	// define voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {		
		// 3D indices assuming data is ordered first x, then y, then z
		//int xi = i%Nx;
		int yi = (i/Nx)%Ny;
		int zi = i/(Nx*Ny); 
		const float w0 = 1.0/3.0;
		const float ws = 1.0/18.0;
		const float wd = 1.0/36.0;
		// only duct boundaries in y- and z-dir's
		if (zi == 0 || zi == Nz-1 || yi == 0 || yi == Ny-1) {
			int offst = 19*i;			
			float rho = f1[offst+0]+f1[offst+1]+f1[offst+2]+f1[offst+3]+f1[offst+4]+
			            f1[offst+5]+f1[offst+6]+f1[offst+7]+f1[offst+8]+f1[offst+9]+
			            f1[offst+10]+f1[offst+11]+f1[offst+12]+f1[offst+13]+f1[offst+14]+
			            f1[offst+15]+f1[offst+16]+f1[offst+17]+f1[offst+18];
			float rAdd = 1.0 - rho;
			f1[offst+0] += w0*rAdd;
			f1[offst+1] += ws*rAdd;
			f1[offst+2] += ws*rAdd;
			f1[offst+3] += ws*rAdd;
			f1[offst+4] += ws*rAdd;
			f1[offst+5] += ws*rAdd;
			f1[offst+6] += ws*rAdd;
			f1[offst+7] += wd*rAdd;
			f1[offst+8] += wd*rAdd;
			f1[offst+9] += wd*rAdd;
			f1[offst+10] += wd*rAdd;
			f1[offst+11] += wd*rAdd;
			f1[offst+12] += wd*rAdd;
			f1[offst+13] += wd*rAdd;
			f1[offst+14] += wd*rAdd;
			f1[offst+15] += wd*rAdd;
			f1[offst+16] += wd*rAdd;
			f1[offst+17] += wd*rAdd;
			f1[offst+18] += wd*rAdd;
		} 
	}		
}



// --------------------------------------------------------
// D3Q19 update kernel.
// This algorithm is based on the optimized "stream-collide-
// save" algorithm recommended by T. Kruger in the 
// textbook: "The Lattice Boltzmann Method: Principles
// and Practice".
// --------------------------------------------------------

__global__ void scsp_stream_collide_save_D3Q19(float* f1,
                                               float* f2,
										       float* r,
										       float* u,
										       float* v,
										       float* w,
										       int* streamIndex,
										       int* voxelType, 
										       iolet* iolets,
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
		int offst = 19*i;	
		float ft[19];
		
		// --------------------------------------------------		
		// STREAMING - load populations from adjacent voxels,
		//             note	that streamIndex[] accounts for
		//             halfway bounceback conditions.
		// --------------------------------------------------
				
		ft[0]  = f1[streamIndex[offst+0]];                   
		ft[1]  = f1[streamIndex[offst+1]]; 
		ft[2]  = f1[streamIndex[offst+2]];  
		ft[3]  = f1[streamIndex[offst+3]];  
		ft[4]  = f1[streamIndex[offst+4]];  
		ft[5]  = f1[streamIndex[offst+5]]; 
		ft[6]  = f1[streamIndex[offst+6]];  
		ft[7]  = f1[streamIndex[offst+7]];  
		ft[8]  = f1[streamIndex[offst+8]]; 
		ft[9]  = f1[streamIndex[offst+9]]; 
		ft[10] = f1[streamIndex[offst+10]]; 	
		ft[11] = f1[streamIndex[offst+11]]; 	
		ft[12] = f1[streamIndex[offst+12]]; 	
		ft[13] = f1[streamIndex[offst+13]]; 	
		ft[14] = f1[streamIndex[offst+14]]; 	
		ft[15] = f1[streamIndex[offst+15]]; 	
		ft[16] = f1[streamIndex[offst+16]]; 	
		ft[17] = f1[streamIndex[offst+17]]; 	
		ft[18] = f1[streamIndex[offst+18]]; 	
		
		// --------------------------------------------------		
		// INLETS/OUTLETS - correct the ft[] values at inlets
		//                  and outlets using Zou-He BC's.
		// --------------------------------------------------
		
		if (vtype > 0) {
			zou_he_BC_D3Q19(vtype,ft,iolets);
		}
		
		// --------------------------------------------------
		// MACROS - calculate the velocity and density.
		// --------------------------------------------------
		
		float rho = ft[0]+ft[1]+ft[2]+ft[3]+ft[4]+ft[5]+ft[6]+ft[7]+ft[8]+ft[9]+ft[10]+ft[11]+
			        ft[12]+ft[13]+ft[14]+ft[15]+ft[16]+ft[17]+ft[18];
		float rhoinv = 1.0/rho;
		float ux = rhoinv*(ft[1] + ft[7] + ft[9]  + ft[13] + ft[15] - (ft[2] + ft[8]  + ft[10] + ft[14] + ft[16]));
		float vy = rhoinv*(ft[3] + ft[7] + ft[11] + ft[14] + ft[17] - (ft[4] + ft[8]  + ft[12] + ft[13] + ft[18]));
		float wz = rhoinv*(ft[5] + ft[9] + ft[11] + ft[16] + ft[18] - (ft[6] + ft[10] + ft[12] + ft[15] + ft[17]));
		
		// --------------------------------------------------
		// COLLISION - perform the BGK collision operator.
		// --------------------------------------------------
		
		// useful constants:
		const float tauinv = 2.0/(6.0*nu + 1.0);   // 1/tau
		const float omtauinv = 1.0 - tauinv;       // 1 - 1/tau
		const float tw0r = (1.0/3.0)*rho*tauinv;   // w[0]*rho/tau
		const float twsr = (1.0/18.0)*rho*tauinv;  // w[1-6]*rho/tau
		const float twdr = (1.0/36.0)*rho*tauinv;  // w[7-18]*rho/tau
		const float omusq = 1.0 - 1.5*(ux*ux + vy*vy + wz*wz);
		const float tux = 3.0*ux;
		const float tvy = 3.0*vy;
		const float twz = 3.0*wz;
		// collision calculations:	
		f2[offst+0] = omtauinv*ft[0] + tw0r*(omusq);
		float cidot3u = tux;
		f2[offst+1] = omtauinv*ft[1] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -tux;
		f2[offst+2] = omtauinv*ft[2] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy;
		f2[offst+3] = omtauinv*ft[3] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -tvy;
		f2[offst+4] = omtauinv*ft[4] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = twz;
		f2[offst+5] = omtauinv*ft[5] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -twz;
		f2[offst+6] = omtauinv*ft[6] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tux+tvy;
		f2[offst+7] = omtauinv*ft[7] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -(tux+tvy);
		f2[offst+8] = omtauinv*ft[8] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tux+twz;
		f2[offst+9] = omtauinv*ft[9] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -(tux+twz);
		f2[offst+10] = omtauinv*ft[10] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy+twz;
		f2[offst+11] = omtauinv*ft[11] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -(tvy+twz);
		f2[offst+12] = omtauinv*ft[12] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tux-tvy;
		f2[offst+13] = omtauinv*ft[13] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy-tux;
		f2[offst+14] = omtauinv*ft[14] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tux-twz;
		f2[offst+15] = omtauinv*ft[15] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = twz-tux;
		f2[offst+16] = omtauinv*ft[16] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy-twz;
		f2[offst+17] = omtauinv*ft[17] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = twz-tvy;
		f2[offst+18] = omtauinv*ft[18] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	
		// --------------------------------------------------		
		// SAVE - write macros to arrays only
		//        when needed.
		// --------------------------------------------------
		
		if (save) {
			r[i] = rho;
			u[i] = ux;
			v[i] = vy;
			w[i] = wz;
		}
					
	}
}



// --------------------------------------------------------
// D3Q19 update kernel.
// This algorithm is based on the optimized "stream-collide-
// save" algorithm recommended by T. Kruger in the 
// textbook: "The Lattice Boltzmann Method: Principles
// and Practice".
// --------------------------------------------------------

__global__ void scsp_stream_collide_save_forcing_D3Q19(float* f1,
                                                       float* f2,
										               float* r,
										               float* u,
										               float* v,
										               float* w,
													   float* Fx,
													   float* Fy,
													   float* Fz,
										               int* streamIndex,
										               int* voxelType, 
										               iolet* iolets,
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
		int offst = 19*i;	
		float ft[19];
		
		// --------------------------------------------------		
		// STREAMING - load populations from adjacent voxels,
		//             note	that streamIndex[] accounts for
		//             halfway bounceback conditions.
		// --------------------------------------------------
				
		ft[0]  = f1[streamIndex[offst+0]];                   
		ft[1]  = f1[streamIndex[offst+1]]; 
		ft[2]  = f1[streamIndex[offst+2]];  
		ft[3]  = f1[streamIndex[offst+3]];  
		ft[4]  = f1[streamIndex[offst+4]];  
		ft[5]  = f1[streamIndex[offst+5]]; 
		ft[6]  = f1[streamIndex[offst+6]];  
		ft[7]  = f1[streamIndex[offst+7]];  
		ft[8]  = f1[streamIndex[offst+8]]; 
		ft[9]  = f1[streamIndex[offst+9]]; 
		ft[10] = f1[streamIndex[offst+10]]; 	
		ft[11] = f1[streamIndex[offst+11]]; 	
		ft[12] = f1[streamIndex[offst+12]]; 	
		ft[13] = f1[streamIndex[offst+13]]; 	
		ft[14] = f1[streamIndex[offst+14]]; 	
		ft[15] = f1[streamIndex[offst+15]]; 	
		ft[16] = f1[streamIndex[offst+16]]; 	
		ft[17] = f1[streamIndex[offst+17]]; 	
		ft[18] = f1[streamIndex[offst+18]]; 	
		
		// --------------------------------------------------		
		// INLETS/OUTLETS - correct the ft[] values at inlets
		//                  and outlets using Zou-He BC's.
		// --------------------------------------------------
		
		if (vtype > 0) {
			zou_he_BC_D3Q19(vtype,ft,iolets);
		}
		
		// --------------------------------------------------
		// MACROS - calculate the velocity and density (force
		//          corrected).
		// --------------------------------------------------	
		
		float rho = ft[0]+ft[1]+ft[2]+ft[3]+ft[4]+ft[5]+ft[6]+ft[7]+ft[8]+ft[9]+ft[10]+ft[11]+
			        ft[12]+ft[13]+ft[14]+ft[15]+ft[16]+ft[17]+ft[18];
		float rhoinv = 1.0/rho;
		float ux = rhoinv*(ft[1] + ft[7] + ft[9]  + ft[13] + ft[15] - (ft[2] + ft[8]  + ft[10] + ft[14] + ft[16]) + 0.5*Fx[i]);
		float vy = rhoinv*(ft[3] + ft[7] + ft[11] + ft[14] + ft[17] - (ft[4] + ft[8]  + ft[12] + ft[13] + ft[18]) + 0.5*Fy[i]);
		float wz = rhoinv*(ft[5] + ft[9] + ft[11] + ft[16] + ft[18] - (ft[6] + ft[10] + ft[12] + ft[15] + ft[17]) + 0.5*Fz[i]);
		
		// --------------------------------------------------
		// COLLISION - perform the BGK collision operator
		//             with Guo forcing.
		// --------------------------------------------------	
		
		// useful constants:
		const float w0 = 1.0/3.0;
		const float ws = 1.0/18.0;
		const float wd = 1.0/36.0;			
		const float omega = 2.0/(6.0*nu + 1.0);   // 1/tau
		const float omomega = 1.0 - omega;        // 1 - 1/tau
		const float omomega2 = 1.0 - 0.5*omega;   // 1 - 1/(2tau)
		const float omusq = 1.0 - 1.5*(ux*ux + vy*vy + wz*wz);
			
		// direction 0
		float evel = 0.0;       // e dot velocity
		float emiu = 0.0-ux;    // e minus u
		float emiv = 0.0-vy;    // e minus v
		float emiw = 0.0-wz;    // e minus w
		float feq = w0*rho*omusq;
		float frc = w0*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw));
		f2[offst+0] = omomega*ft[0] + omega*feq + omomega2*frc;
			
		// direction 1
		evel = ux;
		emiu = 1.0-ux;
		emiv = 0.0-vy;
		emiw = 0.0-wz;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw));
		f2[offst+1] = omomega*ft[1] + omega*feq + omomega2*frc;
		
		// direction 2
		evel = -ux;
		emiu = -1.0-ux;
		emiv = 0.0-vy;
		emiw = 0.0-wz;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw));
		f2[offst+2] = omomega*ft[2] + omega*feq + omomega2*frc;
		
		// direction 3
		evel = vy;
		emiu = 0.0-ux;
		emiv = 1.0-vy;
		emiw = 0.0-wz;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv + 9.0*evel) + Fz[i]*(3.0*emiw));
		f2[offst+3] = omomega*ft[3] + omega*feq + omomega2*frc;
		
		// direction 4
		evel = -vy;
		emiu = 0.0-ux;
		emiv = -1.0-vy;
		emiw = 0.0-wz;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv - 9.0*evel) + Fz[i]*(3.0*emiw));
		f2[offst+4] = omomega*ft[4] + omega*feq + omomega2*frc;
		
		// direction 5
		evel = wz;
		emiu = 0.0-ux;
		emiv = 0.0-vy;
		emiw = 1.0-wz;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw + 9.0*evel));
		f2[offst+5] = omomega*ft[5] + omega*feq + omomega2*frc;
		
		// direction 6
		evel = -wz;
		emiu = 0.0-ux;
		emiv = 0.0-vy;
		emiw = -1.0-wz;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw - 9.0*evel));
		f2[offst+6] = omomega*ft[6] + omega*feq + omomega2*frc;
		
		// direction 7
		evel = ux+vy;
		emiu = 1.0-ux;
		emiv = 1.0-vy;
		emiw = 0.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv + 9.0*evel) + Fz[i]*(3.0*emiw));
		f2[offst+7] = omomega*ft[7] + omega*feq + omomega2*frc;
		
		// direction 8
		evel = -ux-vy;
		emiu = -1.0-ux;
		emiv = -1.0-vy;
		emiw = 0.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv - 9.0*evel) + Fz[i]*(3.0*emiw));
		f2[offst+8] = omomega*ft[8] + omega*feq + omomega2*frc;
		
		// direction 9
		evel = ux+wz;
		emiu = 1.0-ux;
		emiv = 0.0-vy;
		emiw = 1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw + 9.0*evel));
		f2[offst+9] = omomega*ft[9] + omega*feq + omomega2*frc;
		
		// direction 10
		evel = -ux-wz;
		emiu = -1.0-ux;
		emiv = 0.0-vy;
		emiw = -1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw - 9.0*evel));
		f2[offst+10] = omomega*ft[10] + omega*feq + omomega2*frc;
		
		// direction 11
		evel = vy+wz;
		emiu = 0.0-ux;
		emiv = 1.0-vy;
		emiw = 1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv + 9.0*evel) + Fz[i]*(3.0*emiw + 9.0*evel));
		f2[offst+11] = omomega*ft[11] + omega*feq + omomega2*frc;
		
		// direction 12
		evel = -vy-wz;
		emiu = 0.0-ux;
		emiv = -1.0-vy;
		emiw = -1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv - 9.0*evel) + Fz[i]*(3.0*emiw - 9.0*evel));
		f2[offst+12] = omomega*ft[12] + omega*feq + omomega2*frc;
		
		// direction 13
		evel = ux-vy;
		emiu = 1.0-ux;
		emiv = -1.0-vy;
		emiw = 0.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv - 9.0*evel) + Fz[i]*(3.0*emiw));
		f2[offst+13] = omomega*ft[13] + omega*feq + omomega2*frc;
		
		// direction 14
		evel = -ux+vy;
		emiu = -1.0-ux;
		emiv = 1.0-vy;
		emiw = 0.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv + 9.0*evel) + Fz[i]*(3.0*emiw));
		f2[offst+14] = omomega*ft[14] + omega*feq + omomega2*frc;
		
		// direction 15
		evel = ux-wz;
		emiu = 1.0-ux;
		emiv = 0.0-vy;
		emiw = -1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw - 9.0*evel));
		f2[offst+15] = omomega*ft[15] + omega*feq + omomega2*frc;
		
		// direction 16
		evel = -ux+wz;
		emiu = -1.0-ux;
		emiv = 0.0-vy;
		emiw = 1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw + 9.0*evel));
		f2[offst+16] = omomega*ft[16] + omega*feq + omomega2*frc;
		
		// direction 17
		evel = vy-wz;
		emiu = 0.0-ux;
		emiv = 1.0-vy;
		emiw = -1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv + 9.0*evel) + Fz[i]*(3.0*emiw - 9.0*evel));
		f2[offst+17] = omomega*ft[17] + omega*feq + omomega2*frc;
		
		// direction 18
		evel = -vy+wz;
		emiu = 0.0-ux;
		emiv = -1.0-vy;
		emiw = 1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv - 9.0*evel) + Fz[i]*(3.0*emiw + 9.0*evel));
		f2[offst+18] = omomega*ft[18] + omega*feq + omomega2*frc;
			
		// --------------------------------------------------		
		// SAVE - write macros to arrays 
		// --------------------------------------------------
		
		r[i] = rho;
		u[i] = ux;
		v[i] = vy;
		w[i] = wz;
					
	}
}



// --------------------------------------------------------
// D3Q19 update kernel.
// This algorithm is based on the optimized "stream-collide-
// save" algorithm recommended by T. Kruger in the 
// textbook: "The Lattice Boltzmann Method: Principles
// and Practice".
//
// Here, solid voxels are skipped.
// --------------------------------------------------------

__global__ void scsp_stream_collide_save_forcing_solid_D3Q19(
	float* f1,
	float* f2,
	float* r,
	float* u,
	float* v,
	float* w,
	float* Fx,
	float* Fy,
	float* Fz,
	int* streamIndex,
	int* voxelType,
	int* solid,
	iolet* iolets,
	float nu,
	float dt,
	int nVoxels)
{

	// -----------------------------------------------
	// define voxel:
	// -----------------------------------------------
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
		
		// --------------------------------------------------		
		// only proceed if fluid voxel:
		// --------------------------------------------------
		
		if (solid[i] == 0) {
			
			// --------------------------------------------------		
			// voxel-specific parameters:
			// --------------------------------------------------
		
			int vtype = voxelType[i];
			int offst = 19*i;	
			float ft[19];
		
			// --------------------------------------------------		
			// STREAMING - load populations from adjacent voxels,
			//             note	that streamIndex[] accounts for
			//             halfway bounceback conditions.
			// --------------------------------------------------
				
			ft[0]  = f1[streamIndex[offst+0]];                   
			ft[1]  = f1[streamIndex[offst+1]]; 
			ft[2]  = f1[streamIndex[offst+2]];  
			ft[3]  = f1[streamIndex[offst+3]];  
			ft[4]  = f1[streamIndex[offst+4]];  
			ft[5]  = f1[streamIndex[offst+5]]; 
			ft[6]  = f1[streamIndex[offst+6]];  
			ft[7]  = f1[streamIndex[offst+7]];  
			ft[8]  = f1[streamIndex[offst+8]]; 
			ft[9]  = f1[streamIndex[offst+9]]; 
			ft[10] = f1[streamIndex[offst+10]]; 	
			ft[11] = f1[streamIndex[offst+11]]; 	
			ft[12] = f1[streamIndex[offst+12]]; 	
			ft[13] = f1[streamIndex[offst+13]]; 	
			ft[14] = f1[streamIndex[offst+14]]; 	
			ft[15] = f1[streamIndex[offst+15]]; 	
			ft[16] = f1[streamIndex[offst+16]]; 	
			ft[17] = f1[streamIndex[offst+17]]; 	
			ft[18] = f1[streamIndex[offst+18]]; 	
		
			// --------------------------------------------------		
			// INLETS/OUTLETS - correct the ft[] values at inlets
			//                  and outlets using Zou-He BC's.
			// --------------------------------------------------
		
			if (vtype > 0) {
				zou_he_BC_D3Q19(vtype,ft,iolets);
			}
		
			// --------------------------------------------------
			// MACROS - calculate the velocity and density (force
			//          corrected).
			// --------------------------------------------------	
		
			float rho = ft[0]+ft[1]+ft[2]+ft[3]+ft[4]+ft[5]+ft[6]+ft[7]+ft[8]+ft[9]+ft[10]+ft[11]+
				        ft[12]+ft[13]+ft[14]+ft[15]+ft[16]+ft[17]+ft[18];
			float rhoinv = 1.0/rho;
			float ux = rhoinv*(ft[1] + ft[7] + ft[9]  + ft[13] + ft[15] - (ft[2] + ft[8]  + ft[10] + ft[14] + ft[16]) + 0.5*Fx[i]);
			float vy = rhoinv*(ft[3] + ft[7] + ft[11] + ft[14] + ft[17] - (ft[4] + ft[8]  + ft[12] + ft[13] + ft[18]) + 0.5*Fy[i]);
			float wz = rhoinv*(ft[5] + ft[9] + ft[11] + ft[16] + ft[18] - (ft[6] + ft[10] + ft[12] + ft[15] + ft[17]) + 0.5*Fz[i]);
		
			// --------------------------------------------------
			// COLLISION - perform the BGK collision operator
			//             with Guo forcing.
			// --------------------------------------------------	
		
			// useful constants:
			const float w0 = 1.0/3.0;
			const float ws = 1.0/18.0;
			const float wd = 1.0/36.0;			
			const float omega = 2.0/(6.0*nu + 1.0);   // 1/tau
			const float omomega = 1.0 - omega;        // 1 - 1/tau
			const float omomega2 = 1.0 - 0.5*omega;   // 1 - 1/(2tau)
			const float omusq = 1.0 - 1.5*(ux*ux + vy*vy + wz*wz);
			
			// direction 0
			float evel = 0.0;       // e dot velocity
			float emiu = 0.0-ux;    // e minus u
			float emiv = 0.0-vy;    // e minus v
			float emiw = 0.0-wz;    // e minus w
			float feq = w0*rho*omusq;
			float frc = w0*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw));
			f2[offst+0] = omomega*ft[0] + omega*feq + omomega2*frc;
			
			// direction 1
			evel = ux;
			emiu = 1.0-ux;
			emiv = 0.0-vy;
			emiw = 0.0-wz;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw));
			f2[offst+1] = omomega*ft[1] + omega*feq + omomega2*frc;
		
			// direction 2
			evel = -ux;
			emiu = -1.0-ux;
			emiv = 0.0-vy;
			emiw = 0.0-wz;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw));
			f2[offst+2] = omomega*ft[2] + omega*feq + omomega2*frc;
		
			// direction 3
			evel = vy;
			emiu = 0.0-ux;
			emiv = 1.0-vy;
			emiw = 0.0-wz;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv + 9.0*evel) + Fz[i]*(3.0*emiw));
			f2[offst+3] = omomega*ft[3] + omega*feq + omomega2*frc;
		
			// direction 4
			evel = -vy;
			emiu = 0.0-ux;
			emiv = -1.0-vy;
			emiw = 0.0-wz;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv - 9.0*evel) + Fz[i]*(3.0*emiw));
			f2[offst+4] = omomega*ft[4] + omega*feq + omomega2*frc;
		
			// direction 5
			evel = wz;
			emiu = 0.0-ux;
			emiv = 0.0-vy;
			emiw = 1.0-wz;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw + 9.0*evel));
			f2[offst+5] = omomega*ft[5] + omega*feq + omomega2*frc;
		
			// direction 6
			evel = -wz;
			emiu = 0.0-ux;
			emiv = 0.0-vy;
			emiw = -1.0-wz;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw - 9.0*evel));
			f2[offst+6] = omomega*ft[6] + omega*feq + omomega2*frc;
		
			// direction 7
			evel = ux+vy;
			emiu = 1.0-ux;
			emiv = 1.0-vy;
			emiw = 0.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv + 9.0*evel) + Fz[i]*(3.0*emiw));
			f2[offst+7] = omomega*ft[7] + omega*feq + omomega2*frc;
		
			// direction 8
			evel = -ux-vy;
			emiu = -1.0-ux;
			emiv = -1.0-vy;
			emiw = 0.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv - 9.0*evel) + Fz[i]*(3.0*emiw));
			f2[offst+8] = omomega*ft[8] + omega*feq + omomega2*frc;
		
			// direction 9
			evel = ux+wz;
			emiu = 1.0-ux;
			emiv = 0.0-vy;
			emiw = 1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw + 9.0*evel));
			f2[offst+9] = omomega*ft[9] + omega*feq + omomega2*frc;
		
			// direction 10
			evel = -ux-wz;
			emiu = -1.0-ux;
			emiv = 0.0-vy;
			emiw = -1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw - 9.0*evel));
			f2[offst+10] = omomega*ft[10] + omega*feq + omomega2*frc;
		
			// direction 11
			evel = vy+wz;
			emiu = 0.0-ux;
			emiv = 1.0-vy;
			emiw = 1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv + 9.0*evel) + Fz[i]*(3.0*emiw + 9.0*evel));
			f2[offst+11] = omomega*ft[11] + omega*feq + omomega2*frc;
		
			// direction 12
			evel = -vy-wz;
			emiu = 0.0-ux;
			emiv = -1.0-vy;
			emiw = -1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv - 9.0*evel) + Fz[i]*(3.0*emiw - 9.0*evel));
			f2[offst+12] = omomega*ft[12] + omega*feq + omomega2*frc;
		
			// direction 13
			evel = ux-vy;
			emiu = 1.0-ux;
			emiv = -1.0-vy;
			emiw = 0.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv - 9.0*evel) + Fz[i]*(3.0*emiw));
			f2[offst+13] = omomega*ft[13] + omega*feq + omomega2*frc;
		
			// direction 14
			evel = -ux+vy;
			emiu = -1.0-ux;
			emiv = 1.0-vy;
			emiw = 0.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv + 9.0*evel) + Fz[i]*(3.0*emiw));
			f2[offst+14] = omomega*ft[14] + omega*feq + omomega2*frc;
		
			// direction 15
			evel = ux-wz;
			emiu = 1.0-ux;
			emiv = 0.0-vy;
			emiw = -1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw - 9.0*evel));
			f2[offst+15] = omomega*ft[15] + omega*feq + omomega2*frc;
		
			// direction 16
			evel = -ux+wz;
			emiu = -1.0-ux;
			emiv = 0.0-vy;
			emiw = 1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv) + Fz[i]*(3.0*emiw + 9.0*evel));
			f2[offst+16] = omomega*ft[16] + omega*feq + omomega2*frc;
		
			// direction 17
			evel = vy-wz;
			emiu = 0.0-ux;
			emiv = 1.0-vy;
			emiw = -1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv + 9.0*evel) + Fz[i]*(3.0*emiw - 9.0*evel));
			f2[offst+17] = omomega*ft[17] + omega*feq + omomega2*frc;
		
			// direction 18
			evel = -vy+wz;
			emiu = 0.0-ux;
			emiv = -1.0-vy;
			emiw = 1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv - 9.0*evel) + Fz[i]*(3.0*emiw + 9.0*evel));
			f2[offst+18] = omomega*ft[18] + omega*feq + omomega2*frc;
			
			// --------------------------------------------------		
			// SAVE - write macros to arrays 
			// --------------------------------------------------
		
			r[i] = rho;
			u[i] = ux;
			v[i] = vy;
			w[i] = wz;
			
		}						
	}
}



// --------------------------------------------------------
// D3Q19 update kernel.
// This algorithm is based on the optimized "stream-collide-
// save" algorithm recommended by T. Kruger in the 
// textbook: "The Lattice Boltzmann Method: Principles
// and Practice".
// --------------------------------------------------------

__global__ void scsp_stream_collide_save_IBforcing_D3Q19(float* f1,
                                                         float* f2,
										                 float* r,
										                 float* u,
										                 float* v,
										                 float* w,
													     float* uIBvox,
													     float* vIBvox,
													     float* wIBvox,
														 float* weights,
										                 int* streamIndex,
										                 int* voxelType, 
										                 iolet* iolets,
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
		int offst = 19*i;	
		float ft[19];
		
		// --------------------------------------------------		
		// STREAMING - load populations from adjacent voxels,
		//             note	that streamIndex[] accounts for
		//             halfway bounceback conditions.
		// --------------------------------------------------
				
		ft[0]  = f1[streamIndex[offst+0]];                   
		ft[1]  = f1[streamIndex[offst+1]]; 
		ft[2]  = f1[streamIndex[offst+2]];  
		ft[3]  = f1[streamIndex[offst+3]];  
		ft[4]  = f1[streamIndex[offst+4]];  
		ft[5]  = f1[streamIndex[offst+5]]; 
		ft[6]  = f1[streamIndex[offst+6]];  
		ft[7]  = f1[streamIndex[offst+7]];  
		ft[8]  = f1[streamIndex[offst+8]]; 
		ft[9]  = f1[streamIndex[offst+9]]; 
		ft[10] = f1[streamIndex[offst+10]]; 	
		ft[11] = f1[streamIndex[offst+11]]; 	
		ft[12] = f1[streamIndex[offst+12]]; 	
		ft[13] = f1[streamIndex[offst+13]]; 	
		ft[14] = f1[streamIndex[offst+14]]; 	
		ft[15] = f1[streamIndex[offst+15]]; 	
		ft[16] = f1[streamIndex[offst+16]]; 	
		ft[17] = f1[streamIndex[offst+17]]; 	
		ft[18] = f1[streamIndex[offst+18]]; 
		
		// --------------------------------------------------		
		// INLETS/OUTLETS - correct the ft[] values at inlets
		//                  and outlets using Zou-He BC's.
		// --------------------------------------------------
		
		if (vtype > 0) {
			zou_he_BC_D3Q19(vtype,ft,iolets);
		}
		
		// --------------------------------------------------
		// MACROS - calculate the velocity and density.
		// --------------------------------------------------	
		
		float rho = ft[0]+ft[1]+ft[2]+ft[3]+ft[4]+ft[5]+ft[6]+ft[7]+ft[8]+ft[9]+ft[10]+ft[11]+
			        ft[12]+ft[13]+ft[14]+ft[15]+ft[16]+ft[17]+ft[18];
		float rhoinv = 1.0/rho;
		float ux = rhoinv*(ft[1] + ft[7] + ft[9]  + ft[13] + ft[15] - (ft[2] + ft[8]  + ft[10] + ft[14] + ft[16]));
		float vy = rhoinv*(ft[3] + ft[7] + ft[11] + ft[14] + ft[17] - (ft[4] + ft[8]  + ft[12] + ft[13] + ft[18]));
		float wz = rhoinv*(ft[5] + ft[9] + ft[11] + ft[16] + ft[18] - (ft[6] + ft[10] + ft[12] + ft[15] + ft[17]));
		
		// --------------------------------------------------
		// IB FORCE CORRECTION - calculate and add the force
		//                       needed to match the fluid
		//                       velocity with the IB velocity
		// --------------------------------------------------	
		
		float Fx = 0.0;
		float Fy = 0.0;
		float Fz = 0.0;
		if (weights[i] > 0.0){
			float uxIB = uIBvox[i]/weights[i];  // weighted average
			float vyIB = vIBvox[i]/weights[i];  // "              "	
			float wzIB = wIBvox[i]/weights[i];  // "              "		
			Fx = (uxIB - ux)*2.0*rho;
			Fy = (vyIB - vy)*2.0*rho;
			Fz = (wzIB - wz)*2.0*rho;
			ux += 0.5*Fx*rhoinv;
			vy += 0.5*Fy*rhoinv;
			wz += 0.5*Fz*rhoinv;
		}		
		
		// --------------------------------------------------
		// COLLISION - perform the BGK collision operator
		//             with Guo forcing.
		// --------------------------------------------------		
		
		// useful constants:
		const float w0 = 1.0/3.0;
		const float ws = 1.0/18.0;
		const float wd = 1.0/36.0;			
		const float omega = 2.0/(6.0*nu + 1.0);   // 1/tau
		const float omomega = 1.0 - omega;        // 1 - 1/tau
		const float omomega2 = 1.0 - 0.5*omega;   // 1 - 1/(2tau)
		const float omusq = 1.0 - 1.5*(ux*ux + vy*vy + wz*wz);
			
		// direction 0
		float evel = 0.0;       // e dot velocity
		float emiu = 0.0-ux;    // e minus u
		float emiv = 0.0-vy;    // e minus v
		float emiw = 0.0-wz;    // e minus w
		float feq = w0*rho*omusq;
		float frc = w0*(Fx*(3.0*emiu) + Fy*(3.0*emiv) + Fz*(3.0*emiw));
		f2[offst+0] = omomega*ft[0] + omega*feq + omomega2*frc;
			
		// direction 1
		evel = ux;
		emiu = 1.0-ux;
		emiv = 0.0-vy;
		emiw = 0.0-wz;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx*(3.0*emiu + 9.0*evel) + Fy*(3.0*emiv) + Fz*(3.0*emiw));
		f2[offst+1] = omomega*ft[1] + omega*feq + omomega2*frc;
			
		// direction 2
		evel = -ux;
		emiu = -1.0-ux;
		emiv = 0.0-vy;
		emiw = 0.0-wz;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx*(3.0*emiu - 9.0*evel) + Fy*(3.0*emiv) + Fz*(3.0*emiw));
		f2[offst+2] = omomega*ft[2] + omega*feq + omomega2*frc;
			
		// direction 3
		evel = vy;
		emiu = 0.0-ux;
		emiv = 1.0-vy;
		emiw = 0.0-wz;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx*(3.0*emiu) + Fy*(3.0*emiv + 9.0*evel) + Fz*(3.0*emiw));
		f2[offst+3] = omomega*ft[3] + omega*feq + omomega2*frc;
			
		// direction 4
		evel = -vy;
		emiu = 0.0-ux;
		emiv = -1.0-vy;
		emiw = 0.0-wz;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx*(3.0*emiu) + Fy*(3.0*emiv - 9.0*evel) + Fz*(3.0*emiw));
		f2[offst+4] = omomega*ft[4] + omega*feq + omomega2*frc;
			
		// direction 5
		evel = wz;
		emiu = 0.0-ux;
		emiv = 0.0-vy;
		emiw = 1.0-wz;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx*(3.0*emiu) + Fy*(3.0*emiv) + Fz*(3.0*emiw + 9.0*evel));
		f2[offst+5] = omomega*ft[5] + omega*feq + omomega2*frc;
		
		// direction 6
		evel = -wz;
		emiu = 0.0-ux;
		emiv = 0.0-vy;
		emiw = -1.0-wz;
		feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = ws*(Fx*(3.0*emiu) + Fy*(3.0*emiv) + Fz*(3.0*emiw - 9.0*evel));
		f2[offst+6] = omomega*ft[6] + omega*feq + omomega2*frc;
		
		// direction 7
		evel = ux+vy;
		emiu = 1.0-ux;
		emiv = 1.0-vy;
		emiw = 0.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu + 9.0*evel) + Fy*(3.0*emiv + 9.0*evel) + Fz*(3.0*emiw));
		f2[offst+7] = omomega*ft[7] + omega*feq + omomega2*frc;
		
		// direction 8
		evel = -ux-vy;
		emiu = -1.0-ux;
		emiv = -1.0-vy;
		emiw = 0.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu - 9.0*evel) + Fy*(3.0*emiv - 9.0*evel) + Fz*(3.0*emiw));
		f2[offst+8] = omomega*ft[8] + omega*feq + omomega2*frc;
		
		// direction 9
		evel = ux+wz;
		emiu = 1.0-ux;
		emiv = 0.0-vy;
		emiw = 1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu + 9.0*evel) + Fy*(3.0*emiv) + Fz*(3.0*emiw + 9.0*evel));
		f2[offst+9] = omomega*ft[9] + omega*feq + omomega2*frc;
		
		// direction 10
		evel = -ux-wz;
		emiu = -1.0-ux;
		emiv = 0.0-vy;
		emiw = -1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu - 9.0*evel) + Fy*(3.0*emiv) + Fz*(3.0*emiw - 9.0*evel));
		f2[offst+10] = omomega*ft[10] + omega*feq + omomega2*frc;
		
		// direction 11
		evel = vy+wz;
		emiu = 0.0-ux;
		emiv = 1.0-vy;
		emiw = 1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu) + Fy*(3.0*emiv + 9.0*evel) + Fz*(3.0*emiw + 9.0*evel));
		f2[offst+11] = omomega*ft[11] + omega*feq + omomega2*frc;
		
		// direction 12
		evel = -vy-wz;
		emiu = 0.0-ux;
		emiv = -1.0-vy;
		emiw = -1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu) + Fy*(3.0*emiv - 9.0*evel) + Fz*(3.0*emiw - 9.0*evel));
		f2[offst+12] = omomega*ft[12] + omega*feq + omomega2*frc;
		
		// direction 13
		evel = ux-vy;
		emiu = 1.0-ux;
		emiv = -1.0-vy;
		emiw = 0.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu + 9.0*evel) + Fy*(3.0*emiv - 9.0*evel) + Fz*(3.0*emiw));
		f2[offst+13] = omomega*ft[13] + omega*feq + omomega2*frc;
		
		// direction 14
		evel = -ux+vy;
		emiu = -1.0-ux;
		emiv = 1.0-vy;
		emiw = 0.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu - 9.0*evel) + Fy*(3.0*emiv + 9.0*evel) + Fz*(3.0*emiw));
		f2[offst+14] = omomega*ft[14] + omega*feq + omomega2*frc;
		
		// direction 15
		evel = ux-wz;
		emiu = 1.0-ux;
		emiv = 0.0-vy;
		emiw = -1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu + 9.0*evel) + Fy*(3.0*emiv) + Fz*(3.0*emiw - 9.0*evel));
		f2[offst+15] = omomega*ft[15] + omega*feq + omomega2*frc;
		
		// direction 16
		evel = -ux+wz;
		emiu = -1.0-ux;
		emiv = 0.0-vy;
		emiw = 1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu - 9.0*evel) + Fy*(3.0*emiv) + Fz*(3.0*emiw + 9.0*evel));
		f2[offst+16] = omomega*ft[16] + omega*feq + omomega2*frc;
		
		// direction 17
		evel = vy-wz;
		emiu = 0.0-ux;
		emiv = 1.0-vy;
		emiw = -1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu) + Fy*(3.0*emiv + 9.0*evel) + Fz*(3.0*emiw - 9.0*evel));
		f2[offst+17] = omomega*ft[17] + omega*feq + omomega2*frc;
		
		// direction 18
		evel = -vy+wz;
		emiu = 0.0-ux;
		emiv = -1.0-vy;
		emiw = 1.0-wz;
		feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
		frc = wd*(Fx*(3.0*emiu) + Fy*(3.0*emiv - 9.0*evel) + Fz*(3.0*emiw + 9.0*evel));
		f2[offst+18] = omomega*ft[18] + omega*feq + omomega2*frc;						
			
		// --------------------------------------------------		
		// SAVE - write macros to arrays 
		// --------------------------------------------------
		
		r[i] = rho;
		u[i] = ux;
		v[i] = vy;
		w[i] = wz;
					
	}
}




