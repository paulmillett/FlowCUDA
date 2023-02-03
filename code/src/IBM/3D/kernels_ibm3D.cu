
# include "kernels_ibm3D.cuh"
# include <stdio.h>



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_IBM3D(
	float3* r,
	float3* v,	
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) r[i] += v[i];  // assume dt = 1
}



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_dt_IBM3D(
	float3* r,
	float3* v,
	float dt,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) r[i] += v[i]*dt;
}



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_verlet_1_IBM3D(
	float3* r,
	float3* v,
	float3* f,
	float dt,
	float m,	
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		r[i] += v[i]*dt + 0.5*dt*dt*(f[i]/m);
		v[i] += 0.5*dt*(f[i]/m);
	}
}



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_verlet_2_IBM3D(
	float3* v,
	float3* f,
	float dt,
	float m,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		v[i] += 0.5*dt*(f[i]/m);
	}
}



// --------------------------------------------------------
// IBM3D node update kernel (this uses force and a mobility
// constant, instead of velocity):
// --------------------------------------------------------

__global__ void update_node_position_vacuum_IBM3D(
	float3* r,
	float3* f,
	float M,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		r[i] += M*f[i];
	}
}



// --------------------------------------------------------
// IBM3D zero the velocities and forces:
// --------------------------------------------------------

__global__ void zero_velocities_forces_IBM3D(
	float3* v,
	float3* f,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		v[i] = make_float3(0.0f,0.0f,0.0f);
		f[i] = make_float3(0.0f,0.0f,0.0f);
	}
}



// --------------------------------------------------------
// IBM3D add force in the x-direction:
// --------------------------------------------------------

__global__ void add_xdir_force_IBM3D(
	float3* f,
	float fx,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		f[i].x += fx;
	}
}



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_IBM3D(
	float3* r,
	float3* r_start,
	float3* r_end,
	float3* v,	
	int step,
	int nSteps,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		float3 r_old = r[i];		
		//float psi = sin(float(step)/float(nSteps)*M_PI/2.0);
		float psi = 0.5*(sin(M_PI*(float(step)/float(nSteps) - 0.5)) + 1.0); 
		r[i] = r_start[i] + psi*(r_end[i] - r_start[i]);
		v[i] = r[i] - r_old;  // assume dt = 1		
	}
}



// --------------------------------------------------------
// IBM3D reference node update kernel:
// --------------------------------------------------------

__global__ void update_node_ref_position_IBM3D(
	float3* r_ref,
	float3* r_ref_delta,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) r_ref[i] += r_ref_delta[i];
}



// --------------------------------------------------------
// IBM3D reference node update kernel:
// --------------------------------------------------------

__global__ void update_node_ref_position_IBM3D(
	float3* r_ref,
	float3* r_ref_start,
	float3* r_ref_end,
	int step,
	int nSteps,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		float psi = sin(float(step)/float(nSteps)*M_PI/2.0);
		r_ref[i] = r_ref_start[i] + psi*(r_ref_end[i] - r_ref_start[i]);			
	}
}



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void set_reference_node_positions_IBM3D(
	float3* r,
	float3* r0,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) r0[i] = r[i];
}



// --------------------------------------------------------
// IBM3D kernel to interpolate LBM velocity to IBM node:
// --------------------------------------------------------

__global__ void interpolate_velocity_IBM3D(
	float3* r,
	float3* v,
	float* uLBM,
	float* vLBM,
	float* wLBM,
	int Nx,
	int Ny,
	int Nz,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nNodes) {
		
		// --------------------------------------
		// zero out velocities for node "i"
		// --------------------------------------
		
		v[i] = make_float3(0.0);		
				
		// --------------------------------------
		// find nearest LBM voxel (rounded down)
		// --------------------------------------
		
		int i0 = int(floor(r[i].x));
		int j0 = int(floor(r[i].y));
		int k0 = int(floor(r[i].z));
		
		// --------------------------------------
		// loop over footprint
		// --------------------------------------
		
		for (int kk=k0; kk<=k0+1; kk++) {			
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {
					int ndx = voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = r[i].x - float(ii);
					float ry = r[i].y - float(jj);
					float rz = r[i].z - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					v[i].x += del*uLBM[ndx];
					v[i].y += del*vLBM[ndx];
					v[i].z += del*wLBM[ndx];
				}
			}		
		}		
	}	
}



// --------------------------------------------------------
// IBM3D kernel to extrapolate IBM node velocity to LBM lattice
// --------------------------------------------------------

__global__ void extrapolate_velocity_IBM3D(
	float3* r,
	float3* v,
	float* uIBvox,
	float* vIBvox,
	float* wIBvox,
	float* weight,
	int Nx,
	int Ny,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nNodes) {
				
		// --------------------------------------
		// find nearest LBM voxel (rounded down)
		// --------------------------------------
		
		int i0 = int(floor(r[i].x));
		int j0 = int(floor(r[i].y));
		int k0 = int(floor(r[i].z));
		
		// --------------------------------------
		// loop over footprint
		// --------------------------------------
		
		for (int kk=k0; kk<=k0+1; kk++) {
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {				
					int ndx = kk*Nx*Ny + jj*Nx + ii;
					float rx = r[i].x - float(ii);
					float ry = r[i].y - float(jj);
					float rz = r[i].z - float(kk);
					float del = sqrt(rx*rx + ry*ry + rz*rz);
					atomicAdd(&uIBvox[ndx],del*v[i].x);
					atomicAdd(&vIBvox[ndx],del*v[i].y);
					atomicAdd(&wIBvox[ndx],del*v[i].z);
					atomicAdd(&weight[ndx],del);
				}
			}		
		}		
	}	
}



// --------------------------------------------------------
// IBM3D kernel to extrapolate IBM node force to LBM lattice
// --------------------------------------------------------

__global__ void extrapolate_force_IBM3D(
	float3* r,
	float3* f,
	float* fxLBM,
	float* fyLBM,
	float* fzLBM,
	int Nx,
	int Ny,
	int Nz,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nNodes) {
		
		// --------------------------------------
		// find nearest LBM voxel (rounded down)
		// --------------------------------------
		
		int i0 = int(floor(r[i].x));
		int j0 = int(floor(r[i].y));
		int k0 = int(floor(r[i].z));
		
		// --------------------------------------
		// loop over footprint
		// --------------------------------------
		
		for (int kk=k0; kk<=k0+1; kk++) {
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {				
					int ndx = voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = r[i].x - float(ii);
					float ry = r[i].y - float(jj);
					float rz = r[i].z - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					atomicAdd(&fxLBM[ndx],del*f[i].x);
					atomicAdd(&fyLBM[ndx],del*f[i].y);
					atomicAdd(&fzLBM[ndx],del*f[i].z);
				}
			}		
		}		
	}	
}



// --------------------------------------------------------
// IBM3D kernel to calculate viscous force due to velocity
// difference between IBM node and LBM fluid
// --------------------------------------------------------

__global__ void viscous_force_velocity_difference_IBM3D(
	float3* r,
	float3* v,
	float3* f,
	float* fxLBM,
	float* fyLBM,
	float* fzLBM,
	float* uLBM,
	float* vLBM,
	float* wLBM,
	float gam,
	int Nx,
	int Ny,
	int Nz,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nNodes) {
				
		// --------------------------------------
		// find nearest LBM voxel (rounded down)
		// --------------------------------------
		
		int i0 = int(floor(r[i].x));
		int j0 = int(floor(r[i].y));
		int k0 = int(floor(r[i].z));
		
		// --------------------------------------
		// loop over footprint to get 
		// interpolated LBM velocity:
		// --------------------------------------
		
		/*
		float vxLBMi = 0.0;
		float vyLBMi = 0.0;
		float vzLBMi = 0.0;		
		for (int kk=k0; kk<=k0+1; kk++) {
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {				
					int ndx = voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = r[i].x - float(ii);
					float ry = r[i].y - float(jj);
					float rz = r[i].z - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					vxLBMi += del*uLBM[ndx];
					vyLBMi += del*vLBM[ndx];
					vzLBMi += del*wLBM[ndx];
				}
			}
		}
		*/
		
		float vxLBMi = 0.0;
		float vyLBMi = 0.0;
		float vzLBMi = 0.0;		
		for (int kk=k0; kk<=k0+1; kk++) {
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {				
					int ndx = voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = r[i].x - float(ii);
					float ry = r[i].y - float(jj);
					float rz = r[i].z - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					vxLBMi += del*uLBM[ndx];
					vyLBMi += del*vLBM[ndx];
					vzLBMi += del*wLBM[ndx];
					// extrapolate elastic forces to LBM fluid:
					atomicAdd(&fxLBM[ndx],del*f[i].x);
					atomicAdd(&fyLBM[ndx],del*f[i].y);
					atomicAdd(&fzLBM[ndx],del*f[i].z);					
				}
			}
		}
		
		// --------------------------------------
		// calculate friction forces & add them
		// to IBM node:
		// --------------------------------------
		
		float vfx = gam*(vxLBMi - v[i].x);
		float vfy = gam*(vyLBMi - v[i].y);
		float vfz = gam*(vzLBMi - v[i].z);
		f[i].x += vfx;
		f[i].y += vfy;
		f[i].z += vfz;
		
		// --------------------------------------
		// loop over footprint to extrapolate
		// viscous force to LBM lattice:
		// --------------------------------------
		
		/*
		for (int kk=k0; kk<=k0+1; kk++) {
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {				
					int ndx = voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = r[i].x - float(ii);
					float ry = r[i].y - float(jj);
					float rz = r[i].z - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					// we add the negative of viscous force:
					atomicAdd(&fxLBM[ndx],-del*vfx);
					atomicAdd(&fyLBM[ndx],-del*vfy);
					atomicAdd(&fzLBM[ndx],-del*vfz);
				}
			}		
		}
		*/
				
	}	
}



// --------------------------------------------------------
// IBM3D kernel to compute force on node:
// --------------------------------------------------------

__global__ void compute_node_force_IBM3D(
	float3* r,
	float3* r0,
	float3* f,
	float kstiff,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) f[i] = -kstiff*(r[i] - r0[i]);
}



// --------------------------------------------------------
// IBM3D kernel to determine 1D index from 3D indices:
// --------------------------------------------------------

__device__ inline int voxel_ndx(
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
