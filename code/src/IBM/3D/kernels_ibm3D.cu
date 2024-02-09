
# include "kernels_ibm3D.cuh"
# include <stdio.h>



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_IBM3D(
	node* nodes,	
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) nodes[i].r += nodes[i].v;  // assume dt = 1
}



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_dt_IBM3D(
	node* nodes,
	float dt,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) nodes[i].r += nodes[i].v*dt;
}



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_include_force_IBM3D(
	node* nodes,
	float dt,
	float m,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		nodes[i].r += nodes[i].v*dt + 0.5*dt*dt*(nodes[i].f/m);
	}	
}



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_overdamped_IBM3D(
	node* nodes,
	float dt,
	float fric,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		nodes[i].v = nodes[i].f/fric;
		nodes[i].r += dt*nodes[i].v;
	}	
}



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_verlet_1_IBM3D(
	node* nodes,
	float dt,
	float m,	
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {		
		nodes[i].r += nodes[i].v*dt + 0.5*dt*dt*(nodes[i].f/m);
		nodes[i].v += 0.5*dt*(nodes[i].f/m);		
	}
}



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_verlet_2_IBM3D(
	node* nodes,
	float dt,
	float m,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {		
		nodes[i].v += 0.5*dt*(nodes[i].f/m);
	}
}



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_verlet_1_drag_IBM3D(
	node* nodes,
	float dt,
	float m,
	float gam,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {		
		nodes[i].r += nodes[i].v*dt + 0.5*dt*dt*(nodes[i].f - gam*nodes[i].v)/m;
		nodes[i].v += 0.5*dt*(nodes[i].f - gam*nodes[i].v)/m;		
	}
}



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_verlet_2_drag_IBM3D(
	node* nodes,
	float dt,
	float m,
	float gam,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {		
		nodes[i].v = (nodes[i].v + 0.5*dt*nodes[i].f/m)/(1.0 + 0.5*dt*gam/m);
	}
}



// --------------------------------------------------------
// IBM3D node update kernel (this uses force and a mobility
// constant, instead of velocity):
// --------------------------------------------------------

__global__ void update_node_position_vacuum_IBM3D(
	node* nodes,
	float M,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		nodes[i].r += M*nodes[i].f;
	}
}



// --------------------------------------------------------
// IBM3D zero the velocities and forces:
// --------------------------------------------------------

__global__ void zero_velocities_forces_IBM3D(
	node* nodes,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		nodes[i].v = make_float3(0.0f,0.0f,0.0f);
		nodes[i].f = make_float3(0.0f,0.0f,0.0f);
	}
}



// --------------------------------------------------------
// IBM3D enforce a maximum node force:
// --------------------------------------------------------

__global__ void enforce_max_node_force_IBM3D(
	node* nodes,
	float fmax,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		float fi = length(nodes[i].f);
		if (fi > fmax) {
			nodes[i].f *= (fmax/fi);
		}
	}
}



// --------------------------------------------------------
// IBM3D add a drag force:
// --------------------------------------------------------

__global__ void add_drag_force_to_node_IBM3D(
	node* nodes,
	float c,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		nodes[i].f -= c*nodes[i].v;
	}
}



// --------------------------------------------------------
// IBM3D add force in the x-direction:
// --------------------------------------------------------

__global__ void add_xdir_force_IBM3D(
	node* nodes,
	float fx,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		nodes[i].f.x += fx;
	}
}



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_IBM3D(
	node* nodes,
	float3* r_start,
	float3* r_end,
	int step,
	int nSteps,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		float3 r_old = nodes[i].r;		
		//float psi = sin(float(step)/float(nSteps)*M_PI/2.0);
		float psi = 0.5*(sin(M_PI*(float(step)/float(nSteps) - 0.5)) + 1.0); 
		nodes[i].r = r_start[i] + psi*(r_end[i] - r_start[i]);
		nodes[i].v = nodes[i].r - r_old;  // assume dt = 1		
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
	node* nodes,
	float3* r0,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) r0[i] = nodes[i].r;
}



// --------------------------------------------------------
// IBM3D kernel to interpolate LBM velocity to IBM node:
// --------------------------------------------------------

__global__ void interpolate_velocity_IBM3D(
	node* nodes,
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
		
		nodes[i].v = make_float3(0.0);		
				
		// --------------------------------------
		// find nearest LBM voxel (rounded down)
		// --------------------------------------
		
		int i0 = int(floor(nodes[i].r.x));
		int j0 = int(floor(nodes[i].r.y));
		int k0 = int(floor(nodes[i].r.z));
		
		// --------------------------------------
		// loop over footprint
		// --------------------------------------
		
		for (int kk=k0; kk<=k0+1; kk++) {			
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {
					int ndx = voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = nodes[i].r.x - float(ii);
					float ry = nodes[i].r.y - float(jj);
					float rz = nodes[i].r.z - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					nodes[i].v.x += del*uLBM[ndx];
					nodes[i].v.y += del*vLBM[ndx];
					nodes[i].v.z += del*wLBM[ndx];
				}
			}		
		}		
	}	
}



// --------------------------------------------------------
// IBM3D kernel to extrapolate IBM node velocity to LBM lattice
// --------------------------------------------------------

__global__ void extrapolate_velocity_IBM3D(
	node* nodes,
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
		
		int i0 = int(floor(nodes[i].r.x));
		int j0 = int(floor(nodes[i].r.y));
		int k0 = int(floor(nodes[i].r.z));
		
		// --------------------------------------
		// loop over footprint
		// --------------------------------------
		
		for (int kk=k0; kk<=k0+1; kk++) {
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {				
					int ndx = kk*Nx*Ny + jj*Nx + ii;
					float rx = nodes[i].r.x - float(ii);
					float ry = nodes[i].r.y - float(jj);
					float rz = nodes[i].r.z - float(kk);
					float del = sqrt(rx*rx + ry*ry + rz*rz);
					atomicAdd(&uIBvox[ndx],del*nodes[i].v.x);
					atomicAdd(&vIBvox[ndx],del*nodes[i].v.y);
					atomicAdd(&wIBvox[ndx],del*nodes[i].v.z);
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
	node* nodes,
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
		
		int i0 = int(floor(nodes[i].r.x));
		int j0 = int(floor(nodes[i].r.y));
		int k0 = int(floor(nodes[i].r.z));
		
		// --------------------------------------
		// loop over footprint
		// --------------------------------------
		
		for (int kk=k0; kk<=k0+1; kk++) {
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {				
					int ndx = voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = nodes[i].r.x - float(ii);
					float ry = nodes[i].r.y - float(jj);
					float rz = nodes[i].r.z - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					atomicAdd(&fxLBM[ndx],del*nodes[i].f.x);
					atomicAdd(&fyLBM[ndx],del*nodes[i].f.y);
					atomicAdd(&fzLBM[ndx],del*nodes[i].f.z);
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
	node* nodes,
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
		
		int i0 = int(floor(nodes[i].r.x));
		int j0 = int(floor(nodes[i].r.y));
		int k0 = int(floor(nodes[i].r.z));
		
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
					int ndx = voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = nodes[i].r.x - float(ii);
					float ry = nodes[i].r.y - float(jj);
					float rz = nodes[i].r.z - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					vxLBMi += del*uLBM[ndx];
					vyLBMi += del*vLBM[ndx];
					vzLBMi += del*wLBM[ndx];
					// extrapolate elastic forces to LBM fluid:
					atomicAdd(&fxLBM[ndx],del*nodes[i].f.x);
					atomicAdd(&fyLBM[ndx],del*nodes[i].f.y);
					atomicAdd(&fzLBM[ndx],del*nodes[i].f.z);					
				}
			}
		}
		
		// --------------------------------------
		// calculate friction forces & add them
		// to IBM node:
		// --------------------------------------
		
		float vfx = gam*(vxLBMi - nodes[i].v.x);
		float vfy = gam*(vyLBMi - nodes[i].v.y);
		float vfz = gam*(vzLBMi - nodes[i].v.z);
		nodes[i].f.x += vfx;
		nodes[i].f.y += vfy;
		nodes[i].f.z += vfz;
				
	}	
}



// --------------------------------------------------------
// IBM3D kernel to calculate viscous force due to velocity
// difference between IBM node and LBM fluid
// --------------------------------------------------------

__global__ void repulsive_force_solid_lattice_IBM3D(
	node* nodes,
	int* solid,
	float repA,
	float repD,
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
		
		int i0 = int(floor(nodes[i].r.x));
		int j0 = int(floor(nodes[i].r.y));
		int k0 = int(floor(nodes[i].r.z));
		
		// --------------------------------------
		// loop over footprint to get 
		// interpolated LBM velocity:
		// --------------------------------------
				
		for (int kk=k0; kk<=k0+1; kk++) {
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {				
					int ndx = voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					if (solid[ndx] == 1) {
						float rx = nodes[i].r.x - float(ii);
						float ry = nodes[i].r.y - float(jj);
						float rz = nodes[i].r.z - float(kk);
						float r = sqrt(rx*rx + ry*ry + rz*rz);
						if (r <= repD) {
							float force = repA - (repA/repD)*r;
							nodes[i].f.x += force*(rx/r);
							nodes[i].f.y += force*(ry/r);
							nodes[i].f.z += force*(rz/r);
						}						
					}				
				}
			}
		}		
	}	
}



// --------------------------------------------------------
// IBM3D kernel to compute force on node:
// --------------------------------------------------------

__global__ void compute_node_force_IBM3D(
	node* nodes,
	float3* r0,
	float kstiff,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) nodes[i].f = -kstiff*(nodes[i].r - r0[i]);
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
