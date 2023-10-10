
# include "kernels_poisson_ibm3D.cuh"
# include <stdio.h>




// --------------------------------------------------------
// IBM3D kernel to reset the 'G' vector array
// --------------------------------------------------------

__global__ void zero_G_poisson_IBM3D(
	float3* G,
	int nVoxels)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nVoxels) {
		G[i] = make_float3(0.0f,0.0f,0.0f);
	}
}


// --------------------------------------------------------
// IBM3D kernel to extrapolate IBM face normal to 
// LBM lattice
// --------------------------------------------------------

__global__ void extrapolate_interface_normal_poisson_IBM3D(
	node* nodes,
	float3* G,
	int Nx,
	int Ny,
	int Nz,
	int nFaces,
	int cellType,
	cell* cells,
	triangle* faces)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nFaces) {
		
		// --------------------------------------
		// make sure cell type is correct:
		// --------------------------------------
		
		int cID = faces[i].cellID;
		if (cells[cID].cellType != cellType) return;
		
		// --------------------------------------
		// center of mass of face 
		// (note: must take into account PBC's!!!
		// see Wikipedia page on c.o.m.)
		// --------------------------------------
		
		int V0 = faces[i].v0;
		int V1 = faces[i].v1;
		int V2 = faces[i].v2;
		
		float r0x = nodes[V0].r.x, r1x = nodes[V1].r.x, r2x = nodes[V2].r.x;
		float r0y = nodes[V0].r.y, r1y = nodes[V1].r.y, r2y = nodes[V2].r.y;
		float r0z = nodes[V0].r.z, r1z = nodes[V1].r.z, r2z = nodes[V2].r.z;
		
		float t0x = r0x/float(Nx)*2*M_PI, t1x = r1x/float(Nx)*2*M_PI, t2x = r2x/float(Nx)*2*M_PI;
		float t0y = r0y/float(Ny)*2*M_PI, t1y = r1y/float(Ny)*2*M_PI, t2y = r2y/float(Ny)*2*M_PI;
		float t0z = r0z/float(Nz)*2*M_PI, t1z = r1z/float(Nz)*2*M_PI, t2z = r2z/float(Nz)*2*M_PI;
		
		float p0x = cos(t0x), p1x = cos(t1x), p2x = cos(t2x);
		float q0x = sin(t0x), q1x = sin(t1x), q2x = sin(t2x);		
		float p0y = cos(t0y), p1y = cos(t1y), p2y = cos(t2y);
		float q0y = sin(t0y), q1y = sin(t1y), q2y = sin(t2y);		
		float p0z = cos(t0z), p1z = cos(t1z), p2z = cos(t2z);
		float q0z = sin(t0z), q1z = sin(t1z), q2z = sin(t2z);
		
		float pxave = (p0x + p1x + p2x)/3.0;
		float qxave = (q0x + q1x + q2x)/3.0;		
		float pyave = (p0y + p1y + p2y)/3.0;
		float qyave = (q0y + q1y + q2y)/3.0;		
		float pzave = (p0z + p1z + p2z)/3.0;
		float qzave = (q0z + q1z + q2z)/3.0;
		
		float txave = atan2(-qxave,-pxave) + M_PI;
		float tyave = atan2(-qyave,-pyave) + M_PI;
		float tzave = atan2(-qzave,-pzave) + M_PI;
		
		float xf = float(Nx)*txave/(2*M_PI);
		float yf = float(Ny)*tyave/(2*M_PI);
		float zf = float(Nz)*tzave/(2*M_PI);
		
		//float xf = (r[V0].x + r[V1].x + r[V2].x)/3.0;  // this is w/o PBC's!!!
		//float yf = (r[V0].y + r[V1].y + r[V2].y)/3.0;
		//float zf = (r[V0].z + r[V1].z + r[V2].z)/3.0;
		
		// --------------------------------------
		// find nearest LBM voxel (rounded down)
		// --------------------------------------
		
		int i0 = int(floor(xf));
		int j0 = int(floor(yf));
		int k0 = int(floor(zf));
		
		// --------------------------------------
		// loop over footprint
		// --------------------------------------
		
		for (int kk=k0; kk<=k0+1; kk++) {
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {				
					int ndx = voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = xf - float(ii);
					float ry = yf - float(jj);
					float rz = zf - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					atomicAdd(&G[ndx].x,del*faces[i].norm.x);
					atomicAdd(&G[ndx].y,del*faces[i].norm.y);
					atomicAdd(&G[ndx].z,del*faces[i].norm.z);
				}
			}		
		}		
	}	
}



// --------------------------------------------------------
// IBM3D kernel to extrapolate IBM face normal to 
// LBM lattice
// --------------------------------------------------------

__global__ void test_interface_normal_poisson_IBM3D(
	float3* G,
	int Nx,
	int Ny,
	int Nz,
	int nVoxels)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nVoxels) {
		
		// --------------------------------------
		// get xyz indices
		// --------------------------------------
		
		int xi = i%Nx;
		int yi = (i/Nx)%Ny;
		int zi = i/(Nx*Ny);
		
		// --------------------------------------
		// get positions of interface assuming
		// a sphere capsule with radius = 6
		// --------------------------------------
		
		float distx = float(xi) - float(Nx/2);
		float disty = float(yi) - float(Ny/2);
		float distz = float(zi) - float(Nz/2);
		float r = sqrt(distx*distx + disty*disty + distz*distz);
		
		if (r > 5.1 && r < 6.1) {
			G[i].x = distx/r;
			G[i].y = disty/r;
			G[i].z = distz/r;
		}
				
	}	
}



// --------------------------------------------------------
// IBM3D kernel to calculate rhs of the poisson equation
// --------------------------------------------------------

__global__ void calculate_rhs_poisson_IBM3D(
	float3* G,
	cufftComplex* rhs,
	int nvoxels,
	int Nx,
	int Ny,
	int Nz)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nvoxels) {
		
		// --------------------------------------
		// get xyz indices
		// --------------------------------------
		
		int xi = i%Nx;
		int yi = (i/Nx)%Ny;
		int zi = i/(Nx*Ny); 
		
		// --------------------------------------
		// calculate divergence of G
		// --------------------------------------
		
		int iup = voxel_ndx(xi+1,yi,zi,Nx,Ny,Nz);
		int idn = voxel_ndx(xi-1,yi,zi,Nx,Ny,Nz);
		int jup = voxel_ndx(xi,yi+1,zi,Nx,Ny,Nz);
		int jdn = voxel_ndx(xi,yi-1,zi,Nx,Ny,Nz);
		int kup = voxel_ndx(xi,yi,zi+1,Nx,Ny,Nz);
		int kdn = voxel_ndx(xi,yi,zi-1,Nx,Ny,Nz);
		float dGxdx = (G[iup].x - G[idn].x)/2.0;  // dx=1
		float dGydy = (G[jup].y - G[jdn].y)/2.0;  // dy=1
		float dGzdz = (G[kup].z - G[kdn].z)/2.0;  // dz=1
		rhs[i].x = dGxdx + dGydy + dGzdz;
		rhs[i].y = 0.0;
				
	}	
}



// --------------------------------------------------------
// IBM3D kernel to solve the poisson equation in 
// Fourier space
// --------------------------------------------------------

__global__ void solve_poisson_inplace(
	cufftComplex* rhs,
	float* kx,
	float* ky,
	float* kz, 
	int Nx,
	int Ny,
	int Nz)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i<Nx*Ny*Nz) {
		int xi = i%Nx;
		int yi = (i/Nx)%Ny;
		int zi = i/(Nx*Ny); 
		float k2 = kx[xi]*kx[xi] + ky[yi]*ky[yi] + kz[zi]*kz[zi];
		if (i==0) k2 = 1.0f;		
		rhs[i].x = -rhs[i].x/k2;
		rhs[i].y = -rhs[i].y/k2;
	}
}



// --------------------------------------------------------
// IBM3D kernel to change solution from complex to float
// --------------------------------------------------------

__global__ void complex2real(
	cufftComplex* uc,
	float* u, 
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i<nVoxels) {
		// divide by number of elements to recover value
		u[i] = uc[i].x/((float)nVoxels);
	}
}



// --------------------------------------------------------
// IBM3D kernel to rescale indicator array
// --------------------------------------------------------

__global__ void rescale_indicator_array(
	float* u,
	int nVoxels)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i<nVoxels) {
		float ui = u[i];
		ui *= -1.0;
		// 1.6 is typical max value (depends on interface spread function)
		if (ui < 0.0) ui = 0.0;
		if (ui > 1.6) ui = 1.6;
		ui /= 1.6;
		u[i] = ui;   // value is now scaled between 0 and 1
	}
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