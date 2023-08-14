
# include "kernels_poisson_ibm3D.cuh"



// --------------------------------------------------------
// IBM3D kernel to extrapolate IBM face normal to 
// LBM lattice
// --------------------------------------------------------

__global__ void extrapolate_interface_normal_poisson_IBM3D(
	float3* r,
	float3* G,
	int Nx,
	int Ny,
	int Nz,
	int nFaces,
	triangle* faces)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nFaces) {
		
		// --------------------------------------
		// center of mass of face
		// --------------------------------------
		
		int V0 = faces[i].v0;
		int V1 = faces[i].v1;
		int V2 = faces[i].v2;
		float xf = (r[V0].x + r[V1].x + r[V2].x)/3.0;
		float yf = (r[V0].y + r[V1].y + r[V2].y)/3.0;
		float zf = (r[V0].z + r[V1].z + r[V2].z)/3.0;
		
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