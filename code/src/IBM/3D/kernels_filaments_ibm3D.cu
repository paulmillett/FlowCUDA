
# include "kernels_filaments_ibm3D.cuh"
# include <stdio.h>



// --------------------------------------------------------
// IBM3D bead update kernel:
// --------------------------------------------------------

__global__ void update_bead_position_euler_IBM3D(
	bead* beads,
	float dt,
	float m,	
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		beads[i].v += dt*beads[i].f/m;
		beads[i].r += dt*beads[i].v;
	}
}



// --------------------------------------------------------
// IBM3D bead update kernel.  Here, it is assumed that 
// there is no inertia:
// --------------------------------------------------------

__global__ void update_bead_position_euler_overdamped_IBM3D(
	bead* beads,
	float dt,
	float Mob,	
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		beads[i].v += Mob*beads[i].f;
		beads[i].r += dt*beads[i].v;
	}
}



// --------------------------------------------------------
// IBM3D bead update kernel:
// --------------------------------------------------------

__global__ void update_bead_position_verlet_1_IBM3D(
	bead* beads,
	float dt,
	float m,	
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		beads[i].r += beads[i].v*dt + 0.5*dt*dt*(beads[i].f/m);
		beads[i].v += 0.5*dt*(beads[i].f/m);
	}
}



// --------------------------------------------------------
// IBM3D bead update kernel:
// --------------------------------------------------------

__global__ void update_bead_position_verlet_2_IBM3D(
	bead* beads,
	float dt,
	float m,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		beads[i].v += 0.5*dt*(beads[i].f/m);
	}
}



// --------------------------------------------------------
// IBM3D bead update kernel (here, viscous drag is included
// in the verlet algorithm)
// --------------------------------------------------------

__global__ void update_bead_position_verlet_1_drag_IBM3D(
	bead* beads,
	float dt,
	float m,
	float gam,
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		beads[i].r += beads[i].v*dt + 0.5*dt*dt*(beads[i].f/m - gam*beads[i].v/m);
		beads[i].v += 0.5*dt*(beads[i].f/m - gam*beads[i].v/m);
	}
}



// --------------------------------------------------------
// IBM3D bead update kernel (here, viscous drag is included
// in the verlet algorithm)
// --------------------------------------------------------

__global__ void update_bead_position_verlet_2_drag_IBM3D(
	bead* beads,
	float dt,
	float m,
	float gam,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		beads[i].v = (beads[i].v + 0.5*dt*beads[i].f/m)/(1.0 + 0.5*dt*gam/m);
	}
}



// --------------------------------------------------------
// IBM3D kernel to zero bead forces:
// --------------------------------------------------------

__global__ void zero_bead_forces_IBM3D(
	bead* beads,	
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		beads[i].f = make_float3(0.0f,0.0f,0.0f);
		beads[i].fdip = make_float3(0.0f,0.0f,0.0f);
	}
}



// --------------------------------------------------------
// IBM3D zero the bead velocities and forces:
// --------------------------------------------------------

__global__ void zero_bead_velocities_forces_IBM3D(
	bead* beads,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		beads[i].v = make_float3(0.0f,0.0f,0.0f);
		beads[i].f = make_float3(0.0f,0.0f,0.0f);
		beads[i].fdip = make_float3(0.0f,0.0f,0.0f);
	}
}



// --------------------------------------------------------
// IBM3D enforce a maximum bead force:
// --------------------------------------------------------

__global__ void enforce_max_bead_force_IBM3D(
	bead* beads,
	float fmax,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		float fi = length(beads[i].f);
		if (fi > fmax) {
			beads[i].f *= (fmax/fi);
		}
	}
}



// --------------------------------------------------------
// IBM3D add a drag force:
// --------------------------------------------------------

__global__ void add_drag_force_to_bead_IBM3D(
	bead* beads,
	float c,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		beads[i].f -= c*beads[i].v;
	}
}



// --------------------------------------------------------
// IBM3D kernel to compute force on beads due to bonded
// neighbors
// --------------------------------------------------------

__global__ void compute_bead_force_spring_IBM3D(
	bead* beads,
	edgefilam* edges,
	filament* filams,
	int nEdges)
{
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nEdges) {
		// calculate edge length:
		int B0 = edges[i].b0;
		int B1 = edges[i].b1;
		float3 r0 = beads[B0].r; 
		float3 r1 = beads[B1].r;
		float3 r01 = r1 - r0;
		float edgeL = length(r01);		
		// calculate edge stretching force:
		int fID = beads[B0].filamID;
		float length0 = edges[i].length0;
		float ForceMag = filams[fID].ks*(edgeL-length0);
		r01 /= edgeL;  // normalize vector		
		add_force_to_bead(B0,beads, ForceMag*r01);
		add_force_to_bead(B1,beads,-ForceMag*r01);	
	}
}



// --------------------------------------------------------
// IBM3D kernel to compute force on beads due to bonded
// neighbors using the FENE model of Peterson et al. 
// Nature Comm. 12:7247 (2021)
// --------------------------------------------------------

__global__ void compute_bead_force_FENE_IBM3D(
	bead* beads,
	edgefilam* edges,
	filament* filams,
	float delta,
	int nEdges)
{
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nEdges) {
		// calculate edge length:
		int B0 = edges[i].b0;
		int B1 = edges[i].b1;
		float3 r0 = beads[B0].r; 
		float3 r1 = beads[B1].r;
		float3 r01 = r1 - r0;
		float edgeL = length(r01);		
		// calculate edge stretching force:
		int fID = beads[B0].filamID;
		float numer = edgeL - edges[i].length0;
		float denom = 1.0 - numer*numer/delta/delta;
		float ForceMag = filams[fID].ks*numer/denom;
		r01 /= edgeL;  // normalize vector
		add_force_to_bead(B0,beads, ForceMag*r01);
		add_force_to_bead(B1,beads,-ForceMag*r01);		
	}
}



// --------------------------------------------------------
// IBM3D kernel to compute force on beads due to bending
// Note: this approach matches that used by LAMMPS in the
//       "angle_harmonic.cpp" file
// --------------------------------------------------------

__global__ void compute_bead_force_bending_IBM3D(
	bead* beads,
	filament* filams,
	int nBeads)
{		
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nBeads) {
		// get neighboring beads:
		int B0 = i-1;
		int B1 = i;
		int B2 = i+1;
		// if this bead is the first or last bead overall, move along:
		if (B1 == 0 || B1 == nBeads-1) return;
		// if this bead is on the end of a filament, move along:
		if (beads[B0].filamID != beads[B1].filamID) return;  // beads are not bonded
		if (beads[B2].filamID != beads[B1].filamID) return;  // beads are not bonded
		// bond vectors:
		float3 r10 = beads[B0].r - beads[B1].r;
		float3 r12 = beads[B2].r - beads[B1].r;
		float r1 = length(r10);
		float r2 = length(r12);
		// cosine of angle:
		float c = dot(r10,r12)/(r1*r2);
		if (c > 1.0) c = 1.0;
		if (c < -1.0) c = -1.0;
		// sine of angle:
		float s = sqrt(1.0 - c*c);
		if (s < 0.001) s = 0.001;   // LAMMPS uses variable 'SMALL'
		s = 1.0/s;
		// forces:
		int f = beads[B1].filamID;
		float kb = filams[f].kb;
		float dtheta = acos(c) - M_PI; //3.14159265359;
		float a = -2.0*kb*dtheta*s;
		float a11 = a*c/(r1*r1);
		float a12 = -a/(r1*r2);
		float a22 = a*c/(r2*r2);
		float3 F0 = a11*r10 + a12*r12;
		float3 F2 = a22*r12 + a12*r10;
		add_force_to_bead(B0,beads,F0);
		add_force_to_bead(B2,beads,F2);
		beads[i].f -= F0+F2;			
	}	
}



// --------------------------------------------------------
// IBM3D kernel to compute self-propulsion force
// Note: here, I am using the model of Duman et al. Soft
// Matter 14:4483 (2018) which assumes 'fp' is a force
// per unit length along the filament.
//
// Here, also, the force along the edge is added as a 
// dipole force that will be distributed to the fluid 
// following Mahajan & Saintillan PRE 105:014608 (2022) 
// --------------------------------------------------------

__global__ void compute_propulsion_force_IBM3D(
	bead* beads,
	edgefilam* edges,
	filament* filams,
	int nEdges)
{		
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nEdges) {
		// vector pointing to the lower-index bead (toward the headBead)
		int B0 = edges[i].b0;
		int B1 = edges[i].b1;
		float3 r01 = beads[B0].r - beads[B1].r;
		// calculate propulsion force:
		int fID = beads[B0].filamID;
		float ForceMag = filams[fID].fp;
		// add propulsion force to bead
		add_force_to_bead(B0,beads,ForceMag*r01);
		// add force dipole to be distributed to fluid
		add_force_dipole_to_bead(B0,beads,ForceMag*r01);
		add_force_dipole_to_bead(B1,beads,-ForceMag*r01);		
	}
}



// --------------------------------------------------------
// IBM3D kernel to compute self-propulsion velocity
// assuming over-damped motion: u = F/gam
// --------------------------------------------------------

__global__ void compute_propulsion_velocity_IBM3D(
	bead* beads,
	edgefilam* edges,
	filament* filams,
	int nEdges)
{		
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nEdges) {
		// vector pointing to the lower-index bead (toward the headBead)
		int B0 = edges[i].b0;
		int B1 = edges[i].b1;
		float3 r01 = normalize(beads[B0].r - beads[B1].r);
		//float3 r01 = beads[B0].r - beads[B1].r;
		// calculate propulsion force:
		int fID = beads[B0].filamID;
		float velMag = filams[fID].up;
		add_velocity_to_bead(B0,beads,velMag*r01);	
	}
}



// --------------------------------------------------------
// IBM3D kernel to compute thermal force 
// --------------------------------------------------------

__global__ void compute_thermal_force_IBM3D(
	bead* beads,
	curandState* state,
	float pref,
	int nBeads)
{		
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nBeads) {
		float r1 = curand_uniform(&state[i]);
		float r2 = curand_uniform(&state[i]);
		float r3 = curand_uniform(&state[i]);		
		beads[i].f.x += pref*(r1-0.5);
		beads[i].f.y += pref*(r2-0.5);
		beads[i].f.z += pref*(r3-0.5);
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate viscous force due to velocity
// difference between IBM bead and LBM fluid
// --------------------------------------------------------

__global__ void viscous_force_velocity_difference_bead_IBM3D(
	bead* beads,
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
					int ndx = bead_voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = beads[i].r.x - float(ii);
					float ry = beads[i].r.y - float(jj);
					float rz = beads[i].r.z - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					vxLBMi += del*uLBM[ndx];
					vyLBMi += del*vLBM[ndx];
					vzLBMi += del*wLBM[ndx];
					// extrapolate elastic forces to LBM fluid:
					atomicAdd(&fxLBM[ndx],del*beads[i].fdip.x);
					atomicAdd(&fyLBM[ndx],del*beads[i].fdip.y);
					atomicAdd(&fzLBM[ndx],del*beads[i].fdip.z);					
				}
			}
		}
		
		// --------------------------------------
		// calculate friction forces & add them
		// to IBM node:
		// --------------------------------------
		
		float vfx = gam*(vxLBMi - beads[i].v.x);
		float vfy = gam*(vyLBMi - beads[i].v.y);
		float vfz = gam*(vzLBMi - beads[i].v.z);
		beads[i].f.x += vfx;
		beads[i].f.y += vfy;
		beads[i].f.z += vfz;
				
	}	
}



// --------------------------------------------------------
// IBM3D kernel to calculate hydrodynamic force between
// IBM bead and LBM fluid
// --------------------------------------------------------

__global__ void hydrodynamic_force_bead_fluid_IBM3D(
	bead* beads,
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
					int ndx = bead_voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
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
		// calculate drag forces & add them
		// to IBM bead forces:
		// --------------------------------------
		
		float vfx = gam*(vxLBMi - beads[i].v.x);
		float vfy = gam*(vyLBMi - beads[i].v.y);
		float vfz = gam*(vzLBMi - beads[i].v.z);
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
					int ndx = bead_voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = beads[i].r.x - float(ii);
					float ry = beads[i].r.y - float(jj);
					float rz = beads[i].r.z - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					atomicAdd(&fxLBM[ndx],-del*beads[i].f.x);
					atomicAdd(&fyLBM[ndx],-del*beads[i].f.y);
					atomicAdd(&fzLBM[ndx],-del*beads[i].f.z);				
				}
			}
		}
				
	}	
}



// --------------------------------------------------------
// IBM3D kernel to unwrap bead coordinates.  Here, the
// beads of a filament are brought back close to the bead's 
// headBead.  This is done to avoid complications with
// PBCs:
// --------------------------------------------------------

__global__ void unwrap_bead_coordinates_IBM3D(
	bead* beads,
	filament* filams,
	float3 Box,
	int3 pbcFlag,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		int f = beads[i].filamID;
		int j = filams[f].headBead;
		float3 rij = beads[j].r - beads[i].r;		
		beads[i].r = beads[i].r + roundf(rij/Box)*Box*pbcFlag; // PBC's
	}
}



// --------------------------------------------------------
// IBM3D kernel to wrap bead coordinates for PBCs:
// --------------------------------------------------------

__global__ void wrap_bead_coordinates_IBM3D(
	bead* beads,
	float3 Box,
	int3 pbcFlag,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {	
		beads[i].r = beads[i].r - floorf(beads[i].r/Box)*Box*pbcFlag;		
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void bead_wall_forces_ydir_IBM3D(
	bead* beads,
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
	bead* beads,
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
	bead* beads,
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
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void push_beads_into_sphere_IBM3D(
	bead* beads,
	float xs,
	float ys,
	float zs,
	float rs,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		float3 sphere = make_float3(xs,ys,zs);
		float3 ris = beads[i].r - sphere;
		float r = length(ris);
		if (r > (rs-1.5)) {
			ris /= r;
			beads[i].f -= 0.0005*ris;
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to build the binMap array:
// --------------------------------------------------------

__global__ void build_binMap_for_beads_IBM3D(
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
					bins.binMap[offst+cnt] = bin_index_for_beads(bx,by,bz,bins.numBins);
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

__global__ void reset_bin_lists_for_beads_IBM3D(
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

__global__ void build_bin_lists_for_beads_IBM3D(
	bead* beads,
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
	bead* beads,
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
			printf("occup = %i \n", occup);
			occup = bins.binMax;
		}
								
		for (int k=offst; k<offst+occup; k++) {
			int j = bins.binMembers[k];
			if (i==j) continue;
			if (beads[i].filamID == beads[j].filamID) continue;
			pairwise_bead_interaction_forces_WCA(i,j,repA,repD,beads,Box,pbcFlag);			
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
				if (beads[i].filamID == beads[j].filamID) continue;				
				pairwise_bead_interaction_forces_WCA(i,j,repA,repD,beads,Box,pbcFlag);			
			}
		}
				
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate nonbonded bead-node interactions
// using the bin lists.  Here, the nodes are from the 
// 'class_capsules_ibm3D' class
// --------------------------------------------------------

__global__ void nonbonded_bead_node_interactions_IBM3D(
	bead* beads,
	node* nodes,
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
		if (occup > bins.binMax) occup = bins.binMax;
								
		for (int k=offst; k<offst+occup; k++) {
			int j = bins.binMembers[k];
			pairwise_bead_node_interaction_forces_WCA(i,j,repA,repD,beads,nodes,Box,pbcFlag);			
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
			// loop over nodes in this bin:
			for (int k=offst; k<offst+occup; k++) {
				int j = bins.binMembers[k];
				pairwise_bead_node_interaction_forces_WCA(i,j,repA,repD,beads,nodes,Box,pbcFlag);		
			}
		}
				
	}
}

	












// **********************************************************************************************
// Miscellaneous kernels and functions
// **********************************************************************************************













// --------------------------------------------------------
// IBM3D kernel to calculate i-j force:
// --------------------------------------------------------

__device__ inline void pairwise_bead_interaction_forces(
	const int i, 
	const int j,
	const float repA,
	const float repD,
	bead* beads,
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
	bead* beads,
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
// NOTE: here 'i' is a bead and 'j' is a node
// --------------------------------------------------------

__device__ inline void pairwise_bead_node_interaction_forces(
	const int i, 
	const int j,
	const float repA,
	const float repD,
	bead* beads,
	node* nodes,
	float3 Box,
	int3 pbcFlag)
{
	float3 rij = beads[i].r - nodes[j].r;
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
// NOTE: here 'i' is a bead and 'j' is a node
// --------------------------------------------------------

__device__ inline void pairwise_bead_node_interaction_forces_WCA(
	const int i, 
	const int j,
	const float repA,
	const float repD,
	bead* beads,
	node* nodes,
	float3 Box,
	int3 pbcFlag)
{
	float3 rij = beads[i].r - nodes[j].r;
	rij -= roundf(rij/Box)*Box*pbcFlag;  // PBC's	
	const float r = length(rij);
	if (r < repD) {
		float delta = 1.0;
		float sig = 0.8909*(repD - delta);  // this ensures F=0 is at cutoff
		float eps = 0.001;
		float rmd = r - delta;
		float sigor = sig/rmd;
		float sigor6 = sigor*sigor*sigor*sigor*sigor*sigor;
		float sigor12 = sigor6*sigor6;
		float force = 24.0*eps*(2*sigor12 - sigor6)/rmd/rmd;
		beads[i].f += force*rij;
	} 	
}



// --------------------------------------------------------
// add force to vertex using atomicAdd:
// --------------------------------------------------------

__device__ inline void add_force_to_bead(
	int i,
	bead* b,
	const float3 g)
{
	atomicAdd(&b[i].f.x,g.x);
	atomicAdd(&b[i].f.y,g.y);
	atomicAdd(&b[i].f.z,g.z);	
}



// --------------------------------------------------------
// add dipole force to vertex using atomicAdd:
// --------------------------------------------------------

__device__ inline void add_force_dipole_to_bead(
	int i,
	bead* b,
	const float3 g)
{
	atomicAdd(&b[i].fdip.x,g.x);
	atomicAdd(&b[i].fdip.y,g.y);
	atomicAdd(&b[i].fdip.z,g.z);	
}



// --------------------------------------------------------
// add force to vertex using atomicAdd:
// --------------------------------------------------------

__device__ inline void add_velocity_to_bead(
	int i,
	bead* b,
	const float3 g)
{
	atomicAdd(&b[i].v.x,g.x);
	atomicAdd(&b[i].v.y,g.y);
	atomicAdd(&b[i].v.z,g.z);	
}



// --------------------------------------------------------
// IBM3D kernel to calculate i-j force:
// --------------------------------------------------------

__device__ inline int bin_index_for_beads(
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



// --------------------------------------------------------
// IBM3D kernel to determine 1D index from 3D indices:
// --------------------------------------------------------

__device__ inline int bead_voxel_ndx(
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
// IBM3D kernel to initialize curand random num. generator:
// --------------------------------------------------------

__global__ void init_curand_IBM3D(
	curandState* state,
	unsigned long seed,
	int nBeads)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nBeads) {
		curand_init(seed,i,0,&state[i]);
	}    
}




