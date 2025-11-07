# include "kernels_rigids_ibm3D.cuh"
# include <stdio.h>



// --------------------------------------------------------
// IBM3D kernel to zero node forces on a rigid-body:
// --------------------------------------------------------

__global__ void zero_node_forces_rigid_IBM3D(
	rigidnode* nodes,	
	int nNodes)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		nodes[i].f = make_float3(0.0f,0.0f,0.0f);
	}
}



// --------------------------------------------------------
// IBM3D kernel to zero rigid-body forces, torques:
// --------------------------------------------------------

__global__ void zero_rigid_forces_torques_IBM3D(
	rigid* bodies,	
	int nBodies)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBodies) {
		bodies[i].f = make_float3(0.0f,0.0f,0.0f);
		bodies[i].t = make_float3(0.0f,0.0f,0.0f);
	}
}



// --------------------------------------------------------
// IBM3D kernel to sum the forces, torques, and moments of
// inertia for the rigid-body particles:
// --------------------------------------------------------

__global__ void sum_rigid_forces_torques_IBM3D(
	rigidnode* nodes,
	rigid* bodies,
	int nNodes)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		int rigidID = nodes[i].cellID;
		float3 force = nodes[i].f;
		float3 torque = cross(nodes[i].delta,nodes[i].f);
		// add up forces
		atomicAdd(&bodies[rigidID].f.x,force.x);
		atomicAdd(&bodies[rigidID].f.y,force.y);
		atomicAdd(&bodies[rigidID].f.z,force.z);
		// add up torques
		atomicAdd(&bodies[rigidID].t.x,torque.x);
		atomicAdd(&bodies[rigidID].t.y,torque.y);
		atomicAdd(&bodies[rigidID].t.z,torque.z);
	}
}



// --------------------------------------------------------
// IBM3D enforce a maximum rigid-body force & torque:
// --------------------------------------------------------

__global__ void enforce_max_rigid_force_torque_IBM3D(
	rigid* bodies,
	float fmax,
	float tmax,
	int nBodies)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBodies) {
		float fi = length(bodies[i].f);
		float ti = length(bodies[i].t);
		if (fi > fmax) bodies[i].f *= (fmax/fi);
		if (ti > tmax) bodies[i].t *= (tmax/ti);
	}
}



// --------------------------------------------------------
// IBM3D rigid-body update kernel:
// --------------------------------------------------------

__global__ void update_rigid_position_orientation_IBM3D(
	rigid* bodies,
	float dt,
	int nBodies)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
		
	if (i < nBodies) {
		
		// update translational position and velocity to time "t + dt":
		
		float dtm = dt/bodies[i].mass;
		bodies[i].vel += dtm*bodies[i].f;
		bodies[i].com += dt*bodies[i].vel;
				
		// update angular momentum to time "t" (just a placeholder value):
		
		float dt2 = dt/2.0;
		float3 Li = bodies[i].L + dt2*bodies[i].t;
		
		// obtain rotation matrix at time "t":
		
		tensor A = bodies[i].q.get_rot_matrix();
		
		// convert angular momentum to angular velocity in body-fixed frame:
		
		float3 omegaB = (A*Li)/bodies[i].I;
		
		// update quaternion to time "t + dt/2":
		
		quaternion q_half = bodies[i].q;
		q_half.update(dt2,omegaB);
		q_half.normalize();
				
		// obtain rotation matrix at time "t + dt/2":
		
		A = q_half.get_rot_matrix();
		
		// update angular momentum from time "t - dt/2" to time "t + dt/2":
		
		bodies[i].L += dt*bodies[i].t;
		
		// convert angular momentum to angular velocity in body-fixed frame:
		
		omegaB = (A*bodies[i].L)/bodies[i].I;
		
		// update quaternion to time "t + dt":
		
		bodies[i].q.update(dt,omegaB);
		bodies[i].q.normalize();
		
		// update rigid-body angular velocity in space frame at time "t + dt/2":
		
		bodies[i].omega = transpose(A)*omegaB;
				
	}
}



// --------------------------------------------------------
// IBM3D rigid-node update position kernel:
// --------------------------------------------------------

__global__ void update_node_positions_velocities_rigids_IBM3D(
	rigidnode* nodes,
	rigid* bodies,
	int nNodes)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		int cID = nodes[i].cellID;
		tensor R = bodies[cID].q.get_rot_matrix();
		nodes[i].r = bodies[cID].com + R*nodes[i].delta;   // need to check rotation matrix calculation!!!!!!
		nodes[i].v = bodies[cID].vel + cross(bodies[cID].omega,nodes[i].delta);
	}
}



// --------------------------------------------------------
// IBM3D kernel to unwrap node coordinates.  Here, the
// nodes of a rigid-body are brought back close to the rigid's 
// center-of-mass.  This is done to avoid complications with
// PBCs:
// --------------------------------------------------------

__global__ void unwrap_node_coordinates_rigid_IBM3D(
	rigidnode* nodes,
	rigid* bodies,
	float3 Box,
	int3 pbcFlag,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		int rigidID = nodes[i].cellID;
		float3 rij = bodies[rigidID].com - nodes[i].r;
		float3 adjust = roundf(rij/Box)*Box*pbcFlag;
		nodes[i].r = nodes[i].r + adjust;   // PBC's
	}
}



// --------------------------------------------------------
// IBM3D kernel to wrap node coordinates for PBCs:
// --------------------------------------------------------

__global__ void wrap_node_coordinates_rigid_IBM3D(
	rigidnode* nodes,
	float3 Box,
	int3 pbcFlag,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {	
		float3 adjust = floorf(nodes[i].r/Box)*Box*pbcFlag;
		nodes[i].r = nodes[i].r - adjust;   // PBC's
	}
}



// --------------------------------------------------------
// IBM3D kernel to wrap rigid-body COM coordinates for PBCs:
// --------------------------------------------------------

__global__ void wrap_rigid_coordinates_IBM3D(
	rigid* bodies,
	float3 Box,
	int3 pbcFlag,
	int nBodies)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBodies) {	
		bodies[i].com = bodies[i].com - floorf(bodies[i].com/Box)*Box*pbcFlag;		
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void rigid_node_wall_forces_ydir_IBM3D(
	rigidnode* nodes,
	float3 Box,
	float repA,
	float repD,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		const float d = repD;
		const float A = repA;
		const float yi = nodes[i].r.y;
		// bottom wall
		if (yi < d) {
			const float force = A/pow(yi,2) - A/pow(d,2);
			nodes[i].f.y += force;
			if (yi < 0.0001) nodes[i].r.y = 0.0001;
		}
		// top wall
		else if (yi > (Box.y-1.0)-d) {
			const float bmyi = (Box.y-1.0) - yi;
			const float force = A/pow(bmyi,2) - A/pow(d,2);
			nodes[i].f.y -= force;
			if (yi > Box.y-1.0001) nodes[i].r.y = Box.y-1.0001;
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void rigid_node_wall_forces_zdir_IBM3D(
	rigidnode* nodes,
	float3 Box,
	float repA,
	float repD,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		const float d = repD;
		const float A = repA;
		const float zi = nodes[i].r.z;
		// bottom wall
		if (zi < d) {
			const float force = A/pow(zi,2) - A/pow(d,2);
			nodes[i].f.z += force;
			if (zi < 0.0001) nodes[i].r.z = 0.0001;
		}
		// top wall
		else if (zi > (Box.z-1.0)-d) {
			const float bmzi = (Box.z-1.0) - zi;
			const float force = A/pow(bmzi,2) - A/pow(d,2);
			nodes[i].f.z -= force;
			if (zi > Box.z-1.0001) nodes[i].r.z = Box.z-1.0001;
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void rigid_node_wall_forces_ydir_zdir_IBM3D(
	rigidnode* nodes,
	float3 Box,
	float repA,
	float repD,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		const float d = repD;
		const float A = repA;
		const float yi = nodes[i].r.y;
		const float zi = nodes[i].r.z;
		// bottom wall
		if (yi < d) {
			const float force = A/pow(yi,2) - A/pow(d,2);
			nodes[i].f.y += force;
			if (yi < 0.0001) nodes[i].r.y = 0.0001;
		}
		// top wall
		else if (yi > (Box.y-1.0)-d) {
			const float bmyi = (Box.y-1.0) - yi;
			const float force = A/pow(bmyi,2) - A/pow(d,2);
			nodes[i].f.y -= force;
			if (yi > Box.y-1.0001) nodes[i].r.y = Box.y-1.0001;
		}
		// back wall
		if (zi < d) {
			const float force = A/pow(zi,2) - A/pow(d,2);
			nodes[i].f.z += force;
			if (zi < 0.0001) nodes[i].r.z = 0.0001;
		}
		// front wall
		else if (zi > (Box.z-1.0)-d) {
			const float bmzi = (Box.z-1.0) - zi;
			const float force = A/pow(bmzi,2) - A/pow(d,2);
			nodes[i].f.z -= force;
			if (zi > Box.z-1.0001) nodes[i].r.z = Box.z-1.0001;
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void rigid_node_wall_forces_cylinder_IBM3D(
	rigidnode* nodes,
	float3 Box,
	float Rad,
	float repA,
	float repD,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		const float d = repD;
		const float A = repA;
		const float ymid = (Box.y-1.0)/2.0;
		const float zmid = (Box.z-1.0)/2.0;
		const float yi = nodes[i].r.y - ymid;  // distance to channel centerline
		const float zi = nodes[i].r.z - zmid;  // "                            "
		const float ri = sqrt(yi*yi + zi*zi);
		// radial wall		
		if (ri > Rad - d) {
			const float bmri = Rad - ri;
			const float force = A/pow(bmri,2) - A/pow(d,2);
			nodes[i].f.y -= force*(yi/ri);
			nodes[i].f.z -= force*(zi/ri);
		}				
	}
}



// --------------------------------------------------------
// IBM3D kernel to interpolate the gradient of the velocity
// field at the rod position. 
// --------------------------------------------------------

__global__ void hydrodynamic_force_rigid_node_IBM3D(
	rigidnode* nodes,
	float* fxLBM,
	float* fyLBM,
	float* fzLBM,
	float* uLBM,
	float* vLBM,
	float* wLBM,
	float dt,
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
					int ndx = rigid_voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = nodes[i].r.x - float(ii);
					float ry = nodes[i].r.y - float(jj);
					float rz = nodes[i].r.z - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					vxLBMi += del*uLBM[ndx];
					vyLBMi += del*vLBM[ndx];
					vzLBMi += del*wLBM[ndx];				
				}
			}
		}
		
		// --------------------------------------
		// calculate hydrodynamic forces & add them
		// to IBM node forces:
		// --------------------------------------
						
		float fx = 0.01*(vxLBMi - nodes[i].v.x)/dt;
		float fy = 0.01*(vyLBMi - nodes[i].v.y)/dt;
		float fz = 0.01*(vzLBMi - nodes[i].v.z)/dt;
		nodes[i].f.x += fx;
		nodes[i].f.y += fy;
		nodes[i].f.z += fz;
		
		// --------------------------------------
		// distribute the !negative! of the 
		// hydrodynamic bead force to the LBM
		// fluid:
		// --------------------------------------
		
		for (int kk=k0; kk<=k0+1; kk++) {
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {				
					int ndx = rigid_voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = nodes[i].r.x - float(ii);
					float ry = nodes[i].r.y - float(jj);
					float rz = nodes[i].r.z - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					atomicAdd(&fxLBM[ndx],-del*fx);
					atomicAdd(&fyLBM[ndx],-del*fy);
					atomicAdd(&fzLBM[ndx],-del*fz);				
				}
			}
		}		
	}	
}



// --------------------------------------------------------
// IBM3D kernel to assign nodes to bins:
// --------------------------------------------------------

__global__ void build_bin_lists_for_rigid_nodes_IBM3D(
	rigidnode* nodes,
	bindata bins,
	int nNodes)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {		
		
		// -------------------------------
		// calculate bin ID:
		// -------------------------------
		
		int binID = int(floor(nodes[i].r.x/bins.sizeBins))*bins.numBins.z*bins.numBins.y +  
			        int(floor(nodes[i].r.y/bins.sizeBins))*bins.numBins.z +
		            int(floor(nodes[i].r.z/bins.sizeBins));		
						
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
// IBM3D kernel to calculate nonbonded node interactions
// using the bin lists:
// --------------------------------------------------------

__global__ void nonbonded_rigid_node_interactions_IBM3D(
	rigidnode* nodes,
	bindata bins,
	float repA,
	float repD,
	int nNodes,
	float3 Box,	
	int3 pbcFlag)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {		
		
		// -------------------------------
		// calculate bin ID:
		// -------------------------------
		
		int binID = int(floor(nodes[i].r.x/bins.sizeBins))*bins.numBins.z*bins.numBins.y +  
			        int(floor(nodes[i].r.y/bins.sizeBins))*bins.numBins.z +
		            int(floor(nodes[i].r.z/bins.sizeBins));		
		
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
			if (nodes[i].cellID == nodes[j].cellID) continue;
			pairwise_rigid_node_interaction_forces_WCA(i,j,repA,repD,nodes,Box,pbcFlag);			
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
				if (nodes[i].cellID == nodes[j].cellID) continue;				
				pairwise_rigid_node_interaction_forces_WCA(i,j,repA,repD,nodes,Box,pbcFlag);			
			}
		}
				
	}
}













// **********************************************************************************************
// Miscellaneous kernels and functions
// **********************************************************************************************













// --------------------------------------------------------
// IBM3D kernel to calculate i-j force:
// Weeks-Chandler-Anderson potential
// --------------------------------------------------------

__device__ inline void pairwise_rigid_node_interaction_forces_WCA(
	const int i, 
	const int j,
	const float repA,
	const float repD,
	rigidnode* nodes,
	float3 Box,
	int3 pbcFlag)
{
	float3 rij = nodes[i].r - nodes[j].r;
	rij -= roundf(rij/Box)*Box*pbcFlag;  // PBC's	
	const float r = length(rij);
	if (r < repD) {
		float sig = 0.8909*repD;  // this ensures F=0 is at cutoff
		float eps = 0.001;
		float sigor = sig/r;
		float sigor6 = sigor*sigor*sigor*sigor*sigor*sigor;
		float sigor12 = sigor6*sigor6;
		float force = 24.0*eps*(2*sigor12 - sigor6)/r/r;
		nodes[i].f += force*rij;
	} 	
}



// --------------------------------------------------------
// IBM3D kernel to determine 1D index from 3D indices:
// --------------------------------------------------------

__device__ inline int rigid_voxel_ndx(
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







