# include "kernels_rigids_ibm3D.cuh"
# include <stdio.h>



// --------------------------------------------------------
// IBM3D kernel to zero rigid-body forces, torques:
// --------------------------------------------------------

__global__ void zero_rigid_forces_torques_IBM3D(
	rigid* rigids,	
	int nRigids)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nRigids) {
		rigids[i].f = make_float3(0.0f,0.0f,0.0f);
		rigids[i].t = make_float3(0.0f,0.0f,0.0f);
	}
}



// --------------------------------------------------------
// IBM3D enforce a maximum rigid-body force & torque:
// --------------------------------------------------------

__global__ void enforce_max_rigid_force_torque_IBM3D(
	rigid* rigids,
	float fmax,
	float tmax,
	int nRigids)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nRigids) {
		float fi = length(rigids[i].f);
		float ti = length(rigids[i].t);
		if (fi > fmax) rigids[i].f *= (fmax/fi);
		if (ti > tmax) rigids[i].t *= (tmax/ti);
	}
}



// --------------------------------------------------------
// IBM3D rigid-node update position kernel:
// --------------------------------------------------------

__global__ void update_node_positions_rigids_IBM3D(
	rigidnode* nodes,
	rigid* rigids,
	int nNodes)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		int rigidID = nodes[i].cellID;
		tensor R = rigids[rigidID].q.get_rot_matrix();
		nodes[i].r = rigids[rigidID].com + R*nodes[i].delta;   // need to check rotation matrix calculation!!!!!!
	}
}



// --------------------------------------------------------
// IBM3D rigid-body update kernel:
// --------------------------------------------------------

__global__ void update_rigid_position_orientation_IBM3D(
	rigid* rigids,
	float dt,
	int nRigids)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nRigids) {
		
		
		
		// need to add stuff here....
		
		
	}
}



// --------------------------------------------------------
// IBM3D kernel to sum the forces, torques, and moments of
// inertia for the rigid-body particles:
// --------------------------------------------------------

__global__ void sum_rigid_forces_torques_IBM3D(
	rigidnode* nodes,
	rigid* rigids,
	int nNodes)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		int rigidID = nodes[i].cellID;
		float3 force = nodes[i].f;
		float3 torque = cross(nodes[i].delta,nodes[i].f);
		// add up forces
		atomicAdd(&rigids[rigidID].f.x,force.x);
		atomicAdd(&rigids[rigidID].f.y,force.y);
		atomicAdd(&rigids[rigidID].f.z,force.z);
		// add up torques
		atomicAdd(&rigids[rigidID].t.x,torque.x);
		atomicAdd(&rigids[rigidID].t.y,torque.y);
		atomicAdd(&rigids[rigidID].t.z,torque.z);			
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
	rigid* rigids,
	float3 Box,
	int3 pbcFlag,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		int rigidID = nodes[i].cellID;
		float3 rij = rigids[rigidID].com - nodes[i].r;
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
// IBM3D kernel to wrap bead coordinates for PBCs:
// --------------------------------------------------------

__global__ void wrap_rigid_coordinates_IBM3D(
	rigid* rigids,
	float3 Box,
	int3 pbcFlag,
	int nRigids)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nRigids) {	
		rigids[i].com = rigids[i].com - floorf(rigids[i].com/Box)*Box*pbcFlag;		
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


/*
// --------------------------------------------------------
// IBM3D kernel to interpolate the gradient of the velocity
// field at the rod position. 
// --------------------------------------------------------

__global__ void hydrodynamic_force_bead_rod_IBM3D(
	beadrod* beads,
	float* fxLBM,
	float* fyLBM,
	float* fzLBM,
	float* uLBM,
	float* vLBM,
	float* wLBM,
	float dt,
	int nBeadsPerRod,
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
					int ndx = rod_voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
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
		// calculate hydrodynamic forces & add them
		// to IBM bead forces:
		// --------------------------------------
		
		float3 velBead = (beads[i].r - beads[i].rm1)/dt;		
		float fx = (vxLBMi - velBead.x)/dt/float(nBeadsPerRod);
		float fy = (vyLBMi - velBead.y)/dt/float(nBeadsPerRod);
		float fz = (vzLBMi - velBead.z)/dt/float(nBeadsPerRod);
		beads[i].f.x += fx;
		beads[i].f.y += fy;
		beads[i].f.z += fz;
		
		// --------------------------------------
		// distribute the !negative! of the 
		// hydrodynamic bead force to the LBM
		// fluid:
		// --------------------------------------
		
		for (int kk=k0; kk<=k0+1; kk++) {
			for (int jj=j0; jj<=j0+1; jj++) {
				for (int ii=i0; ii<=i0+1; ii++) {				
					int ndx = rod_voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
					float rx = beads[i].r.x - float(ii);
					float ry = beads[i].r.y - float(jj);
					float rz = beads[i].r.z - float(kk);
					float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
					atomicAdd(&fxLBM[ndx],-del*fx);
					atomicAdd(&fyLBM[ndx],-del*fy);
					atomicAdd(&fzLBM[ndx],-del*fz);				
				}
			}
		}		
	}	
}
*/



// --------------------------------------------------------
// IBM3D kernel to assign beads to bins:
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
// IBM3D kernel to calculate nonbonded bead interactions
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










