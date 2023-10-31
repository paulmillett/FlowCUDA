
# include "kernels_nonbonded_ibm3D.cuh"
# include <stdio.h>



// --------------------------------------------------------
// IBM3D kernel to build the binMap array:
// --------------------------------------------------------

__global__ void build_binMap_IBM3D(
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
					bins.binMap[offst+cnt] = bin_index(bx,by,bz,bins.numBins);
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

__global__ void reset_bin_lists_IBM3D(
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
// IBM3D kernel to assign nodes to bins:
// --------------------------------------------------------

__global__ void build_bin_lists_IBM3D(
	node* nodes,
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

__global__ void nonbonded_node_interactions_IBM3D(
	node* nodes,
	cell* cells,
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
		// loop over nodes in the same bin:
		// -------------------------------
				
		int offst = binID*bins.binMax;
		int occup = bins.binOccupancy[binID];
		if (occup > bins.binMax) occup = bins.binMax;
								
		for (int k=offst; k<offst+occup; k++) {
			int j = bins.binMembers[k];
			if (i==j) continue;
			if (nodes[i].cellID == nodes[j].cellID) continue;
			pairwise_interaction_forces(i,j,repA,repD,nodes,cells,Box,pbcFlag);
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
				if (nodes[i].cellID == nodes[j].cellID) continue;			
				pairwise_interaction_forces(i,j,repA,repD,nodes,cells,Box,pbcFlag);			
			}
		}
				
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate nonbonded node-bead interactions
// using the bin lists.  Here, the beads are from the 
// 'class_filaments_ibm3D' class
// --------------------------------------------------------

__global__ void nonbonded_node_bead_interactions_IBM3D(
	node* nodes,
	bead* beads,
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
		// loop over nodes in the same bin:
		// -------------------------------
				
		int offst = binID*bins.binMax;
		int occup = bins.binOccupancy[binID];
		if (occup > bins.binMax) occup = bins.binMax;
								
		for (int k=offst; k<offst+occup; k++) {
			int j = bins.binMembers[k];
			pairwise_node_bead_interaction_forces(i,j,repA,repD,nodes,beads,Box,pbcFlag);
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
				pairwise_node_bead_interaction_forces(i,j,repA,repD,nodes,beads,Box,pbcFlag);	
			}
		}
				
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate i-j force:
// --------------------------------------------------------

__device__ inline void pairwise_interaction_forces(
	const int i, 
	const int j,
	const float repA,
	const float repD,
	node* nodes,
	cell* cells,	
	float3 Box,
	int3 pbcFlag)
{
	float3 rij = nodes[i].r - nodes[j].r;
	rij -= roundf(rij/Box)*Box*pbcFlag;  // PBC's	
	const float r = length(rij);
	if (r < repD) {
		// linear spring force repulsion
		float force = repA - (repA/repD)*r;
		float3 fij = force*(rij/r);
		// WCA force
		//float force = 0.01*pow(0.85/r,12)/r;
		//float3 fij = force*(rij/r);
		
		
		// check if fij is pushing node i away from the c.o.m.
		// of it's cell...  if so, then set force to zero
		// because there is likely cell-cell overlap here
		int cID = nodes[i].cellID;
		float3 ric = nodes[i].r - cells[cID].com;
		ric -= roundf(ric/Box)*Box*pbcFlag;  // PBC's
		float fdir = dot(fij,ric);
		if (fdir > 0.0) fij = make_float3(0.0);
		// add force to node i
		nodes[i].f += fij;
	} 	
}



// --------------------------------------------------------
// IBM3D kernel to calculate i-j force:
// NOTE: here 'i' is a node and 'j' is a bead
// --------------------------------------------------------

__device__ inline void pairwise_node_bead_interaction_forces(
	const int i, 
	const int j,
	const float repA,
	const float repD,
	node* nodes,
	bead* beads,	
	float3 Box,
	int3 pbcFlag)
{
	float3 rij = nodes[i].r - beads[j].r;
	rij -= roundf(rij/Box)*Box*pbcFlag;  // PBC's	
	const float r = length(rij);
	if (r < repD) {
		// linear spring force repulsion
		float force = repA - (repA/repD)*r;
		float3 fij = force*(rij/r);
		nodes[i].f += fij;
	} 	
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void wall_forces_ydir_IBM3D(
	node* nodes,
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

__global__ void wall_forces_zdir_IBM3D(
	node* nodes,
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

__global__ void wall_forces_ydir_zdir_IBM3D(
	node* nodes,
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
// IBM3D kernel to calculate i-j force:
// --------------------------------------------------------

__device__ inline int bin_index(
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


