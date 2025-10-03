
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
// IBM3D kernel to calculate nonbonded node interactions
// using the bin lists:
// --------------------------------------------------------

__global__ void nonbonded_node_lubrication_interactions_IBM3D(
	node* nodes,
	cell* cells,
	bindata bins,
	float Ri,
	float Rj,
	float nu,
	float cutoff,
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
			pairwise_lubrication_forces(i,j,Ri,Rj,cutoff,nu,nodes,cells,Box,pbcFlag);
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
				pairwise_lubrication_forces(i,j,Ri,Rj,cutoff,nu,nodes,cells,Box,pbcFlag);			
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
			pairwise_node_bead_interaction_forces_WCA(i,j,repA,repD,nodes,beads,Box,pbcFlag);
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
				pairwise_node_bead_interaction_forces_WCA(i,j,repA,repD,nodes,beads,Box,pbcFlag);	
			}
		}
				
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate nonbonded node-bead interactions
// using the bin lists.  Here, the beads are from the 
// 'class_rods_ibm3D' class
// --------------------------------------------------------

__global__ void nonbonded_node_bead_rod_interactions_IBM3D(
	node* nodes,
	beadrod* beads,
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
			pairwise_node_bead_interaction_forces_WCA(i,j,repA,repD,nodes,beads,Box,pbcFlag);
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
				pairwise_node_bead_interaction_forces_WCA(i,j,repA,repD,nodes,beads,Box,pbcFlag);	
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
// IBM3D kernel to calculate i-j lubrication force.  The 
// model comes from Ladd & Verberg, Journal of Statistical
// Physics, 104 (2001) 1191.  See Eq. (74).
// --------------------------------------------------------

__device__ inline void pairwise_lubrication_forces(
	const int i, 
	const int j,
	const float Ri,
	const float Rj,
	const float cutoff,
	const float nu,
	node* nodes,
	cell* cells,	
	float3 Box,
	int3 pbcFlag)
{
	float3 rij = nodes[i].r - nodes[j].r;
	rij -= roundf(rij/Box)*Box*pbcFlag;  // PBC's	
	const float r = length(rij);
	if (r < cutoff) {
		// lubrication force:
		float3 uij = rij/r;
		float3 vij = nodes[i].v - nodes[j].v;
		float coeff = (Ri*Rj*Ri*Rj)/(Ri+Rj)/(Ri+Rj);
		float udotv = dot(uij,vij);
		float surfsep = 1.0/r - 1.0/cutoff;  // note, here the ri and rj are not included, because nodes are on surface
		float3 fij = -6.0*M_PI*nu*coeff*uij*udotv*surfsep;	
		
		// add Hertz contact force if separation is less than 0.5dx:
		/*
		if (r < 0.5) {
			float K = 0.01;  // assume coefficient value
			float fcontact = 2.5*K*pow((0.5 - r),1.5);
			fij += fcontact*uij;
		}
		*/
		
			
		// add linear contact force if separation is less than 0.5dx:
		if (r < 0.5) {
			float repA = 0.01;
			float fcontact = repA - (repA/0.5)*r;
			fij += fcontact*uij;			
		}
		
		// if separation goes below 0.05, adjust node i position:
		if (r < 0.05) {
			printf("separation = %f \n", r);
			nodes[i].r = nodes[j].r + 0.05*uij;
		}		
		
		// check if fij is pushing node i away from the c.o.m.
		// of it's cell...  if so, then set force to zero
		// because there is likely cell-cell overlap here
		/*
		int cID = nodes[i].cellID;
		float3 ric = nodes[i].r - cells[cID].com;
		ric -= roundf(ric/Box)*Box*pbcFlag;  // PBC's
		float fdir = dot(fij,ric);
		if (fdir > 0.0) fij = make_float3(0.0);
		*/
					
		// add force to node i:
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
// IBM3D kernel to calculate i-j force:
// NOTE: here 'i' is a node and 'j' is a bead
// --------------------------------------------------------

__device__ inline void pairwise_node_bead_interaction_forces_WCA(
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
		float delta = 1.0;
		float sig = 0.8909*(repD - delta);  // this ensures F=0 is at cutoff
		float eps = 0.001;
		float rmd = r - delta;
		float sigor = sig/rmd;
		float sigor6 = sigor*sigor*sigor*sigor*sigor*sigor;
		float sigor12 = sigor6*sigor6;
		float force = 24.0*eps*(2*sigor12 - sigor6)/rmd/rmd;
		nodes[i].f += force*rij;
	} 	
}



// --------------------------------------------------------
// IBM3D kernel to calculate i-j force:
// NOTE: here 'i' is a node and 'j' is a bead
// --------------------------------------------------------

__device__ inline void pairwise_node_bead_interaction_forces_WCA(
	const int i, 
	const int j,
	const float repA,
	const float repD,
	node* nodes,
	beadrod* beads,	
	float3 Box,
	int3 pbcFlag)
{
	float3 rij = nodes[i].r - beads[j].r;
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
		nodes[i].f += force*rij;
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
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void wall_forces_cylinder_IBM3D(
	node* nodes,
	float3 Box,
	float Rad,
	float repA,
	float repD,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		const float d = 0.5; //repD;           // set this to 0.5dx
		const float A = repA;
		const float ymid = (Box.y-1.0)/2.0;
		const float zmid = (Box.z-1.0)/2.0;
		const float yi = nodes[i].r.y - ymid;  // distance to channel centerline
		const float zi = nodes[i].r.z - zmid;  // "                            "
		const float ri = sqrt(yi*yi + zi*zi);
		// radial wall		
		if (ri > Rad - d) {
			// Hertz contact force:
			const float bmri = Rad - ri;
			const float K = 0.01;  // assume coefficient value
			const float force = 2.5*K*pow((d - bmri),1.5);
			nodes[i].f.y -= force*(yi/ri);
			nodes[i].f.z -= force*(zi/ri);
			
			// soft parabolic contact force:
			/*
			const float bmri = Rad - ri;
			const float force = A/pow(bmri,2) - A/pow(d,2);
			nodes[i].f.y -= force*(yi/ri);
			nodes[i].f.z -= force*(zi/ri);
			*/
			
			// if bead is too close to wall, correct it's position
			const float RadLimit = Rad - 0.0001; 
			if (ri > RadLimit) {
				const float theta = atan2(zi,yi);
				nodes[i].r.y = RadLimit*cos(theta) + ymid;
				nodes[i].r.z = RadLimit*sin(theta) + zmid;
			}
		}				
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces using a 
// lubrication model from Ladd & Verberg, Journal of Statistical
// Physics, 104 (2001) 1191.  See Eq. (74).
// Here, it is assumed that the wall radius is infinite, which
// reduces the term (Ri*Rj)^2/(Ri+Rj)^2 to just Ri^2.  Also,
// the wall velocity is assumed to be zero.  
// --------------------------------------------------------

__global__ void wall_lubrication_forces_cylinder_IBM3D(
	node* nodes,
	float3 Box,
	float chRad,
	float Ri,
	float nu,
	float cutoff,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		const float ymid = (Box.y-1.0)/2.0;
		const float zmid = (Box.z-1.0)/2.0;
		const float yi = nodes[i].r.y - ymid;  // distance to channel centerline
		const float zi = nodes[i].r.z - zmid;  // "                            "
		const float ri = sqrt(yi*yi + zi*zi);
		const float sep = chRad - ri;          // separation to wall
		// radial wall		
		if (sep < cutoff) {
			// lubrication force:			
			const float uy = sep*(yi/ri);   // y-comp of unit vector from node to wall 
			const float uz = sep*(zi/ri);   // z-comp of unit vector from node to wall
			const float vy = nodes[i].v.y;
			const float vz = nodes[i].v.z;
			const float udotv = uy*vy + uz*vz;
			const float force = -6.0*M_PI*nu*Ri*Ri*udotv*(1.0/sep - 1.0/cutoff);
			nodes[i].f.y += force*(yi/ri);
			nodes[i].f.z += force*(zi/ri);	
						
			// if bead is too close to wall, correct it's position:
			const float RadLimit = chRad - 0.0001; 
			if (ri > RadLimit) {
				const float theta = atan2(zi,yi);
				nodes[i].r.y = RadLimit*cos(theta) + ymid;
				nodes[i].r.z = RadLimit*sin(theta) + zmid;
			}
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


