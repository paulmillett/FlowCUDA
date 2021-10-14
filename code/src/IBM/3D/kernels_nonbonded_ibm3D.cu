
# include "kernels_nonbonded_ibm3D.cuh"
# include <stdio.h>



// --------------------------------------------------------
// IBM3D kernel to reset bin arrays:
// --------------------------------------------------------

__global__ void reset_bin_lists_IBM3D(
	int* binOccupancy,
	int* binMembers,
	int binMax,
	int nBins)
{
	// define bin:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBins) {
		
		// -------------------------------
		// reset binOccupancy[] to zero,
		// and binMembers[] array to -1:
		// -------------------------------
		
		binOccupancy[i] = 0;
		int offst = i*binMax;
		for (int k=offst; k<offst+binMax; k++) {
			binMembers[k] = -1;
		}
		
	}	
}



// --------------------------------------------------------
// IBM3D kernel to assign nodes to bins:
// --------------------------------------------------------

__global__ void build_bin_lists_IBM3D(
	float3* vertR,
	int* binOccupancy,
	int* binMembers,	
	int3 numBins,	
	float sizeBins,
	int nNodes,
	int binMax)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {		
		
		// -------------------------------
		// calculate bin ID:
		// -------------------------------
		
		int binID = int(floor(vertR[i].x/sizeBins))*numBins.z*numBins.y +  
			        int(floor(vertR[i].y/sizeBins))*numBins.z +
		            int(floor(vertR[i].z/sizeBins));		
						
		// -------------------------------
		// update the lists:
		// -------------------------------
		
		if (binID >= 0 && binID < numBins.x*numBins.y*numBins.z) {
			atomicAdd(&binOccupancy[binID],1);
			int offst = binID*binMax;
			for (int k=offst; k<offst+binMax; k++) {
				int flag = atomicCAS(&binMembers[k],-1,i); 
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
	float3* vertR,
	float3* vertF,
	int* binOccupancy,
	int* binMembers,
	int* binMap,
	int* cellIDs,
	int3 numBins,	
	float sizeBins,
	int nNodes,
	int binMax,
	int nnbins,
	float3 Box,
	int3 pbcFlag)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {		
		
		// -------------------------------
		// calculate bin ID:
		// -------------------------------
		
		int binID = int(floor(vertR[i].x/sizeBins))*numBins.z*numBins.y +  
			        int(floor(vertR[i].y/sizeBins))*numBins.z +
		            int(floor(vertR[i].z/sizeBins));		
		
		// -------------------------------
		// loop over nodes in the same bin:
		// -------------------------------
				
		int offst = binID*binMax;
		int occup = binOccupancy[binID];
		
		/*
		if (occup > binMax) {
			printf("Warning: linked-list bin has exceeded max capacity.  Occup. # = %i \n",occup);
		}
		*/
				
		for (int k=offst; k<offst+occup; k++) {
			int j = binMembers[k];
			if (i==j) continue;
			if (cellIDs[i]==cellIDs[j]) continue;
			pairwise_interaction_forces(i,j,vertR,vertF,Box,pbcFlag);			
		}
		
		// -------------------------------
		// loop over neighboring bins:
		// -------------------------------
		
        for (int b=0; b<nnbins; b++) {
            // get neighboring bin ID
			int naborbinID = binMap[binID*nnbins + b];
			offst = naborbinID*binMax;
			occup = binOccupancy[naborbinID];
			// loop over nodes in this bin:
			for (int k=offst; k<offst+occup; k++) {
				int j = binMembers[k];
				if (cellIDs[i]==cellIDs[j]) continue;				
				pairwise_interaction_forces(i,j,vertR,vertF,Box,pbcFlag);			
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
	const float3* R,
	float3* F,
	float3 Box,
	int3 pbcFlag)
{
	const float d = 2.0;
	const float A = 2.0;
	float3 rij = R[i] - R[j];
	rij -= roundf(rij/Box)*Box*pbcFlag;  // PBC's	
	const float r = length(rij);
	if (r < d) {
		float force = A/pow(r,2) - A/pow(d,2);
		if (force > 0.2) force = 0.2;
		F[i] += force*(rij/r);
	} 	
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void wall_forces_ydir_IBM3D(
	float3* R,
	float3* F,
	float3 Box,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		const float d = 2.0;
		const float A = 2.0;
		const float yi = R[i].y;
		// bottom wall
		if (yi < d) {
			const float force = A/pow(yi,2) - A/pow(d,2);
			F[i].y += force;
		}
		// top wall
		else if (yi > Box.y-d) {
			const float bmyi = Box.y - yi;
			const float force = A/pow(bmyi,2) - A/pow(d,2);
			F[i].y -= force;
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to build the binMap array:
// --------------------------------------------------------

__global__ void build_binMap_IBM3D(
	int* binMap,
	int3 numBins,
	int nnbins,
	int nBins)
{
	// define bin:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBins) {
	
		// -------------------------------
		// calculate bin's x,y,z coordinates:
		// -------------------------------
				
		int binx = i/(numBins.y*numBins.z);
		int biny = (i/numBins.z)%numBins.y;
		int binz = i%numBins.z;
		
		// -------------------------------
		// determine neighboring bins:
		// -------------------------------
		
		int cnt = 0;
		int offst = i*nnbins;
		
		for (int bx = binx-1; bx < binx+2; bx++) {
			for (int by = biny-1; by < biny+2; by++) {
				for (int bz = binz-1; bz < binz+2; bz++) {
					// do not include current bin
					if (bx==binx && by==biny && bz==binz) continue;
					// bin index of neighbor
					binMap[offst+cnt] = bin_index(bx,by,bz,numBins);
					// update counter
					cnt++;
				}
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

