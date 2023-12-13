# include "kernels_rods_ibm3D.cuh"
# include "kernels_filaments_ibm3D.cuh"
# include <stdio.h>



// --------------------------------------------------------
// IBM3D kernel to zero rod forces, torques, moment of
// inertia:
// --------------------------------------------------------

__global__ void zero_rod_forces_torques_moments_IBM3D(
	rod* rods,	
	int nRods)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nRods) {
		rods[i].f = make_float3(0.0f,0.0f,0.0f);
		rods[i].t = make_float3(0.0f,0.0f,0.0f);
		rods[i].Ixx = 0.0;
		rods[i].Iyy = 0.0;
		rods[i].Izz = 0.0;
		rods[i].Ixy = 0.0;
		rods[i].Ixz = 0.0;
		rods[i].Iyz = 0.0;
	}
}



// --------------------------------------------------------
// IBM3D kernel to zero bead forces:
// --------------------------------------------------------

__global__ void zero_bead_forces_IBM3D(
	beadrod* beads,	
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		beads[i].f = make_float3(0.0f,0.0f,0.0f);
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate rod orientation:
// --------------------------------------------------------

__global__ void set_rod_position_orientation_IBM3D(
	beadrod* beads,
	rod* rods,	
	int nRods)
{
	// define rod:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nRods) {
		int BH = rods[i].headBead;
		int BC = rods[i].centerBead;
		float3 orient = normalize(beads[BH].r - beads[BC].r);
		rods[i].r = beads[BC].r;
		rods[i].p = orient;
	}
}



// --------------------------------------------------------
// IBM3D enforce a maximum bead force:
// --------------------------------------------------------

__global__ void enforce_max_bead_force_IBM3D(
	beadrod* beads,
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
// IBM3D enforce a maximum bead force:
// --------------------------------------------------------

__global__ void enforce_max_rod_force_torque_IBM3D(
	rod* rods,
	float fmax,
	float tmax,
	int nRods)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nRods) {
		float fi = length(rods[i].f);
		float ti = length(rods[i].t);
		if (fi > fmax) rods[i].f *= (fmax/fi);
		if (ti > tmax) rods[i].t *= (tmax/ti);
	}
}



// --------------------------------------------------------
// IBM3D bead update kernel:
// --------------------------------------------------------

__global__ void update_bead_positions_rods_IBM3D(
	beadrod* beads,
	rod* rods,
	float L0,	
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		int rodID = beads[i].rodID;
		float offset = float(rods[rodID].centerBead - i);
		beads[i].r = rods[rodID].r + L0*offset*rods[rodID].p;
	}
}



// --------------------------------------------------------
// IBM3D bead update kernel:
// --------------------------------------------------------

__global__ void update_rod_position_orientation_IBM3D(
	rod* rods,
	float dt,
	float gam,	
	int nRods)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nRods) {
		rods[i].r += dt*(rods[i].f/gam);
		rods[i].p += dt*(cross(rods[i].t,rods[i].p));
		rods[i].p = normalize(rods[i].p);
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate rod propulsion force:
// --------------------------------------------------------

__global__ void compute_propulsion_force_rods_IBM3D(
	rod* rods,	
	int nRods)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nRods) {
		rods[i].f += rods[i].fp*rods[i].p;
	}
}



// --------------------------------------------------------
// IBM3D kernel to compute thermal force 
// --------------------------------------------------------

__global__ void compute_thermal_force_IBM3D(
	beadrod* beads,
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
// IBM3D kernel to compute thermal force 
// --------------------------------------------------------

__global__ void compute_thermal_force_torque_rod_IBM3D(
	rod* rods,
	curandState* state,
	float prefT,
	float prefR,
	int nRods)
{		
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nRods) {
		float r1 = curand_uniform(&state[i]);
		float r2 = curand_uniform(&state[i]);
		float r3 = curand_uniform(&state[i]);		
		rods[i].f.x += prefT*(r1-0.5);
		rods[i].f.y += prefT*(r2-0.5);
		rods[i].f.z += prefT*(r3-0.5);
		float r4 = curand_uniform(&state[i]);
		float r5 = curand_uniform(&state[i]);
		float r6 = curand_uniform(&state[i]);
		rods[i].t.x += prefR*(r4-0.5);
		rods[i].t.y += prefR*(r5-0.5);
		rods[i].t.z += prefR*(r6-0.5);
	}
}



// --------------------------------------------------------
// IBM3D kernel to sum the forces, torques, and moments of
// inertia for the rods:
// --------------------------------------------------------

__global__ void sum_rod_forces_torques_moments_IBM3D(
	beadrod* beads,
	rod* rods,
	float m,	
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		int rodID = beads[i].rodID;
		float3 com = beads[rods[rodID].centerBead].r;
		float3 ricom = beads[i].r - com;
		float3 force = beads[i].f;
		float3 torque = cross(ricom,beads[i].f);
		// add up forces
		atomicAdd(&rods[rodID].f.x,force.x);
		atomicAdd(&rods[rodID].f.y,force.y);
		atomicAdd(&rods[rodID].f.z,force.z);
		// add up torques
		atomicAdd(&rods[rodID].t.x,torque.x);
		atomicAdd(&rods[rodID].t.y,torque.y);
		atomicAdd(&rods[rodID].t.z,torque.z);		
		// add up moments
		atomicAdd(&rods[rodID].Ixx,m*(ricom.y*ricom.y + ricom.z*ricom.z));
		atomicAdd(&rods[rodID].Iyy,m*(ricom.x*ricom.x + ricom.z*ricom.z));
		atomicAdd(&rods[rodID].Izz,m*(ricom.x*ricom.x + ricom.y*ricom.y));		
		atomicAdd(&rods[rodID].Ixy,-m*(ricom.x*ricom.y));
		atomicAdd(&rods[rodID].Ixz,-m*(ricom.x*ricom.z));
		atomicAdd(&rods[rodID].Iyz,-m*(ricom.y*ricom.z));		
	}
}



// --------------------------------------------------------
// IBM3D kernel to unwrap bead coordinates.  Here, the
// beads of a rod are brought back close to the rod's 
// centerBead.  This is done to avoid complications with
// PBCs:
// --------------------------------------------------------

__global__ void unwrap_bead_coordinates_rods_IBM3D(
	beadrod* beads,
	rod* rods,
	float3 Box,
	int3 pbcFlag,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		int f = beads[i].rodID;
		int j = rods[f].centerBead;
		float3 rij = beads[j].r - beads[i].r;		
		beads[i].r = beads[i].r + roundf(rij/Box)*Box*pbcFlag; // PBC's
	}
}



// --------------------------------------------------------
// IBM3D kernel to wrap bead coordinates for PBCs:
// --------------------------------------------------------

__global__ void wrap_bead_coordinates_IBM3D(
	beadrod* beads,
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
// IBM3D kernel to wrap bead coordinates for PBCs:
// --------------------------------------------------------

__global__ void wrap_rod_coordinates_IBM3D(
	rod* rods,
	float3 Box,
	int3 pbcFlag,
	int nRods)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nRods) {	
		rods[i].r = rods[i].r - floorf(rods[i].r/Box)*Box*pbcFlag;		
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void bead_wall_forces_ydir_IBM3D(
	beadrod* beads,
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
	beadrod* beads,
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
	beadrod* beads,
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
	beadrod* beads,
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
// IBM3D kernel to assign beads to bins:
// --------------------------------------------------------

__global__ void build_bin_lists_for_beads_IBM3D(
	beadrod* beads,
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
	beadrod* beads,
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
			if (beads[i].rodID == beads[j].rodID) continue;
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
				if (beads[i].rodID == beads[j].rodID) continue;				
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

__global__ void nonbonded_bead_node_interactions_rods_IBM3D(
	beadrod* beads,
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
// Weeks-Chandler-Anderson potential
// --------------------------------------------------------

__device__ inline void pairwise_bead_interaction_forces_WCA(
	const int i, 
	const int j,
	const float repA,
	const float repD,
	beadrod* beads,
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

__device__ inline void pairwise_bead_node_interaction_forces_WCA(
	const int i, 
	const int j,
	const float repA,
	const float repD,
	beadrod* beads,
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
// IBM3D kernel to solve for angular acceleration by
// solving [I][a]=[T], which is a 3x3 matrix problem
// --------------------------------------------------------

__device__ inline float3 solve_angular_acceleration(
	const float Ixx,
	const float Iyy,
	const float Izz,
	const float Ixy,
	const float Ixz,
	const float Iyz,
	const float3 t)
{	
	// first, set up coefficient matrix, which includes moments of inertia and torques:
	float coeff[3][4] = { {Ixx,Ixy,Ixz,t.x}, {Ixy,Iyy,Iyz,t.y}, {Ixz,Iyz,Izz,t.z} };
	
    // matrix d using coeff as given in cramer's rule
    float d[3][3] = {
        { coeff[0][0], coeff[0][1], coeff[0][2] },
        { coeff[1][0], coeff[1][1], coeff[1][2] },
        { coeff[2][0], coeff[2][1], coeff[2][2] },
    };
    // matrix d1 using coeff as given in cramer's rule
    float d1[3][3] = {
        { coeff[0][3], coeff[0][1], coeff[0][2] },
        { coeff[1][3], coeff[1][1], coeff[1][2] },
        { coeff[2][3], coeff[2][1], coeff[2][2] },
    };
    // matrix d2 using coeff as given in cramer's rule
    float d2[3][3] = {
        { coeff[0][0], coeff[0][3], coeff[0][2] },
        { coeff[1][0], coeff[1][3], coeff[1][2] },
        { coeff[2][0], coeff[2][3], coeff[2][2] },
    };
    // matrix d3 using coeff as given in cramer's rule
    float d3[3][3] = {
        { coeff[0][0], coeff[0][1], coeff[0][3] },
        { coeff[1][0], coeff[1][1], coeff[1][3] },
        { coeff[2][0], coeff[2][1], coeff[2][3] },
    };
	
    // calculating determinant of matrices d, d1, d2, d3
    float D  = determinantOfMatrix(d);
    float D1 = determinantOfMatrix(d1);
    float D2 = determinantOfMatrix(d2);
    float D3 = determinantOfMatrix(d3);
	
	// angular accelerations:
	float aa_x = D1/D;
	float aa_y = D2/D;
	float aa_z = D3/D;
	
	// if rod is alligned along a main axis, 
	// correct the ang. accel. to zero:
	if (Ixx == 0.0f) {
		aa_x = 0.0;
		aa_y = t.y/Iyy;
		aa_z = t.z/Izz;
	}
	if (Iyy == 0.0f) {
		aa_x = t.x/Ixx;
		aa_y = 0.0;
		aa_z = t.z/Izz;
	}
	if (Izz == 0.0f) {
		aa_x = t.x/Ixx;
		aa_y = t.y/Iyy;
		aa_z = 0.0;
	}
	
	return make_float3(aa_x,aa_y,aa_z);
}



// --------------------------------------------------------
// IBM3D kernel to find determinant of 3x3 matrix:
// --------------------------------------------------------

__device__ inline float determinantOfMatrix(float mat[3][3])
{
    float ans = mat[0][0] * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2]) -
		        mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0]) + 
                mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
    return ans;
}
 

 
// --------------------------------------------------------
// IBM3D kernel to initialize curand random num. generator:
// --------------------------------------------------------

__global__ void init_curand_rods_IBM3D(
	curandState* state,
	unsigned long seed,
	int nRods)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nRods) {
		curand_init(seed,i,0,&state[i]);
	}    
}





