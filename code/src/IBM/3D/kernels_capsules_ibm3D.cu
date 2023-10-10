

# include "kernels_capsules_ibm3D.cuh"
# include <stdio.h>



// **********************************************************************************************
// Kernels to compute membrane rest geometries
// **********************************************************************************************



// --------------------------------------------------------
// IBM3D kernel to zero the cell ref vol. and global area:
// --------------------------------------------------------

__global__ void zero_reference_vol_area_IBM3D(
	cell* cells, 
	int nCells)
{
	// define face:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nCells) {
		cells[i].vol0 = 0.0;
		cells[i].area0 = 0.0;
		cells[i].intrain = false;
		cells[i].vel = make_float3(0.0f,0.0f,0.0f);
	}
}



// --------------------------------------------------------
// IBM3D kernel to compute rest triangle properties (Skalak):
// --------------------------------------------------------

__global__ void rest_triangle_skalak_IBM3D(
	node* nodes,
	triangle* faces,
	cell* cells, 
	int nFaces)
{
	// define face:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nFaces) {
		int V0 = faces[i].v0;
		int V1 = faces[i].v1;
		int V2 = faces[i].v2;
		float3 r0 = nodes[V0].r; 
		float3 r1 = nodes[V1].r;
		float3 r2 = nodes[V2].r;
		float3 vec1 = r1 - r0;
		float3 vec2 = r2 - r0;
		float3 norm = cross(vec1,vec2);
		faces[i].area0 = 0.5*length(norm);
		faces[i].l0 = length(vec2);
		faces[i].lp0 = length(vec1);
		faces[i].cosphi0 = dot(vec1,vec2)/(faces[i].lp0*faces[i].l0);
		faces[i].sinphi0 = length(cross(vec1,vec2))/(faces[i].lp0*faces[i].l0);
		faces[i].T1 = 0.0;		
		// calculate global cell geometries:
		int cID = faces[i].cellID;
		float volFace = triangle_signed_volume(r0,r1,r2);
		atomicAdd(&cells[cID].vol0,volFace); 
		atomicAdd(&cells[cID].area0,faces[i].area0);
	}
}



// --------------------------------------------------------
// IBM3D kernel to compute rest edge lengths (Spring):
// --------------------------------------------------------

__global__ void rest_edge_lengths_IBM3D(
	node* nodes,
	edge* edges,
	int nEdges)
{
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nEdges) {
		// calculate resting edge length:
		float3 r0 = nodes[edges[i].v0].r; 
		float3 r1 = nodes[edges[i].v1].r;
		float3 r01 = r1 - r0;
		edges[i].length0 = length(r01);
	}
}



// --------------------------------------------------------
// IBM3D kernel to compute rest edge angles (Spring):
// --------------------------------------------------------

__global__ void rest_edge_angles_IBM3D(
	node* nodes,
	edge* edges,
	triangle* faces,
	int nEdges)
{
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nEdges) {
		// calculate edge vector:
		float3 r01 = nodes[edges[i].v1].r - nodes[edges[i].v0].r;
		r01 /= length(r01);
		// calculate normal vector for face 0:
		int f0 = edges[i].f0;  // face ID
		float3 r0 = nodes[faces[f0].v0].r; 
		float3 r1 = nodes[faces[f0].v1].r;
		float3 r2 = nodes[faces[f0].v2].r;
		float3 n0 = triangle_normalvector(r0,r1,r2);
		// calculate normal vector for face 1:
		int f1 = edges[i].f1;  // face ID
		r0 = nodes[faces[f1].v0].r; 
		r1 = nodes[faces[f1].v1].r;
		r2 = nodes[faces[f1].v2].r;
		float3 n1 = triangle_normalvector(r0,r1,r2);
		// angle between faces:
		edges[i].theta0 = angle_between_faces(n0,n1,r01);	
	}
}



// --------------------------------------------------------
// IBM3D kernel to compute rest triangle areas &
// cell volumes (Spring):
// --------------------------------------------------------

__global__ void rest_triangle_areas_IBM3D(
	node* nodes,
	triangle* faces,
	cell* cells, 
	int nFaces)
{
	// define face:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nFaces) {
		int V0 = faces[i].v0;
		int V1 = faces[i].v1;
		int V2 = faces[i].v2;
		float3 r0 = nodes[V0].r; 
		float3 r1 = nodes[V1].r;
		float3 r2 = nodes[V2].r;
		float3 norm = cross(r1 - r0, r2 - r0);
		faces[i].area0 = 0.5*length(norm);
		// calculate global cell geometries:
		int cID = faces[i].cellID;
		float volFace = triangle_signed_volume(r0,r1,r2);
		atomicAdd(&cells[cID].vol0,volFace); 
		atomicAdd(&cells[cID].area0,faces[i].area0);		
	}
}



// --------------------------------------------------------
// IBM3D kernel to compute rest cell volumes only:
// (note: this can be done when using Skalak model)
// --------------------------------------------------------

__global__ void rest_cell_volumes_IBM3D(
	node* nodes,
	triangle* faces,
	cell* cells, 
	int nFaces)
{
	// define face:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nFaces) {
		int V0 = faces[i].v0;
		int V1 = faces[i].v1;
		int V2 = faces[i].v2;
		float3 r0 = nodes[V0].r; 
		float3 r1 = nodes[V1].r;
		float3 r2 = nodes[V2].r;
		int cID = faces[i].cellID;
		float volFace = triangle_signed_volume(r0,r1,r2);
		atomicAdd(&cells[cID].vol0,volFace); 
	}
}






// **********************************************************************************************
// Skalak force kernel
// **********************************************************************************************



// --------------------------------------------------------
// IBM3D kernel to compute force on nodes based on the 
// Skalak elastic membrane model.  Here, we follow the details of 
// Timm Kruger's Thesis (Appendix C), or see Kruger et al. 
// Computers & Mathem. Appl. 61 (2011) 3485. 
// --------------------------------------------------------

__global__ void compute_node_force_membrane_skalak_IBM3D(
	triangle* faces,
	node* nodes,
	cell* cells,	
	int nFaces)
{
	// define face:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nFaces) {
				
		// get cell properties:
		int cID = faces[i].cellID;
		float gs = cells[cID].ks;
		float C  = cells[cID].C;
		
		// calculate current shape of triangle:
		int V0 = faces[i].v0;
		int V1 = faces[i].v1;
		int V2 = faces[i].v2;
		float3 r0 = nodes[V0].r; 
		float3 r1 = nodes[V1].r;
		float3 r2 = nodes[V2].r;
		float3 vec1 = r1 - r0;
		float3 vec2 = r2 - r0;
		const float l = length(vec2);
		const float lp = length(vec1);
		const float cosphi = dot(vec1,vec2)/(lp*l);
		const float sinphi = length(cross(vec1,vec2))/(lp*l);
		float3 norm = cross(vec1,vec2);
		const float  area = 0.5*length(norm);
		faces[i].area = area;
		faces[i].norm = norm;	
		
		// Variables in the reference state
	    const float l0 = faces[i].l0;
	    const float lp0 = faces[i].lp0;
	    const float cosphi0 = faces[i].cosphi0;
	    const float sinphi0 = faces[i].sinphi0;
	    const float area0 = faces[i].area0;			
		const float a1 = -l0*sinphi0/(2*area0);
	    const float a2 = -a1;
	    const float b1 = (l0*cosphi0 - lp0)/(2*area0);
	    const float b2 = -l0*cosphi0/(2*area0);
		
		// Displacement gradient tensor D: eq. (C.9) in Kruger
	    const float Dxx = lp/lp0;
	    const float Dxy = ((l/l0*cosphi) - (lp/lp0*cosphi0))/sinphi0;
	    const float Dyx = 0.0;
	    const float Dyy = l/l0*sinphi/sinphi0;

	    // Tensor G: (C.12)
	    const float Gxx = Dxx*Dxx + Dyx*Dyx;
	    const float Gxy = Dxx*Dxy + Dyx*Dyy;
	    const float Gyx = Gxy;  // symmetry
	    const float Gyy = Dxy*Dxy + Dyy*Dyy;

	    // Strain invariants, C.11 and C.12
	    const float i1 = (Gxx + Gyy) - 2.0;
	    const float i2 = (Gxx*Gyy - Gxy*Gyx) - 1.0;
		
	    // Principal stretch ratios, lambda1,2 = sqrt(eigenvalues of G tensor)
	    float lamb1 = sqrt(0.5*( Gxx + Gyy + sqrt((Gxx-Gyy)*(Gxx-Gyy) + 4.0*Gxy*Gxy)));		
	    float lamb2 = sqrt(0.5*( Gxx + Gyy - sqrt((Gxx-Gyy)*(Gxx-Gyy) + 4.0*Gxy*Gxy)));
		
		// Principal tension (Skalak model)
		const float J = lamb1*lamb2;
		faces[i].T1 = (gs/J)*(lamb1*lamb1*(lamb1*lamb1-1.0) + C*J*J*(J*J-1.0));
		faces[i].T1 /= gs;
		
		// Elastic strain energy:
		//faces[i].T1 = (gs*(i1*i1 + 2.0*i1 - 2.0*i2) + gs*C*i2*i2)/12.0;
		//faces[i].T1 *= area/gs;
		
	    // Derivatives of Skalak energy density E used in chain rule below: eq. (C.14)
	    float dEdI1 = 2.0*gs*(i1 + 1.0);
	    float dEdI2 = 2.0*gs*(C*i2 - 1.0);
		dEdI1 /= 12.0;    // prefactor according to Krueger (some others have 1/8)
		dEdI2 /= 12.0;    // "                            "
		// Derivatives of Neo-Hookean energy density E used in chain rule below: eq. (C.14)
	    //const float dEdI1 = gs/6.0;
	    //const float dEdI2 = -gs/(6.0*(i2+1.0)*(i2+1.0));
		
		// Derivatives of Is (C.15)
	    const float dI1dGxx = 1;
	    const float dI1dGxy = 0;
	    const float dI1dGyx = 0;
	    const float dI1dGyy = 1;
	    const float dI2dGxx = Gyy;
	    const float dI2dGxy = -Gyx;  // Note: Krueger has a factor 2 here, because he
	                                 // uses the symmetry of the G-matrix.
	    const float dI2dGyx = -Gxy;  // But we don't use it. So, Krueger is missing
	                                 // the yx term, whereas we have it.
	    const float dI2dGyy = Gxx;

	    // Derivatives of G (C.16)
	    const float dGxxdV1x = 2.0*a1*Dxx;
	    const float dGxxdV1y = 0.0;
	    const float dGxxdV2x = 2.0*a2*Dxx;
	    const float dGxxdV2y = 0.0;

	    const float dGxydV1x = a1*Dxy + b1*Dxx;
	    const float dGxydV1y = a1*Dyy;
	    const float dGxydV2x = a2*Dxy + b2*Dxx;
	    const float dGxydV2y = a2*Dyy;

	    const float dGyxdV1x = a1*Dxy + b1*Dxx;
	    const float dGyxdV1y = a1*Dyy;
	    const float dGyxdV2x = a2*Dxy + b2*Dxx;
	    const float dGyxdV2y = a2*Dyy;

	    const float dGyydV1x = 2.0*b1*Dxy;
	    const float dGyydV1y = 2.0*b1*Dyy;
	    const float dGyydV2x = 2.0*b2*Dxy;
	    const float dGyydV2y = 2.0*b2*Dyy;

	    // Calculate forces per area in rotated system: chain rule as in appendix C of
	    // Krüger (chain rule applied in eq. (C.13), but for the energy density). Only
	    // two nodes are needed, third one is calculated from momentum conservation
	    // Note: If you calculate the derivatives in a straightforward manner, you get
	    // 8 terms (done here). Krueger exploits the symmetry of the G-matrix, which
	    // results in 6 elements, but with an additional factor 2 for the xy-elements
	    // (see also above at the definition of dI2dGxy).
	    float2 f1_rot;
	    float2 f2_rot;
	    f1_rot.x = -(dEdI1 * dI1dGxx * dGxxdV1x) - (dEdI1 * dI1dGxy * dGxydV1x) -
	                (dEdI1 * dI1dGyx * dGyxdV1x) - (dEdI1 * dI1dGyy * dGyydV1x) -
	                (dEdI2 * dI2dGxx * dGxxdV1x) - (dEdI2 * dI2dGxy * dGxydV1x) -
	                (dEdI2 * dI2dGyx * dGyxdV1x) - (dEdI2 * dI2dGyy * dGyydV1x);
	    f1_rot.y = -(dEdI1 * dI1dGxx * dGxxdV1y) - (dEdI1 * dI1dGxy * dGxydV1y) -
	                (dEdI1 * dI1dGyx * dGyxdV1y) - (dEdI1 * dI1dGyy * dGyydV1y) -
	                (dEdI2 * dI2dGxx * dGxxdV1y) - (dEdI2 * dI2dGxy * dGxydV1y) -
	                (dEdI2 * dI2dGyx * dGyxdV1y) - (dEdI2 * dI2dGyy * dGyydV1y);
	    f2_rot.x = -(dEdI1 * dI1dGxx * dGxxdV2x) - (dEdI1 * dI1dGxy * dGxydV2x) -
	                (dEdI1 * dI1dGyx * dGyxdV2x) - (dEdI1 * dI1dGyy * dGyydV2x) -
	                (dEdI2 * dI2dGxx * dGxxdV2x) - (dEdI2 * dI2dGxy * dGxydV2x) -
	                (dEdI2 * dI2dGyx * dGyxdV2x) - (dEdI2 * dI2dGyy * dGyydV2x);
	    f2_rot.y = -(dEdI1 * dI1dGxx * dGxxdV2y) - (dEdI1 * dI1dGxy * dGxydV2y) -
	                (dEdI1 * dI1dGyx * dGyxdV2y) - (dEdI1 * dI1dGyy * dGyydV2y) -
	                (dEdI2 * dI2dGxx * dGxxdV2y) - (dEdI2 * dI2dGxy * dGxydV2y) -
	                (dEdI2 * dI2dGyx * dGyxdV2y) - (dEdI2 * dI2dGyy * dGyydV2y);

	    // Multiply by undeformed area
	    f1_rot *= area0;
	    f2_rot *= area0;	    
		
		// Rotate forces back into original position of triangle.  This is done
		// by finding the xu and yu directions in 3D space. See Kruger Fig. 7.1B. 
		// xu = normalized direction from node 0 to node 1 (vec1)
		// yu = normalized vector rejection of vec1 and vec2.  yu is orthogonal to xu.		
		float3 xu = normalize(vec1);
		float3 yu = normalize(vec2 - dot(vec2,xu)*xu);
	    float3 force0 = f1_rot.x*xu + f1_rot.y*yu;
	    float3 force1 = f2_rot.x*xu + f2_rot.y*yu;
	    float3 force2 = -force0-force1;
		
		// add forces to nodes
		add_force_to_vertex(V0,nodes,force0);
		add_force_to_vertex(V1,nodes,force1);
		add_force_to_vertex(V2,nodes,force2);
						
		// add to global cell geometries:		
		float volFace = triangle_signed_volume(r0,r1,r2);
		atomicAdd(&cells[cID].vol,volFace); 
		atomicAdd(&cells[cID].area,area);
		
		// add to global cell center-of-mass:
		int nF = cells[cID].nFaces;
		float3 rr = (r0+r1+r2)/3.0/float(nF);  // COM of face
		atomicAdd(&cells[cID].com.x,rr.x);
		atomicAdd(&cells[cID].com.y,rr.y);
		atomicAdd(&cells[cID].com.z,rr.z);
					
	}	
}






// **********************************************************************************************
// Spring force kernels
// **********************************************************************************************



// --------------------------------------------------------
// IBM3D kernel to compute force on node based on the 
// membrane model of Jancigova et al (Int. J. Numer. Meth.
// Fluids, 92:1368 (2020)):
// --------------------------------------------------------

__global__ void compute_node_force_membrane_area_IBM3D(
	triangle* faces,
	node* nodes,
	cell* cells,	
	float ka,
	int nFaces)
{
	// define face:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nFaces) {
				
		// calculate area and normal vector of face:
		int V0 = faces[i].v0;
		int V1 = faces[i].v1;
		int V2 = faces[i].v2;
		float3 r0 = nodes[V0].r; 
		float3 r1 = nodes[V1].r;
		float3 r2 = nodes[V2].r;
		float3 norm = cross(r1 - r0, r2 - r0);
		float  area = 0.5*length(norm);			
		faces[i].area = area;
		faces[i].norm = norm;
		
		// calculate local area dilation force:
		float area0 = faces[i].area0;	
		float darea = area - area0;
		//float areaRatio = (area-area0)/area0;
		//float areaForceMag = ka*(areaRatio + areaRatio/abs(0.09-areaRatio*areaRatio));		
		float3 centroid = (r0+r1+r2)/3.0;
		float3 ar0 = centroid - r0;
		float3 ar1 = centroid - r1;
		float3 ar2 = centroid - r2;
		float areaForceMag = ka*darea/(length2(ar0) + length2(ar1) + length2(ar2));
		add_force_to_vertex(V0,nodes,areaForceMag*ar0);
		add_force_to_vertex(V1,nodes,areaForceMag*ar1);
		add_force_to_vertex(V2,nodes,areaForceMag*ar2);			
		
		// add to global cell geometries:
		int cID = faces[i].cellID;
		float volFace = triangle_signed_volume(r0,r1,r2);
		atomicAdd(&cells[cID].vol,volFace); 
		atomicAdd(&cells[cID].area,area);
					
	}	
}



// --------------------------------------------------------
// IBM3D kernel to compute force on node based on the 
// membrane model of Jancigova et al (Int. J. Numer. Meth.
// Fluids, 92:1368 (2020)):
// --------------------------------------------------------

__global__ void compute_node_force_membrane_edge_IBM3D(
	triangle* faces,
	node* nodes,
	edge* edges,
	float ks,
	int nEdges)
{
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nEdges) {
		// calculate edge length:
		int V0 = edges[i].v0;
		int V1 = edges[i].v1;
		float3 r0 = nodes[V0].r; 
		float3 r1 = nodes[V1].r;
		float3 r01 = r1 - r0;
		float edgeL = length(r01);
		float length0 = edges[i].length0;
		// calculate edge stretching force:
		//float lengthRatio = (edgeL-length0)/length0;
		float lengthForceMag = ks*(edgeL-length0); //ks*(lengthRatio + lengthRatio/abs(9.0-lengthRatio*lengthRatio));
		r01 /= edgeL;  // normalize vector
		add_force_to_vertex(V0,nodes, lengthForceMag*r01);
		add_force_to_vertex(V1,nodes,-lengthForceMag*r01);		
	}
}



// --------------------------------------------------------
// IBM3D kernel to compute force on node based on the 
// membrane model of Jancigova et al (Int. J. Numer. Meth.
// Fluids, 92:1368 (2020)):
// --------------------------------------------------------

__global__ void compute_node_force_membrane_bending_IBM3D(
	triangle* faces,
	node* nodes,
	edge* edges,
	cell* cells,
	int nEdges)
{		
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nEdges) {
		// get basic data about edge:
		int V0 = edges[i].v0;
		int V1 = edges[i].v1;
		float3 n01 = normalize(nodes[V1].r - nodes[V0].r);	
		// calculate bending force magnitude:
		int F0 = edges[i].f0;
		int F1 = edges[i].f1;
		int cID = faces[F0].cellID;
		float kb = cells[cID].kb;
		float3 n0 = faces[F0].norm;  // normals were calculated above in
		float3 n1 = faces[F1].norm;  // "compute_node_force_membrane_area_IBM3D()"
		float dtheta = angle_between_faces(n0,n1,n01) - edges[i].theta0;
		float bendForceMag = kb*dtheta;  // = kb*(dtheta + dtheta/abs(2.467 - dtheta*dtheta));				
		// apply to the four points:
		int pA = V0;
		int pB = V1;
		int pC = unique_triangle_vertex(faces[F0].v0,faces[F0].v1,faces[F0].v2,pA,pB);
		int pD = unique_triangle_vertex(faces[F1].v0,faces[F1].v1,faces[F1].v2,pA,pB);
		float3 A = nodes[pA].r;
		float3 B = nodes[pB].r;
		float3 C = nodes[pC].r;
		float3 D = nodes[pD].r;
		float3 nC = n0/length2(n0);
		float3 nD = n1/length2(n1);
		float BmA = length(B-A);
		float3 FA = bendForceMag*(nC*dot(A-B,C-B)/BmA + nD*dot(A-B,D-B)/BmA);
		float3 FB = bendForceMag*(nC*dot(A-B,A-C)/BmA + nD*dot(A-B,A-D)/BmA);
		float3 FC = -bendForceMag*BmA*nC;
		float3 FD = -bendForceMag*BmA*nD;
		add_force_to_vertex(pA,nodes,FA);
		add_force_to_vertex(pB,nodes,FB);
		add_force_to_vertex(pC,nodes,FC);
		add_force_to_vertex(pD,nodes,FD);											
	}	
}



// --------------------------------------------------------
// IBM3D kernel to compute force on node based on the 
// membrane model of Jancigova et al (Int. J. Numer. Meth.
// Fluids, 92:1368 (2020)):
// --------------------------------------------------------

__global__ void compute_node_force_membrane_volume_IBM3D(
	triangle* faces,
	node* nodes,
	cell* cells,	
	int nFaces)
{
	// define face:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nFaces) {				
		// calculate volume conservation force:
		int cID = faces[i].cellID;
		int V0 = faces[i].v0;
		int V1 = faces[i].v1;
		int V2 = faces[i].v2;
		float kv = cells[cID].kv;
		float area = faces[i].area;
		float3 unitnorm = normalize(faces[i].norm);
		float volRatio = (cells[cID].vol - cells[cID].vol0)/cells[cID].vol0;
		float volForceMag = -kv*volRatio;
		volForceMag *= area;	
		volForceMag /= 3.0;	
		add_force_to_vertex(V0,nodes,volForceMag*unitnorm);
		add_force_to_vertex(V1,nodes,volForceMag*unitnorm);
		add_force_to_vertex(V2,nodes,volForceMag*unitnorm);					
	}	
}



// --------------------------------------------------------
// IBM3D kernel to compute force on node based on the 
// membrane model of Jancigova et al (Int. J. Numer. Meth.
// Fluids, 92:1368 (2020)):
// --------------------------------------------------------

__global__ void compute_node_force_membrane_globalarea_IBM3D(
	triangle* faces,
	node* nodes,
	cell* cells,	
	float kag,
	int nFaces)
{
	// define face:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nFaces) {				
		// calculate global area conservation force:
		int cID = faces[i].cellID;
		int V0 = faces[i].v0;
		int V1 = faces[i].v1;
		int V2 = faces[i].v2;
		float3 r0 = nodes[V0].r; 
		float3 r1 = nodes[V1].r;
		float3 r2 = nodes[V2].r;
		float area = cells[cID].area;
		float area0 = cells[cID].area0;
		float areaRatio = (area-area0)/area0;
		float3 centroid = (r0+r1+r2)/3.0;
		float3 ar0 = centroid - r0;
		float3 ar1 = centroid - r1;
		float3 ar2 = centroid - r2;
		float areaForceMag = kag*faces[i].area*areaRatio;
		areaForceMag /= length2(ar0) + length2(ar1) + length2(ar2);
		add_force_to_vertex(V0,nodes,areaForceMag*ar0);
		add_force_to_vertex(V1,nodes,areaForceMag*ar1);
		add_force_to_vertex(V2,nodes,areaForceMag*ar2);			
	}	
}





// **********************************************************************************************
// Miscellaneous kernels and functions
// **********************************************************************************************



// --------------------------------------------------------
// compute signed volume of a triangle face with origin:
// --------------------------------------------------------

__device__ inline float triangle_signed_volume(
	const float3 r0,
	const float3 r1,
	const float3 r2)
{
	return (- r0.z*r1.y*r2.x + r0.z*r1.x*r2.y + r0.y*r1.z*r2.x
	        - r0.x*r1.z*r2.y - r0.y*r1.x*r2.z + r0.x*r1.y*r2.z)/6.0;
}



// --------------------------------------------------------
// compute normal vector of a triangle face:
// --------------------------------------------------------

__device__ inline float3 triangle_normalvector(
	const float3 r0,
	const float3 r1,
	const float3 r2)
{
	return cross(r1 - r0, r2 - r0);
}



// --------------------------------------------------------
// compute angle between normals of two faces:
// n0 is the normal vector to face 0.
// n1 is the normal vector to face 1.
// r01 is the unit vector between common points 0 and 1
// shared by the two faces.
// --------------------------------------------------------

__device__ inline float angle_between_faces(
	const float3 n0,
	const float3 n1,
	const float3 r01)
{
	// left-handed:
	float3 n1xn0 = cross(n1,n0);
	return M_PI - std::atan2( dot(n1xn0,r01), dot(n0,n1) );
	// right-handed:
	//float3 n0xn1 = cross(n0,n1);
	//float3 r10 = make_float3(0.0,0.0,0.0) - r01;
	//return M_PI - std::atan2( dot(n0xn1,r10), dot(n0,n1) );
}



// --------------------------------------------------------
// find the vertex (v0, v1, v2) that is not A or B:
// --------------------------------------------------------

__device__ inline int unique_triangle_vertex(
	const int v0,
	const int v1,
	const int v2,
	const int A,
	const int B)
{
	if (v0 != A && v0 != B) {
		return v0;
	} else if (v1 != A && v1 != B) {
		return v1;
	} else {
		return v2;
	}	
}



// --------------------------------------------------------
// add force to vertex using atomicAdd:
// --------------------------------------------------------

__device__ inline void add_force_to_vertex(
	int i,
	node* nodes,
	const float3 g)
{
	atomicAdd(&nodes[i].f.x,g.x);
	atomicAdd(&nodes[i].f.y,g.y);
	atomicAdd(&nodes[i].f.z,g.z);	
}



// --------------------------------------------------------
// IBM3D kernel to zero node forces:
// --------------------------------------------------------

__global__ void zero_node_forces_IBM3D(
	node* nodes,	
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) nodes[i].f = make_float3(0.0);
}



// --------------------------------------------------------
// IBM3D kernel to zero cell volumes:
// --------------------------------------------------------

__global__ void zero_cell_volumes_IBM3D(
	cell* cells,	
	int nCells)
{
	// define cell:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nCells) {
		cells[i].vol = 0.0;
		cells[i].area = 0.0;
		cells[i].com = make_float3(0.0);
	}
}



// --------------------------------------------------------
// IBM3D kernel to unwrap node coordinates.  Here, the
// nodes of a cell are brought back close to the cell's 
// refNode.  This is done to avoid complications with
// PBCs:
// --------------------------------------------------------

__global__ void unwrap_node_coordinates_IBM3D(
	node* nodes,
	cell* cells,
	float3 Box,
	int3 pbcFlag,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		int c = nodes[i].cellID;
		int j = cells[c].refNode;
		float3 rij = nodes[j].r - nodes[i].r;		
		nodes[i].r = nodes[i].r + roundf(rij/Box)*Box*pbcFlag; // PBC's
	}
}



// --------------------------------------------------------
// IBM3D kernel to wrap node coordinates for PBCs:
// --------------------------------------------------------

__global__ void wrap_node_coordinates_IBM3D(
	node* nodes,	
	float3 Box,
	int3 pbcFlag,
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {	
		nodes[i].r = nodes[i].r - floorf(nodes[i].r/Box)*Box*pbcFlag;		
	}
}



// --------------------------------------------------------
// IBM3D kernel to reduce cell volume:
// --------------------------------------------------------

__global__ void change_cell_volumes_IBM3D(
	cell* cells,
	float dV,
	int nCells)
{
	// define cell:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nCells) {
		cells[i].vol0 += dV;
		cells[i].area0 = pow(M_PI,1./3.)*pow(6*cells[i].vol0,2./3.);
	}
}



// --------------------------------------------------------
// IBM3D kernel to scale edge length:
// --------------------------------------------------------

__global__ void scale_edge_lengths_IBM3D(
	edge* edges,
	float scale,
	int nEdges)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nEdges) edges[i].length0 *= scale;
}



// --------------------------------------------------------
// IBM3D kernel to scale face area:
// --------------------------------------------------------

__global__ void scale_face_areas_IBM3D(
	triangle* faces,
	float scale,
	int nFaces)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	if (i < nFaces) {
		faces[i].area0 *= scale*scale;
		faces[i].l0 *= scale;
		faces[i].lp0 *= scale;
	}
}



// --------------------------------------------------------
// IBM3D kernel to scale cell global area & volume:
// --------------------------------------------------------

__global__ void scale_cell_areas_volumes_IBM3D(
	cell* cells,
	float scale,
	int nCells)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nCells) {
		cells[i].area0 *= scale*scale;
		cells[i].vol0 *= scale*scale*scale;	
		if (cells[i].vol0 > 5000) printf("%i cell vol0 = %f \n",i,cells[i].vol0); 	
	}
}



// --------------------------------------------------------
// IBM3D node update kernel:
// --------------------------------------------------------

__global__ void update_node_position_verlet_1_cellType2_stationary_IBM3D(
	node* nodes,
	cell* cells,
	float dt,
	float m,	
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) {
		int cID = nodes[i].cellID;
		if (cells[cID].cellType != 2) {
			nodes[i].r += nodes[i].v*dt + 0.5*dt*dt*(nodes[i].f/m);
			nodes[i].v += 0.5*dt*(nodes[i].f/m);
		}		
	}
}





