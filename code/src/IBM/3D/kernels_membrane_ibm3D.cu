
# include "kernels_membrane_ibm3D.cuh"
# include <stdio.h>



// --------------------------------------------------------
// IBM3D kernel to compute rest edge lengths:
// --------------------------------------------------------

__global__ void rest_edge_lengths_IBM3D(
	float3* vertR,
	edge* edges,
	int nEdges)
{
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nEdges) {
		// calculate resting edge length:
		float3 r0 = vertR[edges[i].v0]; 
		float3 r1 = vertR[edges[i].v1];
		float3 r01 = r1 - r0;
		edges[i].length0 = length(r01);
	}
}



// --------------------------------------------------------
// IBM3D kernel to compute rest edge angles:
// --------------------------------------------------------

__global__ void rest_edge_angles_IBM3D(
	float3* vertR,
	edge* edges,
	triangle* faces,
	int nEdges)
{
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nEdges) {
		// calculate edge vector:
		float3 r01 = vertR[edges[i].v1] - vertR[edges[i].v0];
		r01 /= length(r01);
		// calculate normal vector for face 0:
		int f0 = edges[i].f0;  // face ID
		float3 r0 = vertR[faces[f0].v0]; 
		float3 r1 = vertR[faces[f0].v1];
		float3 r2 = vertR[faces[f0].v2];
		float3 n0 = triangle_normalvector(r0,r1,r2);
		// calculate normal vector for face 1:
		int f1 = edges[i].f1;  // face ID
		r0 = vertR[faces[f1].v0]; 
		r1 = vertR[faces[f1].v1];
		r2 = vertR[faces[f1].v2];
		float3 n1 = triangle_normalvector(r0,r1,r2);
		// angle between faces:
		edges[i].theta0 = angle_between_faces(n0,n1,r01);	
	}
}



// --------------------------------------------------------
// IBM3D kernel to compute rest triangle areas:
// --------------------------------------------------------

__global__ void rest_triangle_areas_IBM3D(
	float3* vertR,
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
		float3 r0 = vertR[V0]; 
		float3 r1 = vertR[V1];
		float3 r2 = vertR[V2];
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
// IBM3D kernel to compute force on node based on the 
// membrane model of Jancigova et al (Int. J. Numer. Meth.
// Fluids, 92:1368 (2020)):
// --------------------------------------------------------

__global__ void compute_node_force_membrane_area_IBM3D(
	triangle* faces,
	float3* vertR,
	float3* vertF,
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
		float3 r0 = vertR[V0]; 
		float3 r1 = vertR[V1];
		float3 r2 = vertR[V2];
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
		add_force_to_vertex(V0,vertF,areaForceMag*ar0);
		add_force_to_vertex(V1,vertF,areaForceMag*ar1);
		add_force_to_vertex(V2,vertF,areaForceMag*ar2);			
		
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
	float3* vertR,
	float3* vertF,
	edge* edges,
	float ks,
	float kb,
	int nEdges)
{
	// define edge:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nEdges) {
		
		// --------------------------------
		// Edge stretching force:
		// --------------------------------
				
		// calculate edge length:
		int V0 = edges[i].v0;
		int V1 = edges[i].v1;
		float3 r0 = vertR[V0]; 
		float3 r1 = vertR[V1];
		float3 r01 = r1 - r0;
		float edgeL = length(r01);
		float length0 = edges[i].length0;
		
		// calculate edge elongation force:
		float lengthRatio = (edgeL-length0)/length0;
		float lengthForceMag = ks*(lengthRatio + lengthRatio/abs(9.0-lengthRatio*lengthRatio));
		r01 /= edgeL;  // normalize vector
		add_force_to_vertex(V0,vertF, lengthForceMag*r01);
		add_force_to_vertex(V1,vertF,-lengthForceMag*r01);
						
		// --------------------------------
		// Edge bending force:
		// --------------------------------
	
		// bend force magnitude between faces:		
		int F0 = edges[i].f0;
		int F1 = edges[i].f1;
		float3 n0 = faces[F0].norm;  // normals were calculated above in
		float3 n1 = faces[F1].norm;  // "compute_node_force_membrane_area_IBM3D()"
		float dtheta = angle_between_faces(n0,n1,r01/edgeL) - edges[i].theta0;
		float bendForceMag = kb*dtheta;  // = kb*(dtheta + dtheta/abs(2.467 - dtheta*dtheta));
				
		// apply to the four points:
		int pA = V0;
		int pB = V1;
		int pC = unique_triangle_vertex(faces[F0].v0,faces[F0].v1,faces[F0].v2,pA,pB);
		int pD = unique_triangle_vertex(faces[F1].v0,faces[F1].v1,faces[F1].v2,pA,pB);
		float3 A = vertR[pA];
		float3 B = vertR[pB];
		float3 C = vertR[pC];
		float3 D = vertR[pD];
		float3 nC = n0/length2(n0);
		float3 nD = n1/length2(n1);
		float BmA = length(B-A);
		float3 FA = bendForceMag*(nC*dot(A-B,C-B)/BmA + nD*dot(A-B,D-B)/BmA);
		float3 FB = bendForceMag*(nC*dot(A-B,A-C)/BmA + nD*dot(A-B,A-D)/BmA);
		float3 FC = -bendForceMag*BmA*nC;
		float3 FD = -bendForceMag*BmA*nD;		
		add_force_to_vertex(pA,vertF,FA);
		add_force_to_vertex(pB,vertF,FB);
		add_force_to_vertex(pC,vertF,FC);
		add_force_to_vertex(pD,vertF,FD);		
										
	}	
}



// --------------------------------------------------------
// IBM3D kernel to compute force on node based on the 
// membrane model of Jancigova et al (Int. J. Numer. Meth.
// Fluids, 92:1368 (2020)):
// --------------------------------------------------------

__global__ void compute_node_force_membrane_volume_IBM3D(
	triangle* faces,
	float3* vertF,
	cell* cells,	
	float kv,
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
		float area = faces[i].area;
		float3 unitnorm = faces[i].norm/length(faces[i].norm);
		float volRatio = (cells[cID].vol - cells[cID].vol0)/cells[cID].vol0;
		float volForceMag = -kv*volRatio;
		volForceMag *= area;	
		volForceMag /= 3.0;	
		add_force_to_vertex(V0,vertF,volForceMag*unitnorm);
		add_force_to_vertex(V1,vertF,volForceMag*unitnorm);
		add_force_to_vertex(V2,vertF,volForceMag*unitnorm);					
	}	
}



// --------------------------------------------------------
// IBM3D kernel to compute force on node based on the 
// membrane model of Jancigova et al (Int. J. Numer. Meth.
// Fluids, 92:1368 (2020)):
// --------------------------------------------------------

__global__ void compute_node_force_membrane_globalarea_IBM3D(
	triangle* faces,
	float3* vertR,
	float3* vertF,
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
		float3 r0 = vertR[V0]; 
		float3 r1 = vertR[V1];
		float3 r2 = vertR[V2];
		float area = cells[cID].area;
		float area0 = cells[cID].area0;
		float areaRatio = (area-area0)/area0;
		float3 centroid = (r0+r1+r2)/3.0;
		float3 ar0 = centroid - r0;
		float3 ar1 = centroid - r1;
		float3 ar2 = centroid - r2;
		float areaForceMag = kag*faces[i].area*areaRatio;
		areaForceMag /= length2(ar0) + length2(ar1) + length2(ar2);
		add_force_to_vertex(V0,vertF,areaForceMag*ar0);
		add_force_to_vertex(V1,vertF,areaForceMag*ar1);
		add_force_to_vertex(V2,vertF,areaForceMag*ar2);			
	}	
}



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
	float3* f,
	const float3 g)
{
	atomicAdd(&f[i].x,g.x);
	atomicAdd(&f[i].y,g.y);
	atomicAdd(&f[i].z,g.z);
}



// --------------------------------------------------------
// IBM3D kernel to zero node forces and cell volumes:
// --------------------------------------------------------

__global__ void zero_node_forces_IBM3D(
	float3* vertF,	
	int nNodes)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nNodes) vertF[i] = make_float3(0.0);
}



// --------------------------------------------------------
// IBM3D kernel to zero node forces and cell volumes:
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
	}
}



// --------------------------------------------------------
// IBM3D kernel to reduce cell volume:
// --------------------------------------------------------

__global__ void change_cell_volumes_IBM3D(
	cell* cells,
	float change,
	int nCells)
{
	// define cell:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nCells) cells[i].vol0 += change;
}


