
# include "kernels_membrane_ibm3D.cuh"



// --------------------------------------------------------
// IBM3D kernel to compute force on node based on the 
// membrane model of Zavodszky et al (Frontiers in Physiology
// vol 8, p 563, 2017):
// --------------------------------------------------------

__global__ void compute_node_force_membrane_area_IBM3D(
	triangle* faces,
	float3* vertR,
	float3* vertF,
	cell* cells,	
	float ka,
	int nFaces)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nFaces) {
		
		// cell ID for this face:
		int cID = faces[i].cellID;
		
		// vertex ID's for this face:
		int V0 = faces[i].v0;
		int V1 = faces[i].v1;
		int V2 = faces[i].v2;
		
		// vertex positions:
		float3 r0 = vertR[V0]; 
		float3 r1 = vertR[V1];
		float3 r2 = vertR[V2];
		
		// calculate volume associated with this face,
		// and add it to the volume of cell:
		float volFace = triangle_signed_volume(r0,r1,r2);
		atomicAdd(&cells[cID].vol,volFace);
		
		// calculate area and normal vector of face:
		float area = 0.0;
		float3 norm = make_float3(0.0);
		float area0 = faces[i].area0;
		triangle_area_normalvector(r0,r1,r2,area,norm);
		
		// calculate local area dilation force:
		float areaRatio = (area-area0)/area0;
		float areaForceMag = ka*(areaRatio + areaRatio/abs(0.09-areaRatio*areaRatio));
		float3 centroid = (r0+r1+r2)/3.0;
		float3 ar0 = centroid - r0;
		float3 ar1 = centroid - r1;
		float3 ar2 = centroid - r2;
		add_force_to_vertex(vertF[V0],areaForceMag*ar0);
		add_force_to_vertex(vertF[V1],areaForceMag*ar1);
		add_force_to_vertex(vertF[V2],areaForceMag*ar2);		
		faces[i].area = area;
		faces[i].norm = norm;			
	}	
}



// --------------------------------------------------------
// IBM3D kernel to compute force on node based on the 
// membrane model of Zavodszky et al (Frontiers in Physiology
// vol 8, p 563, 2017):
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
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nEdges) {
		
		// vertex ID's for this edge:
		int V0 = edges[i].v0;
		int V1 = edges[i].v1;		
		
		// vertex positions:
		float3 r0 = vertR[V0]; 
		float3 r1 = vertR[V1];
		
		// calculate edge length:
		float3 r01 = r1 - r0;
		float edgeL = length(r01);
		float length0 = edges[i].length0;
		
		// calculate edge elongation force:
		float lengthRatio = (edgeL-length0)/length0;
		float lengthForceMag = ks*(lengthRatio + lengthRatio/abs(9.0-lengthRatio*lengthRatio));
		r01 /= edgeL;  // normalize vector
		add_force_to_vertex(vertF[V0], lengthForceMag*r01);
		add_force_to_vertex(vertF[V0],-lengthForceMag*r01);
		
		// face ID's for this edge:
		int F0 = edges[i].f0;
		int F1 = edges[i].f1;
		
		// normal vectors for faces:
		float3 n0 = faces[F0].norm;
		float3 n1 = faces[F1].norm;
		
		// angle between faces:
		float theta = angle_between_faces(n0,n1,r01);
		
		// calculate bending force:
		float dtheta = theta - edges[i].theta0;
		float bendForceMag = kb*(dtheta + dtheta/abs(2.467 - dtheta*dtheta));
		
		
		
						
	}	
}



// --------------------------------------------------------
// IBM3D kernel to compute force on node based on the 
// membrane model of Zavodszky et al (Frontiers in Physiology
// vol 8, p 563, 2017):
// --------------------------------------------------------

__global__ void compute_node_force_membrane_volume_IBM3D(
	triangle* faces,
	float3* vertF,
	cell* cells,	
	float kv,
	int nFaces)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nFaces) {
		
		// cell ID for this face:
		int cID = faces[i].cellID;
		
		// vertex ID's for this face:
		int V0 = faces[i].v0;
		int V1 = faces[i].v1;
		int V2 = faces[i].v2;
		
		// area and normal vector of this face:
		int area = faces[i].area;
		float3 norm = faces[i].norm;
				
		// calculate volume conservation force:
		float volRatio = (cells[cID].vol - cells[cID].vol0)/cells[cID].vol0;
		float volForceMag = -kv*(volRatio + volRatio/abs(0.01-volRatio*volRatio));
		volForceMag *= area/cells[cID].areaAve;		
		add_force_to_vertex(vertF[V0],volForceMag*norm);
		add_force_to_vertex(vertF[V1],volForceMag*norm);
		add_force_to_vertex(vertF[V2],volForceMag*norm);					
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
// compute area and normal vector of a triangle face:
// --------------------------------------------------------

__device__ inline void triangle_area_normalvector(
	const float3 r0,
	const float3 r1,
	const float3 r2,
	float area,
	float3 norm)
{
	norm = cross(r1 - r0, r2 - r0);
	float normL = length(norm);
	area = 0.5*normL;
	norm /= normL;
}



// --------------------------------------------------------
// compute angle between normals of two faces:
// --------------------------------------------------------

__device__ inline float angle_between_faces(
	const float3 n0,
	const float3 n1,
	const float3 r01)
{
	float3 n0xn1 = cross(n0,n1);
	return std::atan2( dot(n0xn1,r01), dot(n0,n1) );
}



// --------------------------------------------------------
// add force to vertex using atomicAdd:
// --------------------------------------------------------

__device__ inline void add_force_to_vertex(
	float3 f,
	const float3 g)
{
	atomicAdd(&f.x,g.x);
	atomicAdd(&f.y,g.y);
	atomicAdd(&f.z,g.z);
}



