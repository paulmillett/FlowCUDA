# ifndef MEMBRANE_DATA_H
# define MEMBRANE_DATA_H


// --------------------------------------------------------
// struct that defines a node in an IBM mesh:
// --------------------------------------------------------

struct node {
	float3 r;
	float3 v;
	float3 f;
	int cellID;
};



// --------------------------------------------------------
// struct that defines a triangular face in an IBM mesh:
// --------------------------------------------------------

struct triangle {
	int cellID;
	int faceType;
	int v0,v1,v2;
	float area;
	float area0;
	float l0,lp0;
	float cosphi0;
	float sinphi0;
	float T1;
	float3 norm;
};



// --------------------------------------------------------
// struct that defines an edge in an IBM mesh:
// --------------------------------------------------------

struct edge {
	int v0,v1;
	int f0,f1;
	float length0;
	float theta0;
};



// --------------------------------------------------------
// struct that defines a vertex in an IBM mesh:
// --------------------------------------------------------

struct cell {
	int cellType;
	int refNode;
	int nNodes;
	int nFaces;
	int nEdges;
	int indxN0;  // starting node index for cell
	int indxF0;  // starting face index for cell
	int indxE0;  // starting edge index for cell
	int trainID;
	bool intrain;
	float vol;
	float vol0;
	float area;
	float area0;
	float rad;
	float ks;
	float kb;
	float kv;
	float C;
	float Ca;
	float maxT1;
	float D;
	float3 com;
	float3 vel;
	float3 p;    // orientation vector
};

# endif  // MEMBRANE_DATA_H

