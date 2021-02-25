# ifndef MEMBRANE_DATA_H
# define MEMBRANE_DATA_H


// --------------------------------------------------------
// struct that defines a triangular face in an IBM mesh:
// --------------------------------------------------------

struct triangle {
	int cellID;
	int v0,v1,v2;
	float area;
	float area0;
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
	float vol;
	float vol0;
	float areaAve;
	float3 com;
};

# endif  // MEMBRANE_DATA_H

