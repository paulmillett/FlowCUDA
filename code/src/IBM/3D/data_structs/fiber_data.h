# ifndef FIBER_DATA_H
# define FIBER_DATA_H



// --------------------------------------------------------
// struct that defines a bead on a fiber in an IBM mesh:
// --------------------------------------------------------

struct beadfiber {
	float3 r;
	float3 v;
	float3 f;
	float3 rm1;
	float3 rstar;
	float3 d2r;
	int fiberID;
	int posID;
};



// --------------------------------------------------------
// struct that defines a fiber edge in an IBM mesh:
// --------------------------------------------------------

struct edgefiber {
	int b0,b1;
	int posID;
};



// --------------------------------------------------------
// struct that defines a flexible fiber in an IBM mesh:
// --------------------------------------------------------

struct fiber {
	int headBead;
	int nBeads;
	int nEdges;
	int indxB0;   // starting bead index for fiber
	int indxE0;   // starting edge index for fiber
	float rad;    // radius of bead 
	float gam;
	float3 com;
	float3 vel;
};


# endif  // FIBER_DATA_H

