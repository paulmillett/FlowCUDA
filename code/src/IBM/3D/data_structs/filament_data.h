# ifndef FILAMENT_DATA_H
# define FILAMENT_DATA_H



// --------------------------------------------------------
// struct that defines a bead in an IBM mesh:
// --------------------------------------------------------

struct bead {
	float3 r;
	float3 v;
	float3 f;
	float3 fdip;
	int filamID;
};



// --------------------------------------------------------
// struct that defines a filament edge in an IBM mesh:
// --------------------------------------------------------

struct edgefilam {
	int b0,b1;
	float length0;
};



// --------------------------------------------------------
// struct that defines a flexible filament in an IBM mesh:
// --------------------------------------------------------

struct filament {
	int filamType;
	int headBead;
	int nBeads;
	int nEdges;
	int indxB0;  // starting bead index for filament
	int indxE0;  // starting edge index for filament
	float rad;   // radius of bead 
	float ks;
	float kb;
	float fp;
	float up;    // sometimes 'up' is used
	float3 com;
	float3 vel;
};


# endif  // FILAMENT_DATA_H

