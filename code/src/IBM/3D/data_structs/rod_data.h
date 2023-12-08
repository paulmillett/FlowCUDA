# ifndef ROD_DATA_H
# define ROD_DATA_H



// --------------------------------------------------------
// struct that defines a bead on a rod in an IBM mesh:
// --------------------------------------------------------

struct beadrod {
	float3 r;
	float3 f;
	int rodID;
};



// --------------------------------------------------------
// struct that defines a rigid rod in an IBM mesh:
// --------------------------------------------------------

struct rod {
	int rodType;
	int headBead;
	int tailBead;
	int centerBead;
	int nBeads;
	int indxB0;   // starting bead index for filament
	float rad;    // radius of bead 
	float fp;     // propulsion force
	float up;     // propulsion velocity
	float3 r;     // position
	float3 v;     // velocity
	float3 f;     // force	
	float3 t;     // torque
	float3 p;     // orientation vector
	float Ixx,Iyy,Izz;  // moments of inertia
	float Ixy,Ixz,Iyz;  // products of inertia
};


# endif  // ROD_DATA_H
