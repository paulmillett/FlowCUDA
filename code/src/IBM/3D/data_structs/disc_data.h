# ifndef DISC_DATA_H
# define DISC_DATA_H

# include "tensor.h"
# include "quaternion.h"



// --------------------------------------------------------
// struct that defines a bead on a disc in an IBM mesh:
// --------------------------------------------------------

struct beaddisc {
	int discID;
	float3 r;
	float3 v;
	float3 f;
	float3 rm1;
	float3 rrel;   // separation vector to disc center
	float3 uf;     // fluid velocity at bead position	
	tensor gradu;  // gradient of fluid velocity at bead position
	float3 wallContactHist;   // accumulated tangential distance during contact with wall 
};



// --------------------------------------------------------
// struct that defines a rigid disc in an IBM mesh:
// --------------------------------------------------------

struct disc {
	int discType;
	int centerBead;
	int nBeads;
	int indxB0;    // starting bead index for disc
	float rad;     // radius of disc 
	float h2;      // half of disc thickness
	float ar;      // aspect ratio of disc
	float mobParT; // mobility coefficient translational (parallel)
	float mobPerT; // mobility coefficient translational (perpendicular)
	float mobParR; // mobility coefficient rotational    (parallel)
	float mobPerR; // mobility coefficient rotational    (parallel)
	float3 r;      // position
	float3 v;      // velocity
	float3 f;      // force	
	float3 t;      // torque
	float3 p;      // orientation vector
	float3 uf;     // fluid velocity at rod position
	tensor gradu;  // gradient of fluid velocity
	quaternion q;  // quaternion that fully defines disc orientation
};


# endif  // DISC_DATA_H
