# ifndef RIGID_DATA_H
# define RIGID_DATA_H


# include "quaternion.h"



// --------------------------------------------------------
// struct that defines a node in a rigid-cell mesh:
// --------------------------------------------------------

struct rigidnode {
	float3 r;
	float3 v;
	float3 f;
	float3 delta;
	int cellID;
};



// --------------------------------------------------------
// struct that defines a RIGID cell consisting of
// an enclosed surface mesh:
// --------------------------------------------------------

struct rigid {
	int cellType;
	int refNode;
	int nNodes;
	int indxN0;      // starting node index for cell
	float vol;
	float area;
	float rad;
	float mass;      // mass of rigid-body
	float3 com;      // position
	float3 vel;      // velocity
	float3 f;        // force
	float3 t;        // torque
	float3 L;        // angular momentum
	float3 I;        // principal moments of inertia
	float3 p;        // orientation vector
	float3 omega;    // angular velocity
	quaternion q;    // quaternion (orientation vector)
};



# endif  // RIGID_DATA_H