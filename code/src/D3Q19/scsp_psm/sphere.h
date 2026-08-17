# ifndef SPHERE_H
# define SPHERE_H


# include "../../IBM/3D/data_structs/quaternion.h"


// --------------------------------------------------------
// struct that defines a RIGID sphere for the PSM method:
// --------------------------------------------------------

struct sphere {
	int sphereType;	
	float vol;       // volume of sphere
	float rad;       // radius of sphere
	float mass;      // mass of sphere
	float I;         // moment of inertia (solid sphere = 2/5 * mass * R^2)
	float3 r;        // position
	float3 v;        // velocity
	float3 f;        // force
	float3 t;        // torque
	float3 L;        // angular momentum
	float3 w;        // angular velocity
	quaternion q;    // quaternion (orientation vector)
};



# endif  // SPHERE_H