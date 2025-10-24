# ifndef RIGID_DATA_H
# define RIGID_DATA_H


# include "tensor.h"
# include "../../../Utils/helper_math.h"



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
// struct that defines a bead on a rod in an IBM mesh:
// --------------------------------------------------------

struct quaternion {
	
	float w,x,y,z;
	
	__host__ __device__ tensor get_rot_matrix() {
		tensor A;
		A.xx = w*w + x*x - y*y - z*z;
		A.xy = 2.0*(x*y + w*z); 
		A.xz = 2.0*(x*z - w*y);
		A.yx = 2.0*(x*y - w*z);
		A.yy = w*w - x*x + y*y - z*z;
		A.yz = 2.0*(y*z + w*x);
		A.zx = 2.0*(x*z + w*y);
		A.zy = 2.0*(y*z - w*x);
		A.zz = w*w - x*x - y*y + z*z;
		return A;
	}
	
	__host__ __device__ void normalize() {
		float norm = 1.0 / sqrt(w*w + x*x + y*y + z*z);
		w *= norm;
		x *= norm;
		y *= norm;
		z *= norm;
	}
	
	__host__ __device__ void update(float dt, float3 omega_body) {
		float dw = 0.5 * (-x*omega_body.x - y*omega_body.y - z*omega_body.z);
		float dx = 0.5 * ( w*omega_body.x - z*omega_body.y + y*omega_body.z);
		float dy = 0.5 * ( z*omega_body.x + w*omega_body.y - x*omega_body.z);
		float dz = 0.5 * (-y*omega_body.x + x*omega_body.y + w*omega_body.z);
		w += dt*dw;
		x += dt*dx;
		y += dt*dy;
		z += dt*dz;
	}
	
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
	float3 com;      // position
	float3 vel;      // velocity
	float3 f;        // force
	float3 t;        // torque
	float3 L;        // angular momentum
	float3 I;        // principal moments of inertia
	float3 p;        // orientation vector
	quaternion q;    // quaternion (orientation vector)
};



# endif  // RIGID_DATA_H