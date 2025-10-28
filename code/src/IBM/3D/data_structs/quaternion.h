# ifndef QUATERNION_H
# define QUATERNION_H


# include "tensor.h"
# include "../../../Utils/helper_math.h"



// --------------------------------------------------------
// struct that defines a bead on a rod in an IBM mesh:
// --------------------------------------------------------

struct quaternion {
	
	float w,x,y,z;
	
	inline __host__ __device__ void set_values(float q0, float q1, float q2, float q3) {
		w = q0;
		x = q1;
		y = q2;
		z = q3;
	}
	
	inline __host__ __device__ void set_values(tensor R) {
		// squares of quaternion components:
		float wsq = 0.25*(1.0 + R.xx + R.yy + R.zz);
		float xsq = 0.25*(1.0 + R.xx - R.yy - R.zz);
		float ysq = 0.25*(1.0 - R.xx + R.yy - R.zz);
		float zsq = 0.25*(1.0 - R.xx - R.yy + R.zz);
		// use largest to calculate other components:
		if (wsq >= 0.25) {
			w = sqrt(wsq);
			x = (R.zy - R.yz)/(4.0*w);
			y = (R.xz - R.zx)/(4.0*w);
			z = (R.yx - R.xy)/(4.0*w);
		}
		else if (xsq >= 0.25) {
			x = sqrt(xsq);
			w = (R.zy - R.yz)/(4.0*x);
			y = (R.xy + R.yx)/(4.0*x);
			z = (R.xz + R.zx)/(4.0*x);
		}
		else if (ysq >= 0.25) {
			y = sqrt(ysq);
			w = (R.xz - R.zx)/(4.0*y);
			x = (R.xy + R.yx)/(4.0*y);
			z = (R.yz + R.zy)/(4.0*y);
		}
		else if (zsq >= 0.25) {
			z = sqrt(ysq);
			w = (R.yx - R.xy)/(4.0*z);
			x = (R.xz + R.zx)/(4.0*z);
			y = (R.yz + R.zy)/(4.0*z);
		}
	}
	
	inline __host__ __device__ tensor get_rot_matrix() {
		tensor A;
		// the commented code is the transpose of A, 
		// which is what Allen & Tildsley write in Eq. (3.36)
		A.xx = w*w + x*x - y*y - z*z;
		A.xy = 2.0*(x*y - w*z);  // 2.0*(x*y + w*z); 
		A.xz = 2.0*(x*z + w*y);  // 2.0*(x*z - w*y);
		A.yx = 2.0*(x*y + w*z);  // 2.0*(x*y - w*z);
		A.yy = w*w - x*x + y*y - z*z;
		A.yz = 2.0*(y*z - w*x);  // 2.0*(y*z + w*x);
		A.zx = 2.0*(x*z - w*y);  // 2.0*(x*z + w*y);
		A.zy = 2.0*(y*z + w*x);  // 2.0*(y*z - w*x);
		A.zz = w*w - x*x - y*y + z*z;
		return A;
	}
	
	inline __host__ __device__ void normalize() {
		float norm = 1.0 / sqrt(w*w + x*x + y*y + z*z);
		w *= norm;
		x *= norm;
		y *= norm;
		z *= norm;
	}
	
	inline __host__ __device__ void update(float dt, float3 omega_body) {
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



# endif  // QUATERNION_H