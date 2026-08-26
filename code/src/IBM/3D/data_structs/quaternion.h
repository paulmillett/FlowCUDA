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
		float wsq = 0.25f * (1.0f + R.xx + R.yy + R.zz);
		float xsq = 0.25f * (1.0f + R.xx - R.yy - R.zz);
		float ysq = 0.25f * (1.0f - R.xx + R.yy - R.zz);
		float zsq = 0.25f * (1.0f - R.xx - R.yy + R.zz);
		// use largest branch to calculate other components (prevents div-by-zero):
		if (wsq >= 0.25f) {
		    w = sqrt(wsq);
		    x = (R.zy - R.yz) / (4.0f * w);
		    y = (R.xz - R.zx) / (4.0f * w);
		    z = (R.yx - R.xy) / (4.0f * w);
		}
		else if (xsq >= 0.25f) {
		    x = sqrt(xsq);
		    w = (R.zy - R.yz) / (4.0f * x);
		    y = (R.xy + R.yx) / (4.0f * x);
		    z = (R.xz + R.zx) / (4.0f * x);
		}
		else if (ysq >= 0.25f) {
		    y = sqrt(ysq);
		    w = (R.xz - R.zx) / (4.0f * y);
		    x = (R.xy + R.yx) / (4.0f * y);
		    z = (R.yz + R.zy) / (4.0f * y);
		}
		else {
		    z = sqrt(zsq);
		    w = (R.yx - R.xy) / (4.0f * z);
		    x = (R.xz + R.zx) / (4.0f * z);
		    y = (R.yz + R.zy) / (4.0f * z);
		}
	}
	
	inline __host__ __device__ tensor get_rot_matrix() {
		tensor A;
		// the commented code is the transpose of A, 
		// which is what Allen & Tildsley write in Eq. (3.36)
		A.xx = w*w + x*x - y*y - z*z;
		A.xy = 2.0f*(x*y - w*z);  // 2.0*(x*y + w*z); 
		A.xz = 2.0f*(x*z + w*y);  // 2.0*(x*z - w*y);
		A.yx = 2.0f*(x*y + w*z);  // 2.0*(x*y - w*z);
		A.yy = w*w - x*x + y*y - z*z;
		A.yz = 2.0f*(y*z - w*x);  // 2.0*(y*z + w*x);
		A.zx = 2.0f*(x*z - w*y);  // 2.0*(x*z + w*y);
		A.zy = 2.0f*(y*z + w*x);  // 2.0*(y*z - w*x);
		A.zz = w*w - x*x - y*y + z*z;
		return A;
	}
	
	inline __host__ __device__ void normalize() {
		// rsqrtf = fact CUDA reciprical sqrt
		float inv_norm = rsqrtf(w * w + x * x + y * y + z * z);
		w *= inv_norm;
		x *= inv_norm;
		y *= inv_norm;
		z *= inv_norm;
	}
	
	// assuming omega_body is angular velocity in the world frame
	inline __host__ __device__ void update(float dt, float3 omega_body) {
		float dw = 0.5f * (-x*omega_body.x - y*omega_body.y - z*omega_body.z);
		float dx = 0.5f * ( w*omega_body.x + z*omega_body.y - y*omega_body.z);
		float dy = 0.5f * (-z*omega_body.x + w*omega_body.y + x*omega_body.z);
		float dz = 0.5f * ( y*omega_body.x - x*omega_body.y + w*omega_body.z);
		w += dt*dw;
		x += dt*dx;
		y += dt*dy;
		z += dt*dz;
	}
	
	// extract orientation vector "p" from quaternion 
	// assuming initial "p" was along z-axis (0,0,1) (this is used for rigid discs simulations)
	inline __host__ __device__ float3 orientation_vec() {
		float3 p;
		p.x = 2.0f*(x*z + w*y);
		p.y = 2.0f*(y*z - w*x);
		p.z = w*w - x*x - y*y + z*z;
		return p;
	}
	
	// overload operator* for (quaternion * float3)
	// implementation uses the Goldsmith's optimized formula: 
	// v' = v + 2 * cross(q.vec, cross(q.vec, v) + q.w * v)
	inline __host__ __device__ float3 operator*(float3 v) {
		float3 q_vec = make_float3(x,y,z);
		float3 t = cross(q_vec,v)*2.0f;
		float3 rotated_vector = v + (t*w) + cross(q_vec,t);
		return rotated_vector;
	}
	
	// overload operator* for (quaternion * quaternion)
	// computes Hamilton product: q_new = (*this) * q
	inline __host__ __device__ quaternion operator*(const quaternion& q) const {
		quaternion qnew;
		qnew.w = w * q.w - x * q.x - y * q.y - z * q.z;
		qnew.x = w * q.x + x * q.w + y * q.z - z * q.y;
		qnew.y = w * q.y - x * q.z + y * q.w + z * q.x;
		qnew.z = w * q.z + x * q.y - y * q.x + z * q.w;
		return qnew;
	}
	
	// computes reverse Hamilton product: q_new = q * (*this)
	// useful for global-frame rotational updates
	inline __host__ __device__ quaternion premultiply(const quaternion& q) const {
		quaternion qnew;
		qnew.w = q.w * w - q.x * x - q.y * y - q.z * z;
		qnew.x = q.w * x + q.x * w + q.y * z - q.z * y;
		qnew.y = q.w * y - q.x * z + q.y * w + q.z * x;
		qnew.z = q.w * z + q.x * y - q.y * x + q.z * w;
		return qnew;
	}

};



# endif  // QUATERNION_H