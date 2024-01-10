# ifndef TENSOR_H
# define TENSOR_H

# include "../../../Utils/helper_math.h"



// --------------------------------------------------------
// struct that defines a bead on a rod in an IBM mesh:
// --------------------------------------------------------

struct tensor {
	float xx,xy,xz;
	float yx,yy,yz;
	float zx,zy,zz;
};



// --------------------------------------------------------
// operators:
// --------------------------------------------------------

inline __host__ __device__ tensor operator+(tensor a, tensor b)
{
    tensor c;
	c.xx = a.xx+b.xx; c.xy = a.xy+b.xy; c.xz = a.xz+b.xz;
	c.yx = a.yx+b.yx; c.yy = a.yy+b.yy; c.yz = a.yz+b.yz;
	c.zx = a.zx+b.zx; c.zy = a.zy+b.zy; c.zz = a.zz+b.zz;
	return c;
}

inline __host__ __device__ tensor operator-(tensor a, tensor b)
{
    tensor c;
	c.xx = a.xx-b.xx; c.xy = a.xy-b.xy; c.xz = a.xz-b.xz;
	c.yx = a.yx-b.yx; c.yy = a.yy-b.yy; c.yz = a.yz-b.yz;
	c.zx = a.zx-b.zx; c.zy = a.zy-b.zy; c.zz = a.zz-b.zz;
	return c;
}

inline __host__ __device__ tensor operator*(tensor a, tensor b)
{
    tensor c;
	c.xx = a.xx*b.xx + a.xy*b.yx + a.xz*b.zx;
	c.xy = a.xx*b.xy + a.xy*b.yy + a.xz*b.zy;
	c.xz = a.xx*b.xz + a.xy*b.yz + a.xz*b.zz;	
	c.yx = a.yx*b.xx + a.yy*b.yx + a.yz*b.zx;
	c.yy = a.yx*b.xy + a.yy*b.yy + a.yz*b.zy;
	c.yz = a.yx*b.xz + a.yy*b.yz + a.yz*b.zz;	
	c.zx = a.zx*b.xx + a.zy*b.yx + a.zz*b.zx;
	c.zy = a.zx*b.xy + a.zy*b.yy + a.zz*b.zy;
	c.zz = a.zx*b.xz + a.zy*b.yz + a.zz*b.zz;
	return c;
}

inline __host__ __device__ tensor operator*(float a, tensor b)
{
    tensor c;
	c.xx = a*b.xx;
	c.xy = a*b.xy;
	c.xz = a*b.xz;
	c.yx = a*b.yx;
	c.yy = a*b.yy;
	c.yz = a*b.yz;
	c.zx = a*b.zx;
	c.zy = a*b.zy;
	c.zz = a*b.zz;
	return c;
}

inline __host__ __device__ float3 operator*(tensor a, float3 b)
{
    float3 c = make_float3(0.0f,0.0f,0.0f);
	c.x = a.xx*b.x + a.xy*b.y + a.xz*b.z;
	c.y = a.yx*b.x + a.yy*b.y + a.yz*b.z;
	c.z = a.zx*b.x + a.zy*b.y + a.zz*b.z;
	return c;
}

inline __host__ __device__ tensor identity()
{
    tensor c;
	c.xx = 1.0f; c.xy = 0.0f; c.xz = 0.0f;
	c.yx = 0.0f; c.yy = 1.0f; c.yz = 0.0f;
	c.zx = 0.0f; c.zy = 0.0f; c.zz = 1.0f;
	return c;
}

inline __host__ __device__ tensor dyadic(float3 a)
{
    tensor c;
	float axx = a.x*a.x;
	float axy = a.x*a.y;
	float axz = a.x*a.z;
	float ayy = a.y*a.y;
	float ayz = a.y*a.z;
	float azz = a.z*a.z;
	c.xx = axx; c.xy = axy; c.xz = axz;
	c.yx = axy; c.yy = ayy; c.yz = ayz;
	c.zx = axz; c.zy = ayz; c.zz = azz;
	return c;
}

inline __host__ __device__ tensor transpose(tensor a)
{
    tensor c = a;
	c.xy = a.yx;
	c.yx = a.xy;
	c.xz = a.zx;
	c.zx = a.xz;
	c.yz = a.zy;
	c.zy = a.yz;	
	return c;
}




# endif  // TENSOR_H
