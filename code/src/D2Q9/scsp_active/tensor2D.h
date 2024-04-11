# ifndef TENSOR2D_H
# define TENSOR2D_H

# include "../../Utils/helper_math.h"



// --------------------------------------------------------
// struct that defines a 2D tensor:
// --------------------------------------------------------

struct tensor2D {
	float xx,xy;
	float yx,yy;
};



// --------------------------------------------------------
// operators:
// --------------------------------------------------------

inline __host__ __device__ tensor2D operator+(tensor2D a, tensor2D b)
{
    tensor2D c;
	c.xx = a.xx + b.xx;
	c.xy = a.xy + b.xy;
	c.yx = a.yx + b.yx;
	c.yy = a.yy + b.yy;
	return c;
}

inline __host__ __device__ void operator+=(tensor2D &a, tensor2D b)
{
	a.xx += b.xx;
    a.xy += b.xy;
	a.yx += b.yx;
	a.yy += b.yy;
}

inline __host__ __device__ tensor2D operator-(tensor2D a, tensor2D b)
{
    tensor2D c;
	c.xx = a.xx - b.xx;
	c.xy = a.xy - b.xy;
	c.yx = a.yx - b.yx;
	c.yy = a.yy - b.yy;
	return c;
}

inline __host__ __device__ tensor2D operator*(tensor2D a, tensor2D b)
{
    tensor2D c;
	c.xx = a.xx*b.xx + a.xy*b.yx;
	c.xy = a.xx*b.xy + a.xy*b.yy;
	c.yx = a.yx*b.xx + a.yy*b.yx;
	c.yy = a.yx*b.xy + a.yy*b.yy;
	return c;
}

inline __host__ __device__ tensor2D operator*(float a, tensor2D b)
{
    tensor2D c;
	c.xx = a*b.xx;
	c.xy = a*b.xy;
	c.yx = a*b.yx;
	c.yy = a*b.yy;	
	return c;
}

inline __host__ __device__ float2 operator*(tensor2D a, float2 b)
{
    float2 c = make_float2(0.0f,0.0f);
	c.x = a.xx*b.x + a.xy*b.y;
	c.y = a.yx*b.x + a.yy*b.y;
	return c;
}

inline __host__ __device__ void operator*=(tensor2D &a, float b)
{
	a.xx *= b;
    a.xy *= b;
	a.yx *= b;
	a.yy *= b;
}

inline __host__ __device__ tensor2D identity2D()
{
    tensor2D c;
	c.xx = 1.0f;
	c.xy = 0.0f;
	c.yx = 0.0f; 
	c.yy = 1.0f;
	return c;
}

inline __host__ __device__ tensor2D dyadic(float2 a)
{
    tensor2D c;
	c.xx = a.x*a.x;
	c.xy = a.x*a.y;
	c.yx = a.y*a.x;
	c.yy = a.y*a.y;
	return c;
}

inline __host__ __device__ tensor2D dyadic(float2 a, float2 b)
{
    tensor2D c;
	c.xx = a.x*b.x;
	c.xy = a.x*b.y;
	c.yx = a.y*b.x;
	c.yy = a.y*b.y;
	return c;
}

inline __host__ __device__ tensor2D transpose(tensor2D a)
{
    tensor2D c = a;
	c.xy = a.yx;
	c.yx = a.xy;	
	return c;
}


# endif  // TENSOR2D_H
