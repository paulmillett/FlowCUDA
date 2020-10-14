
# include "zou_he_BC_D3Q19.cuh"
# include <stdio.h>

// --------------------------------------------------------
// Zou-He velocity boundary condition for "west" boundaries:
// --------------------------------------------------------

__device__ void zou_he_velo_west_D3Q19(int offst,
                                       float* f2,
                                       float* ft,
                                       float uBC,
                                       float vBC,
                                       float wBC,
									   float rhoBC)
{
    const float nxy = 0.5*(ft[3]+ft[11]+ft[17] - (ft[4]+ft[12]+ft[18])) - vBC*rhoBC/3.0;			
	const float nxz = 0.5*(ft[5]+ft[11]+ft[18] - (ft[6]+ft[12]+ft[17])) - wBC*rhoBC/3.0;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[2] + rhoBC*uBC/3.0;
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[5];
	f2[offst+6] = ft[6];
	f2[offst+7] = ft[8] + rhoBC*(uBC+vBC)/6.0 - nxy;
	f2[offst+8] = ft[8];	
	f2[offst+9] = ft[10] + rhoBC*(uBC+wBC)/6.0 - nxz;
	f2[offst+10] = ft[10];
	f2[offst+11] = ft[11];
	f2[offst+12] = ft[12];
	f2[offst+13] = ft[14] + rhoBC*(uBC-vBC)/6.0 + nxy;
	f2[offst+14] = ft[14];
	f2[offst+15] = ft[16] + rhoBC*(uBC-wBC)/6.0 + nxz;
	f2[offst+16] = ft[16];
	f2[offst+17] = ft[17];
	f2[offst+18] = ft[18];
}

// --------------------------------------------------------
// Zou-He velocity boundary condition for "east" boundaries:
// --------------------------------------------------------

__device__ void zou_he_velo_east_D3Q19(int offst,
                                       float* f2,
                                       float* ft,
                                       float uBC,
                                       float vBC,
                                       float wBC,
									   float rhoBC)
{
    const float nxy = 0.5*(ft[3]+ft[11]+ft[17] - (ft[4]+ft[12]+ft[18])) - vBC*rhoBC/3.0;			
	const float nxz = 0.5*(ft[5]+ft[11]+ft[18] - (ft[6]+ft[12]+ft[17])) - wBC*rhoBC/3.0;			
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[1] - rhoBC*uBC/3.0;
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[5];
	f2[offst+6] = ft[6];
	f2[offst+7] = ft[7];
	f2[offst+8] = ft[7] + rhoBC*(-uBC-vBC)/6.0 + nxy;	
	f2[offst+9] = ft[9];
	f2[offst+10] = ft[9] + rhoBC*(-uBC-wBC)/6.0 + nxz;
	f2[offst+11] = ft[11];
	f2[offst+12] = ft[12];
	f2[offst+13] = ft[13];
	f2[offst+14] = ft[13] + rhoBC*(-uBC+vBC)/6.0 - nxy;
	f2[offst+15] = ft[15];
	f2[offst+16] = ft[15] + rhoBC*(-uBC+wBC)/6.0 - nxz;
	f2[offst+17] = ft[17];
	f2[offst+18] = ft[18];
}

// --------------------------------------------------------
// Zou-He velocity boundary condition for "south" boundaries:
// --------------------------------------------------------

__device__ void zou_he_velo_south_D3Q19(int offst,
                                        float* f2,
                                        float* ft,
                                        float uBC,
                                        float vBC,
                                        float wBC,
									    float rhoBC)
{
    const float nyx = 0.5*(ft[1]+ft[9]+ft[15] - (ft[2]+ft[10]+ft[16])) - uBC*rhoBC/3.0;			
	const float nyz = 0.5*(ft[5]+ft[9]+ft[16] - (ft[6]+ft[10]+ft[15])) - wBC*rhoBC/3.0;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[4] + rhoBC*vBC/3.0;			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[5];
	f2[offst+6] = ft[6];
	f2[offst+7] = ft[8] + rhoBC*(vBC + uBC)/6.0 - nyx;
	f2[offst+8] = ft[8];	
	f2[offst+9] = ft[9];
	f2[offst+10] = ft[10];
	f2[offst+11] = ft[12] + rhoBC*(vBC + wBC)/6.0 - nyz;
	f2[offst+12] = ft[12];
	f2[offst+13] = ft[13];
	f2[offst+14] = ft[13] + rhoBC*(vBC - uBC)/6.0 + nyx;
	f2[offst+15] = ft[15];
	f2[offst+16] = ft[16];
	f2[offst+17] = ft[18] + rhoBC*(vBC - wBC)/6.0 + nyz;
	f2[offst+18] = ft[18];
}

// --------------------------------------------------------
// Zou-He velocity boundary condition for "north" boundaries:
// --------------------------------------------------------

__device__ void zou_he_velo_north_D3Q19(int offst,
                                        float* f2,
                                        float* ft,
                                        float uBC,
                                        float vBC,
                                        float wBC,
									    float rhoBC)
{
    const float nyx = 0.5*(ft[1]+ft[9]+ft[15] - (ft[2]+ft[10]+ft[16])) - uBC*rhoBC/3.0;			
	const float nyz = 0.5*(ft[5]+ft[9]+ft[16] - (ft[6]+ft[10]+ft[15])) - wBC*rhoBC/3.0;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[3] - rhoBC*vBC/3.0;
	f2[offst+5] = ft[5];
	f2[offst+6] = ft[6];
	f2[offst+7] = ft[7];
	f2[offst+8] = ft[7] + rhoBC*(-vBC - uBC)/6 + nyx;	
	f2[offst+9] = ft[9];
	f2[offst+10] = ft[10];
	f2[offst+11] = ft[11];
	f2[offst+12] = ft[11] + rhoBC*(-vBC - wBC)/6.0 + nyz;
	f2[offst+13] = ft[14] + rhoBC*(-vBC + uBC)/6.0 - nyx;
	f2[offst+14] = ft[14];
	f2[offst+15] = ft[15];
	f2[offst+16] = ft[16];
	f2[offst+17] = ft[17];
	f2[offst+18] = ft[17] + rhoBC*(-vBC + wBC)/6.0 - nyz;
}

// --------------------------------------------------------
// Zou-He velocity boundary condition for "bottom" boundaries:
// --------------------------------------------------------

__device__ void zou_he_velo_bottom_D3Q19(int offst,
                                         float* f2,
                                         float* ft,
                                         float uBC,
                                         float vBC,
                                         float wBC,
									     float rhoBC)
{
    const float nzx = 0.5*(ft[1]+ft[7]+ft[13] - (ft[2]+ft[8]+ft[14])) - uBC*rhoBC/3.0;			
	const float nzy = 0.5*(ft[3]+ft[7]+ft[14] - (ft[4]+ft[8]+ft[13])) - vBC*rhoBC/3.0;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[6] + rhoBC*wBC/3.0;
	f2[offst+6] = ft[6];
	f2[offst+7] = ft[7];
	f2[offst+8] = ft[8];	
	f2[offst+9] = ft[10] + rhoBC*(wBC + uBC)/6.0 - nzx;
	f2[offst+10] = ft[10];
	f2[offst+11] = ft[12] + rhoBC*(wBC + vBC)/6.0 - nzy;
	f2[offst+12] = ft[12];
	f2[offst+13] = ft[13];
	f2[offst+14] = ft[14];
	f2[offst+15] = ft[15];
	f2[offst+16] = ft[15] + rhoBC*(wBC - uBC)/6.0 + nzx;
	f2[offst+17] = ft[17];
	f2[offst+18] = ft[17] + rhoBC*(wBC - vBC)/6.0 + nzy;
}

// --------------------------------------------------------
// Zou-He velocity boundary condition for "top" boundaries:
// --------------------------------------------------------

__device__ void zou_he_velo_top_D3Q19(int offst,
                                      float* f2,
                                      float* ft,
                                      float uBC,
                                      float vBC,
                                      float wBC,
									  float rhoBC)
{
    const float nzx = 0.5*(ft[1]+ft[7]+ft[13] - (ft[2]+ft[8]+ft[14])) - uBC*rhoBC/3.0;			
	const float nzy = 0.5*(ft[3]+ft[7]+ft[14] - (ft[4]+ft[8]+ft[13])) - vBC*rhoBC/3.0;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[5];
	f2[offst+6] = ft[5] - rhoBC*wBC/3.0;
	f2[offst+7] = ft[7];
	f2[offst+8] = ft[8];	
	f2[offst+9] = ft[9];
	f2[offst+10] = ft[9] + rhoBC*(-wBC - uBC)/6.0 + nzx;
	f2[offst+11] = ft[11];
	f2[offst+12] = ft[11] + rhoBC*(-wBC - vBC)/6.0 + nzy;
	f2[offst+13] = ft[13];
	f2[offst+14] = ft[14];
	f2[offst+15] = ft[16] + rhoBC*(-wBC + uBC)/6.0 - nzx;
	f2[offst+16] = ft[16];
	f2[offst+17] = ft[18] + rhoBC*(-wBC + vBC)/6.0 - nzy;
	f2[offst+18] = ft[18];
}

// --------------------------------------------------------
// Zou-He pressure boundary condition for "west" boundaries:
// --------------------------------------------------------

__device__ void zou_he_pres_west_D3Q19(int offst,
                                       float* f2,
                                       float* ft,
									   float uBC,
                                       float vBC,
                                       float wBC,
									   float rhoBC)
{
	const float nxy = 0.5*(ft[3]+ft[11]+ft[17] - (ft[4]+ft[12]+ft[18])) - vBC*rhoBC/3.0;			
	const float nxz = 0.5*(ft[5]+ft[11]+ft[18] - (ft[6]+ft[12]+ft[17])) - wBC*rhoBC/3.0;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[2] + rhoBC*uBC/3.0;
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[5];
	f2[offst+6] = ft[6];
	f2[offst+7] = ft[8] + rhoBC*(uBC+vBC)/6.0 - nxy;
	f2[offst+8] = ft[8];	
	f2[offst+9] = ft[10] + rhoBC*(uBC+wBC)/6.0 - nxz;
	f2[offst+10] = ft[10];
	f2[offst+11] = ft[11];
	f2[offst+12] = ft[12];
	f2[offst+13] = ft[14] + rhoBC*(uBC-vBC)/6.0 + nxy;
	f2[offst+14] = ft[14];
	f2[offst+15] = ft[16] + rhoBC*(uBC-wBC)/6.0 + nxz;
	f2[offst+16] = ft[16];
	f2[offst+17] = ft[17];
	f2[offst+18] = ft[18];
}

// --------------------------------------------------------
// Zou-He pressure boundary condition for "east" boundaries:
// --------------------------------------------------------

__device__ void zou_he_pres_east_D3Q19(int offst,
                                       float* f2,
                                       float* ft,
                                       float uBC,
									   float vBC,
                                       float wBC,
									   float rhoBC)
{
    const float nxy = 0.5*(ft[3]+ft[11]+ft[17] - (ft[4]+ft[12]+ft[18])) - vBC*rhoBC/3.0;			
	const float nxz = 0.5*(ft[5]+ft[11]+ft[18] - (ft[6]+ft[12]+ft[17])) - wBC*rhoBC/3.0;			
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[1] - rhoBC*uBC/3.0;
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[5];
	f2[offst+6] = ft[6];
	f2[offst+7] = ft[7];
	f2[offst+8] = ft[7] + rhoBC*(-uBC-vBC)/6.0 + nxy;	
	f2[offst+9] = ft[9];
	f2[offst+10] = ft[9] + rhoBC*(-uBC-wBC)/6.0 + nxz;
	f2[offst+11] = ft[11];
	f2[offst+12] = ft[12];
	f2[offst+13] = ft[13];
	f2[offst+14] = ft[13] + rhoBC*(-uBC+vBC)/6.0 - nxy;
	f2[offst+15] = ft[15];
	f2[offst+16] = ft[15] + rhoBC*(-uBC+wBC)/6.0 - nxz;
	f2[offst+17] = ft[17];
	f2[offst+18] = ft[18];
}

// --------------------------------------------------------
// Zou-He pressure boundary condition for "south" boundaries:
// --------------------------------------------------------

__device__ void zou_he_pres_south_D3Q19(int offst,
                                        float* f2,
                                        float* ft,
                                        float uBC,
 									    float vBC,
                                        float wBC,
 									    float rhoBC)
{
    const float nyx = 0.5*(ft[1]+ft[9]+ft[15] - (ft[2]+ft[10]+ft[16])) - uBC*rhoBC/3.0;			
	const float nyz = 0.5*(ft[5]+ft[9]+ft[16] - (ft[6]+ft[10]+ft[15])) - wBC*rhoBC/3.0;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[4] + rhoBC*vBC/3.0;			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[5];
	f2[offst+6] = ft[6];
	f2[offst+7] = ft[8] + rhoBC*(vBC + uBC)/6.0 - nyx;
	f2[offst+8] = ft[8];	
	f2[offst+9] = ft[9];
	f2[offst+10] = ft[10];
	f2[offst+11] = ft[12] + rhoBC*(vBC + wBC)/6.0 - nyz;
	f2[offst+12] = ft[12];
	f2[offst+13] = ft[13];
	f2[offst+14] = ft[13] + rhoBC*(vBC - uBC)/6.0 + nyx;
	f2[offst+15] = ft[15];
	f2[offst+16] = ft[16];
	f2[offst+17] = ft[18] + rhoBC*(vBC - wBC)/6.0 + nyz;
	f2[offst+18] = ft[18];
}

// --------------------------------------------------------
// Zou-He pressure boundary condition for "north" boundaries:
// --------------------------------------------------------

__device__ void zou_he_pres_north_D3Q19(int offst,
                                        float* f2,
                                        float* ft,
                                        float uBC,
 									    float vBC,
                                        float wBC,
 									    float rhoBC)
{
    const float nyx = 0.5*(ft[1]+ft[9]+ft[15] - (ft[2]+ft[10]+ft[16])) - uBC*rhoBC/3.0;			
	const float nyz = 0.5*(ft[5]+ft[9]+ft[16] - (ft[6]+ft[10]+ft[15])) - wBC*rhoBC/3.0;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[3] - rhoBC*vBC/3.0;
	f2[offst+5] = ft[5];
	f2[offst+6] = ft[6];
	f2[offst+7] = ft[7];
	f2[offst+8] = ft[7] + rhoBC*(-vBC - uBC)/6 + nyx;	
	f2[offst+9] = ft[9];
	f2[offst+10] = ft[10];
	f2[offst+11] = ft[11];
	f2[offst+12] = ft[11] + rhoBC*(-vBC - wBC)/6.0 + nyz;
	f2[offst+13] = ft[14] + rhoBC*(-vBC + uBC)/6.0 - nyx;
	f2[offst+14] = ft[14];
	f2[offst+15] = ft[15];
	f2[offst+16] = ft[16];
	f2[offst+17] = ft[17];
	f2[offst+18] = ft[17] + rhoBC*(-vBC + wBC)/6.0 - nyz;
}

// --------------------------------------------------------
// Zou-He pressure boundary condition for "bottom" boundaries:
// --------------------------------------------------------

__device__ void zou_he_pres_bottom_D3Q19(int offst,
                                         float* f2,
                                         float* ft,
                                         float uBC,
  									     float vBC,
                                         float wBC,
  									     float rhoBC)
{
    const float nzx = 0.5*(ft[1]+ft[7]+ft[13] - (ft[2]+ft[8]+ft[14])) - uBC*rhoBC/3.0;			
	const float nzy = 0.5*(ft[3]+ft[7]+ft[14] - (ft[4]+ft[8]+ft[13])) - vBC*rhoBC/3.0;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[6] + rhoBC*wBC/3.0;
	f2[offst+6] = ft[6];
	f2[offst+7] = ft[7];
	f2[offst+8] = ft[8];	
	f2[offst+9] = ft[10] + rhoBC*(wBC + uBC)/6.0 - nzx;
	f2[offst+10] = ft[10];
	f2[offst+11] = ft[12] + rhoBC*(wBC + vBC)/6.0 - nzy;
	f2[offst+12] = ft[12];
	f2[offst+13] = ft[13];
	f2[offst+14] = ft[14];
	f2[offst+15] = ft[15];
	f2[offst+16] = ft[15] + rhoBC*(wBC - uBC)/6.0 + nzx;
	f2[offst+17] = ft[17];
	f2[offst+18] = ft[17] + rhoBC*(wBC - vBC)/6.0 + nzy;
}

// --------------------------------------------------------
// Zou-He pressure boundary condition for "top" boundaries:
// --------------------------------------------------------

__device__ void zou_he_pres_top_D3Q19(int offst,
                                      float* f2,
                                      float* ft,
                                      float uBC,
								      float vBC,
                                      float wBC,
								      float rhoBC)
{
    const float nzx = 0.5*(ft[1]+ft[7]+ft[13] - (ft[2]+ft[8]+ft[14])) - uBC*rhoBC/3.0;			
	const float nzy = 0.5*(ft[3]+ft[7]+ft[14] - (ft[4]+ft[8]+ft[13])) - vBC*rhoBC/3.0;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[5];
	f2[offst+6] = ft[5] - rhoBC*wBC/3.0;
	f2[offst+7] = ft[7];
	f2[offst+8] = ft[8];	
	f2[offst+9] = ft[9];
	f2[offst+10] = ft[9] + rhoBC*(-wBC - uBC)/6.0 + nzx;
	f2[offst+11] = ft[11];
	f2[offst+12] = ft[11] + rhoBC*(-wBC - vBC)/6.0 + nzy;
	f2[offst+13] = ft[13];
	f2[offst+14] = ft[14];
	f2[offst+15] = ft[16] + rhoBC*(-wBC + uBC)/6.0 - nzx;
	f2[offst+16] = ft[16];
	f2[offst+17] = ft[18] + rhoBC*(-wBC + vBC)/6.0 - nzy;
	f2[offst+18] = ft[18];
}

