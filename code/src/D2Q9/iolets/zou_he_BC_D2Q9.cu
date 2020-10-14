
# include "zou_he_BC_D2Q9.cuh"
# include <stdio.h>

// --------------------------------------------------------
// Zou-He velocity boundary condition for "west" boundaries:
// --------------------------------------------------------

__device__ void zou_he_velo_west_D2Q9(int offst,
                                      float* f2,
                                      float* ft,
                                      float uBC,
                                      float vBC,
									  float rhoBC)
{
	float ru = rhoBC*uBC;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[3] + (2.0/3.0)*ru;
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[7] + (1.0/6.0)*ru - 0.5*(ft[2]-ft[4]);
	f2[offst+6] = ft[6];
	f2[offst+7] = ft[7];
	f2[offst+8] = ft[6] + (1.0/6.0)*ru - 0.5*(ft[4]-ft[2]);	
}

// --------------------------------------------------------
// Zou-He velocity boundary condition for "east" boundaries:
// --------------------------------------------------------

__device__ void zou_he_velo_east_D2Q9(int offst,
                                      float* f2,
                                      float* ft,
                                      float uBC,
                                      float vBC,
									  float rhoBC)
{
	float ru = rhoBC*uBC;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[1] - (2.0/3.0)*ru;			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[5];
	f2[offst+6] = ft[8] - (1.0/6.0)*ru + 0.5*(ft[4]-ft[2]);
	f2[offst+7] = ft[5] - (1.0/6.0)*ru + 0.5*(ft[2]-ft[4]);
	f2[offst+8] = ft[8];
}

// --------------------------------------------------------
// Zou-He velocity boundary condition for "south" boundaries:
// --------------------------------------------------------

__device__ void zou_he_velo_south_D2Q9(int offst,
                                       float* f2,
                                       float* ft,
                                       float uBC,
                                       float vBC,
									   float rhoBC)
{
	float rv = rhoBC*vBC;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[4] + (2.0/3.0)*rv;
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[7] + (1.0/6.0)*rv - 0.5*(ft[1]-ft[3]);
	f2[offst+6] = ft[8] + (1.0/6.0)*rv - 0.5*(ft[3]-ft[1]);
	f2[offst+7] = ft[7];
	f2[offst+8] = ft[8];
}

// --------------------------------------------------------
// Zou-He velocity boundary condition for "north" boundaries:
// --------------------------------------------------------

__device__ void zou_he_velo_north_D2Q9(int offst,
                                       float* f2,
                                       float* ft,
                                       float uBC,
                                       float vBC,
									   float rhoBC)
{
	float rv = rhoBC*vBC;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[2] - (2.0/3.0)*rv;
	f2[offst+5] = ft[5];
	f2[offst+6] = ft[6];
	f2[offst+7] = ft[5] - (1.0/6.0)*rv + 0.5*(ft[1]-ft[3]);
	f2[offst+8] = ft[6] - (1.0/6.0)*rv + 0.5*(ft[3]-ft[1]);
}

// --------------------------------------------------------
// Zou-He pressure boundary condition for "west" boundaries:
// --------------------------------------------------------

__device__ void zou_he_pres_west_D2Q9(int offst,
                                      float* f2,
                                      float* ft,
									  float uBC,
                                      float vBC,
                                      float rhoBC)
{
	float ru = rhoBC*uBC;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[3] + (2.0/3.0)*ru;
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[7] + (1.0/6.0)*ru + 0.5*(ft[4]-ft[2]);
	f2[offst+6] = ft[6];
	f2[offst+7] = ft[7];
	f2[offst+8] = ft[6] + (1.0/6.0)*ru + 0.5*(ft[2]-ft[4]);	
}

// --------------------------------------------------------
// Zou-He pressure boundary condition for "east" boundaries:
// --------------------------------------------------------

__device__ void zou_he_pres_east_D2Q9(int offst,
                                      float* f2,
                                      float* ft,
									  float uBC,
                                      float vBC,
                                      float rhoBC)
{
	float ru = rhoBC*uBC;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[1] - (2.0/3.0)*ru;			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[5];
	f2[offst+6] = ft[8] - (1.0/6.0)*ru + 0.5*(ft[4]-ft[2]);
	f2[offst+7] = ft[5] - (1.0/6.0)*ru + 0.5*(ft[2]-ft[4]);
	f2[offst+8] = ft[8];
}

// --------------------------------------------------------
// Zou-He pressure boundary condition for "south" boundaries:
// --------------------------------------------------------

__device__ void zou_he_pres_south_D2Q9(int offst,
                                       float* f2,
                                       float* ft,
                                       float uBC,
									   float vBC,
                                       float rhoBC)
{
	float rv = rhoBC*vBC;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[4] - (2.0/3.0)*rv;
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[4];
	f2[offst+5] = ft[7] - (1.0/6.0)*rv + 0.5*(ft[3]-ft[1]);
	f2[offst+6] = ft[8] - (1.0/6.0)*rv + 0.5*(ft[1]-ft[3]);
	f2[offst+7] = ft[7];
	f2[offst+8] = ft[8];
}

// --------------------------------------------------------
// Zou-He pressure boundary condition for "north" boundaries:
// --------------------------------------------------------

__device__ void zou_he_pres_north_D2Q9(int offst,
                                       float* f2,
                                       float* ft,
                                       float uBC,
									   float vBC,
                                       float rhoBC)
{
	float rv = rhoBC*vBC;
	f2[offst+0] = ft[0];
	f2[offst+1] = ft[1];
	f2[offst+2] = ft[2];
	f2[offst+3] = ft[3];			
	f2[offst+4] = ft[2] - (2.0/3.0)*rv;
	f2[offst+5] = ft[5];
	f2[offst+6] = ft[6];
	f2[offst+7] = ft[5] - (1.0/6.0)*rv + 0.5*(ft[1]-ft[3]);
	f2[offst+8] = ft[6] - (1.0/6.0)*rv + 0.5*(ft[3]-ft[1]);
}


