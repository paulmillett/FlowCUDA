
# include "zou_he_BC_D2Q9.cuh"
# include <stdio.h>



// --------------------------------------------------------
// Zou-He velocity boundary condition for "west" boundaries:
// --------------------------------------------------------

__device__ void zou_he_BC_D2Q9(int vtype,
                               float* ft,
							   iolet2D* iolets)
{
	
	// ----------------------------------------------
	// get iolet type:
	// ----------------------------------------------
		
	const int ioi = vtype - 1;  // iolet index
	const int iotype = iolets[ioi].type; 
	
	// ----------------------------------------------
	// Zou-He velocity boundary (East)...
	// ----------------------------------------------
	
	if (iotype == 1) {
		const float uBC = iolets[ioi].uBC;
		const float rhoBC = (ft[0]+ft[2]+ft[4] + 2.0*(ft[1]+ft[5]+ft[8])) / (1.0 + uBC);
		const float ru = rhoBC*uBC;
		ft[3] = ft[1] - (2.0/3.0)*ru;			
		ft[6] = ft[8] - (1.0/6.0)*ru + 0.5*(ft[4]-ft[2]);
		ft[7] = ft[5] - (1.0/6.0)*ru + 0.5*(ft[2]-ft[4]);
	}
	
	// ----------------------------------------------
	// Zou-He velocity boundary (West)...
	// ----------------------------------------------
	
	else if (iotype == 2) {
		const float uBC = iolets[ioi].uBC;
		const float rhoBC = (ft[0]+ft[2]+ft[4] + 2.0*(ft[3]+ft[7]+ft[6])) / (1.0 - uBC);
		const float ru = rhoBC*uBC;
		ft[1] = ft[3] + (2.0/3.0)*ru;		
		ft[5] = ft[7] + (1.0/6.0)*ru - 0.5*(ft[2]-ft[4]);		
		ft[8] = ft[6] + (1.0/6.0)*ru - 0.5*(ft[4]-ft[2]);	
	}
	
	// ----------------------------------------------
	// Zou-He velocity boundary (North)...
	// ----------------------------------------------
	
	else if (iotype == 3) {
		const float vBC = iolets[ioi].vBC;
		const float rhoBC = (ft[0]+ft[1]+ft[3] + 2.0*(ft[2]+ft[5]+ft[6])) / (1.0 + vBC);
		const float rv = rhoBC*vBC;				
		ft[4] = ft[2] - (2.0/3.0)*rv;		
		ft[7] = ft[5] - (1.0/6.0)*rv + 0.5*(ft[1]-ft[3]);
		ft[8] = ft[6] - (1.0/6.0)*rv + 0.5*(ft[3]-ft[1]);
	}	
	
	// ----------------------------------------------
	// Zou-He velocity boundary (South)...
	// ----------------------------------------------
	
	else if (iotype == 4) {
		const float vBC = iolets[ioi].vBC;
		const float rhoBC = (ft[0]+ft[1]+ft[3] + 2.0*(ft[4]+ft[7]+ft[8])) / (1.0 - vBC);
		const float rv = rhoBC*vBC;		
		ft[2] = ft[4] + (2.0/3.0)*rv;		
		ft[5] = ft[7] + (1.0/6.0)*rv - 0.5*(ft[1]-ft[3]);
		ft[6] = ft[8] + (1.0/6.0)*rv - 0.5*(ft[3]-ft[1]);		
	}	
	
	// ----------------------------------------------					
	// Zou-He pressure boundary (East)...
	// ----------------------------------------------
	
	else if (iotype == 11) {
		const float rhoBC = iolets[ioi].rBC;
		const float uBC = (ft[0]+ft[2]+ft[4] + 2.0*(ft[1]+ft[5]+ft[8]))/rhoBC - 1.0;				
		const float ru = rhoBC*uBC;		
		ft[3] = ft[1] - (2.0/3.0)*ru;			
		ft[6] = ft[8] - (1.0/6.0)*ru + 0.5*(ft[4]-ft[2]);
		ft[7] = ft[5] - (1.0/6.0)*ru + 0.5*(ft[2]-ft[4]);
	}	
	
	// ----------------------------------------------
	// Zou-He pressure boundary (West)...
	// ----------------------------------------------
	
	else if (iotype == 12) {			
		const float rhoBC = iolets[ioi].rBC;
		const float uBC = (ft[0]+ft[2]+ft[4] + 2.0*(ft[3]+ft[7]+ft[6]))/rhoBC - 1.0;
		const float ru = rhoBC*uBC;
		ft[1] = ft[3] + (2.0/3.0)*ru;		
		ft[5] = ft[7] + (1.0/6.0)*ru + 0.5*(ft[4]-ft[2]);		
		ft[8] = ft[6] + (1.0/6.0)*ru + 0.5*(ft[2]-ft[4]);	
	}	
	
	// ----------------------------------------------
	// Zou-He pressure boundary (North)...
	// ----------------------------------------------
	
	else if (iotype == 13) {
		const float rhoBC = iolets[ioi].rBC;
		const float vBC = (ft[0]+ft[1]+ft[3] + 2.0*(ft[2]+ft[5]+ft[6]))/rhoBC - 1.0;
		const float rv = rhoBC*vBC;
		ft[4] = ft[2] - (2.0/3.0)*rv;
		ft[7] = ft[5] - (1.0/6.0)*rv + 0.5*(ft[1]-ft[3]);
		ft[8] = ft[6] - (1.0/6.0)*rv + 0.5*(ft[3]-ft[1]);
	}	
	
	// ----------------------------------------------
	// Zou-He pressure boundary (South)...
	// ----------------------------------------------
	
	else if (iotype == 14) {
		const float rhoBC = iolets[ioi].rBC;
		const float vBC = (ft[0]+ft[1]+ft[3] + 2.0*(ft[4]+ft[7]+ft[8]))/rhoBC - 1.0;
		const float rv = rhoBC*vBC;		
		ft[2] = ft[4] - (2.0/3.0)*rv;		
		ft[5] = ft[7] - (1.0/6.0)*rv + 0.5*(ft[3]-ft[1]);
		ft[6] = ft[8] - (1.0/6.0)*rv + 0.5*(ft[1]-ft[3]);		
	}
	
}
	
	
	
	
	
/*    OLD CODE...	

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

*/
