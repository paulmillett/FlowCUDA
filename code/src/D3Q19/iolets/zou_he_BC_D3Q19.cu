
# include "zou_he_BC_D3Q19.cuh"
# include <stdio.h>

// --------------------------------------------------------
// Zou-He boundary conditions:
// --------------------------------------------------------

__device__ void zou_he_BC_D3Q19(int vtype,
                                float* ft,
							    iolet* iolets)
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
		const float vBC = iolets[ioi].vBC;
		const float wBC = iolets[ioi].wBC;
		const float rhoBC = (ft[0]+ft[3]+ft[4]+ft[5]+ft[6]+ft[11]+ft[12]+ft[17]+ft[18]+
		                     2.0*(ft[1]+ft[7]+ft[9]+ft[13]+ft[15]))/(1.0 + uBC);
	    const float nxy = 0.5*(ft[3]+ft[11]+ft[17] - (ft[4]+ft[12]+ft[18])) - vBC*rhoBC/3.0;			
		const float nxz = 0.5*(ft[5]+ft[11]+ft[18] - (ft[6]+ft[12]+ft[17])) - wBC*rhoBC/3.0;			
		ft[2] = ft[1] - rhoBC*uBC/3.0;
		ft[8] = ft[7] + rhoBC*(-uBC-vBC)/6.0 + nxy;	
		ft[10] = ft[9] + rhoBC*(-uBC-wBC)/6.0 + nxz;
		ft[14] = ft[13] + rhoBC*(-uBC+vBC)/6.0 - nxy;
		ft[16] = ft[15] + rhoBC*(-uBC+wBC)/6.0 - nxz;		
	}
	
	// ----------------------------------------------
	// Zou-He velocity boundary (West)...
	// ----------------------------------------------
	
	else if (iotype == 2) {
		const float uBC = iolets[ioi].uBC;
		const float vBC = iolets[ioi].vBC;
		const float wBC = iolets[ioi].wBC;
		const float rhoBC = (ft[0]+ft[3]+ft[4]+ft[5]+ft[6]+ft[11]+ft[12]+ft[17]+ft[18]+
		                     2.0*(ft[2]+ft[8]+ft[10]+ft[14]+ft[16]))/(1.0 - uBC);		
	    const float nxy = 0.5*(ft[3]+ft[11]+ft[17] - (ft[4]+ft[12]+ft[18])) - vBC*rhoBC/3.0;			
		const float nxz = 0.5*(ft[5]+ft[11]+ft[18] - (ft[6]+ft[12]+ft[17])) - wBC*rhoBC/3.0;
		ft[1] = ft[2] + rhoBC*uBC/3.0;		
		ft[7] = ft[8] + rhoBC*(uBC+vBC)/6.0 - nxy;
		ft[9] = ft[10] + rhoBC*(uBC+wBC)/6.0 - nxz;		
		ft[13] = ft[14] + rhoBC*(uBC-vBC)/6.0 + nxy;
		ft[15] = ft[16] + rhoBC*(uBC-wBC)/6.0 + nxz;
		
	}
	
	// ----------------------------------------------
	// Zou-He velocity boundary (North)...
	// ----------------------------------------------
	
	else if (iotype == 3) {
		const float uBC = iolets[ioi].uBC;
		const float vBC = iolets[ioi].vBC;
		const float wBC = iolets[ioi].wBC;
		const float rhoBC = (ft[0]+ft[1]+ft[2]+ft[5]+ft[6]+ft[9]+ft[10]+ft[15]+ft[16]+
		                     2.0*(ft[3]+ft[7]+ft[11]+ft[14]+ft[17]))/(1.0 + vBC);
	    const float nyx = 0.5*(ft[1]+ft[9]+ft[15] - (ft[2]+ft[10]+ft[16])) - uBC*rhoBC/3.0;			
		const float nyz = 0.5*(ft[5]+ft[9]+ft[16] - (ft[6]+ft[10]+ft[15])) - wBC*rhoBC/3.0;					
		ft[4] = ft[3] - rhoBC*vBC/3.0;		
		ft[8] = ft[7] + rhoBC*(-vBC - uBC)/6 + nyx;		
		ft[12] = ft[11] + rhoBC*(-vBC - wBC)/6.0 + nyz;
		ft[13] = ft[14] + rhoBC*(-vBC + uBC)/6.0 - nyx;		
		ft[18] = ft[17] + rhoBC*(-vBC + wBC)/6.0 - nyz;
	}	
	
	// ----------------------------------------------
	// Zou-He velocity boundary (South)...
	// ----------------------------------------------
	
	else if (iotype == 4) {
		const float uBC = iolets[ioi].uBC;
		const float vBC = iolets[ioi].vBC;
		const float wBC = iolets[ioi].wBC;
		const float rhoBC = (ft[0]+ft[1]+ft[2]+ft[5]+ft[6]+ft[9]+ft[10]+ft[15]+ft[16]+
		                     2.0*(ft[4]+ft[8]+ft[12]+ft[13]+ft[18]))/(1.0 - vBC);	
	    const float nyx = 0.5*(ft[1]+ft[9]+ft[15] - (ft[2]+ft[10]+ft[16])) - uBC*rhoBC/3.0;			
		const float nyz = 0.5*(ft[5]+ft[9]+ft[16] - (ft[6]+ft[10]+ft[15])) - wBC*rhoBC/3.0;		
		ft[3] = ft[4] + rhoBC*vBC/3.0;			
		ft[7] = ft[8] + rhoBC*(vBC + uBC)/6.0 - nyx;		
		ft[11] = ft[12] + rhoBC*(vBC + wBC)/6.0 - nyz;		
		ft[14] = ft[13] + rhoBC*(vBC - uBC)/6.0 + nyx;		
		ft[17] = ft[18] + rhoBC*(vBC - wBC)/6.0 + nyz;
	}	
	
	// ----------------------------------------------
	// Zou-He velocity boundary (Top)...
	// ----------------------------------------------
	
	else if (iotype == 5) {
		const float uBC = iolets[ioi].uBC;
		const float vBC = iolets[ioi].vBC;
		const float wBC = iolets[ioi].wBC;
		const float rhoBC = (ft[0]+ft[1]+ft[2]+ft[3]+ft[4]+ft[7]+ft[8]+ft[13]+ft[14]+
		                     2.0*(ft[5]+ft[9]+ft[11]+ft[16]+ft[18]))/(1.0 + wBC);
	    const float nzx = 0.5*(ft[1]+ft[7]+ft[13] - (ft[2]+ft[8]+ft[14])) - uBC*rhoBC/3.0;			
		const float nzy = 0.5*(ft[3]+ft[7]+ft[14] - (ft[4]+ft[8]+ft[13])) - vBC*rhoBC/3.0;		
		ft[6] = ft[5] - rhoBC*wBC/3.0;		
		ft[10] = ft[9] + rhoBC*(-wBC - uBC)/6.0 + nzx;
		ft[12] = ft[11] + rhoBC*(-wBC - vBC)/6.0 + nzy;		
		ft[15] = ft[16] + rhoBC*(-wBC + uBC)/6.0 - nzx;
		ft[17] = ft[18] + rhoBC*(-wBC + vBC)/6.0 - nzy;
	}	
	
	// ----------------------------------------------
	// Zou-He velocity boundary (Bottom)...
	// ----------------------------------------------
	
	else if (iotype == 6) {
		const float uBC = iolets[ioi].uBC;
		const float vBC = iolets[ioi].vBC;
		const float wBC = iolets[ioi].wBC;
		const float rhoBC = (ft[0]+ft[1]+ft[2]+ft[3]+ft[4]+ft[7]+ft[8]+ft[13]+ft[14]+
		                     2.0*(ft[6]+ft[10]+ft[12]+ft[15]+ft[17]))/(1.0 - wBC);	
	    const float nzx = 0.5*(ft[1]+ft[7]+ft[13] - (ft[2]+ft[8]+ft[14])) - uBC*rhoBC/3.0;			
		const float nzy = 0.5*(ft[3]+ft[7]+ft[14] - (ft[4]+ft[8]+ft[13])) - vBC*rhoBC/3.0;		
		ft[5] = ft[6] + rhoBC*wBC/3.0;		
		ft[9] = ft[10] + rhoBC*(wBC + uBC)/6.0 - nzx;
		ft[11] = ft[12] + rhoBC*(wBC + vBC)/6.0 - nzy;		
		ft[16] = ft[15] + rhoBC*(wBC - uBC)/6.0 + nzx;
		ft[18] = ft[17] + rhoBC*(wBC - vBC)/6.0 + nzy;
	}	
	
	// ----------------------------------------------
	// Zou-He pressure boundary (East)...
	// ----------------------------------------------
	
	else if (iotype == 11) {
		const float vBC = iolets[ioi].vBC;
		const float wBC = iolets[ioi].wBC;
		const float rhoBC = iolets[ioi].rBC;
		const float uBC = -1.0 + (ft[0]+ft[3]+ft[4]+ft[5]+ft[6]+ft[11]+ft[12]+ft[17]+ft[18]+
		                          2.0*(ft[1]+ft[7]+ft[9]+ft[13]+ft[15]))/rhoBC;				
	    const float nxy = 0.5*(ft[3]+ft[11]+ft[17] - (ft[4]+ft[12]+ft[18])) - vBC*rhoBC/3.0;			
		const float nxz = 0.5*(ft[5]+ft[11]+ft[18] - (ft[6]+ft[12]+ft[17])) - wBC*rhoBC/3.0;		
		ft[2] = ft[1] - rhoBC*uBC/3.0;		
		ft[8] = ft[7] + rhoBC*(-uBC-vBC)/6.0 + nxy;	
		ft[10] = ft[9] + rhoBC*(-uBC-wBC)/6.0 + nxz;		
		ft[14] = ft[13] + rhoBC*(-uBC+vBC)/6.0 - nxy;
		ft[16] = ft[15] + rhoBC*(-uBC+wBC)/6.0 - nxz;		
	}	
	
	// ----------------------------------------------
	// Zou-He pressure boundary (West)...
	// ----------------------------------------------
	
	else if (iotype == 12) {			
		const float vBC = iolets[ioi].vBC;
		const float wBC = iolets[ioi].wBC;
		const float rhoBC = iolets[ioi].rBC;
		const float uBC = 1.0 - (ft[0]+ft[3]+ft[4]+ft[5]+ft[6]+ft[11]+ft[12]+ft[17]+ft[18]+
		                         2.0*(ft[2]+ft[8]+ft[10]+ft[14]+ft[16]))/rhoBC;	
		const float nxy = 0.5*(ft[3]+ft[11]+ft[17] - (ft[4]+ft[12]+ft[18])) - vBC*rhoBC/3.0;			
		const float nxz = 0.5*(ft[5]+ft[11]+ft[18] - (ft[6]+ft[12]+ft[17])) - wBC*rhoBC/3.0;
		ft[1] = ft[2] + rhoBC*uBC/3.0;		
		ft[7] = ft[8] + rhoBC*(uBC+vBC)/6.0 - nxy;
		ft[9] = ft[10] + rhoBC*(uBC+wBC)/6.0 - nxz;		
		ft[13] = ft[14] + rhoBC*(uBC-vBC)/6.0 + nxy;
		ft[15] = ft[16] + rhoBC*(uBC-wBC)/6.0 + nxz;		
	}	
	
	// ----------------------------------------------
	// Zou-He pressure boundary (North)...
	// ----------------------------------------------
	
	else if (iotype == 13) {
		const float uBC = iolets[ioi].uBC;
		const float wBC = iolets[ioi].wBC;
		const float rhoBC = iolets[ioi].rBC;
		const float vBC = -1.0 + (ft[0]+ft[1]+ft[2]+ft[5]+ft[6]+ft[9]+ft[10]+ft[15]+ft[16]+
				                  2.0*(ft[3]+ft[7]+ft[11]+ft[14]+ft[17]))/rhoBC;
	    const float nyx = 0.5*(ft[1]+ft[9]+ft[15] - (ft[2]+ft[10]+ft[16])) - uBC*rhoBC/3.0;			
		const float nyz = 0.5*(ft[5]+ft[9]+ft[16] - (ft[6]+ft[10]+ft[15])) - wBC*rhoBC/3.0;					
		ft[4] = ft[3] - rhoBC*vBC/3.0;		
		ft[8] = ft[7] + rhoBC*(-vBC - uBC)/6 + nyx;		
		ft[12] = ft[11] + rhoBC*(-vBC - wBC)/6.0 + nyz;
		ft[13] = ft[14] + rhoBC*(-vBC + uBC)/6.0 - nyx;		
		ft[18] = ft[17] + rhoBC*(-vBC + wBC)/6.0 - nyz;
	}	
	
	// ----------------------------------------------
	// Zou-He pressure boundary (South)...
	// ----------------------------------------------
	
	else if (iotype == 14) {
		const float uBC = iolets[ioi].uBC;
		const float wBC = iolets[ioi].wBC;
		const float rhoBC = iolets[ioi].rBC;
		const float vBC = 1.0 - (ft[0]+ft[1]+ft[2]+ft[5]+ft[6]+ft[9]+ft[10]+ft[15]+ft[16]+
				                 2.0*(ft[4]+ft[8]+ft[12]+ft[13]+ft[18]))/rhoBC;		
	    const float nyx = 0.5*(ft[1]+ft[9]+ft[15] - (ft[2]+ft[10]+ft[16])) - uBC*rhoBC/3.0;			
		const float nyz = 0.5*(ft[5]+ft[9]+ft[16] - (ft[6]+ft[10]+ft[15])) - wBC*rhoBC/3.0;		
		ft[3] = ft[4] + rhoBC*vBC/3.0;			
		ft[7] = ft[8] + rhoBC*(vBC + uBC)/6.0 - nyx;		
		ft[11] = ft[12] + rhoBC*(vBC + wBC)/6.0 - nyz;		
		ft[14] = ft[13] + rhoBC*(vBC - uBC)/6.0 + nyx;		
		ft[17] = ft[18] + rhoBC*(vBC - wBC)/6.0 + nyz;
	}	
	
	// ----------------------------------------------
	// Zou-He pressure boundary (Top)...
	// ----------------------------------------------
	
	else if (iotype == 15) {
		const float uBC = iolets[ioi].uBC;
		const float vBC = iolets[ioi].vBC;
		const float rhoBC = iolets[ioi].rBC;
		const float wBC = -1.0 + (ft[0]+ft[1]+ft[2]+ft[3]+ft[4]+ft[7]+ft[8]+ft[13]+ft[14]+
				                  2.0*(ft[5]+ft[9]+ft[11]+ft[16]+ft[18]))/rhoBC;	
	    const float nzx = 0.5*(ft[1]+ft[7]+ft[13] - (ft[2]+ft[8]+ft[14])) - uBC*rhoBC/3.0;			
		const float nzy = 0.5*(ft[3]+ft[7]+ft[14] - (ft[4]+ft[8]+ft[13])) - vBC*rhoBC/3.0;		
		ft[6] = ft[5] - rhoBC*wBC/3.0;		
		ft[10] = ft[9] + rhoBC*(-wBC - uBC)/6.0 + nzx;
		ft[12] = ft[11] + rhoBC*(-wBC - vBC)/6.0 + nzy;		
		ft[15] = ft[16] + rhoBC*(-wBC + uBC)/6.0 - nzx;
		ft[17] = ft[18] + rhoBC*(-wBC + vBC)/6.0 - nzy;
	}	
	
	// ----------------------------------------------
	// Zou-He pressure boundary (Bottom)...
	// ----------------------------------------------
	
	else if (iotype == 16) {
		const float uBC = iolets[ioi].uBC;
		const float vBC = iolets[ioi].vBC;
		const float rhoBC = iolets[ioi].rBC;
		const float wBC = 1.0 - (ft[0]+ft[1]+ft[2]+ft[3]+ft[4]+ft[7]+ft[8]+ft[13]+ft[14]+
				                 2.0*(ft[6]+ft[10]+ft[12]+ft[15]+ft[17]))/rhoBC;
	    const float nzx = 0.5*(ft[1]+ft[7]+ft[13] - (ft[2]+ft[8]+ft[14])) - uBC*rhoBC/3.0;			
		const float nzy = 0.5*(ft[3]+ft[7]+ft[14] - (ft[4]+ft[8]+ft[13])) - vBC*rhoBC/3.0;		
		ft[5] = ft[6] + rhoBC*wBC/3.0;		
		ft[9] = ft[10] + rhoBC*(wBC + uBC)/6.0 - nzx;
		ft[11] = ft[12] + rhoBC*(wBC + vBC)/6.0 - nzy;		
		ft[16] = ft[15] + rhoBC*(wBC - uBC)/6.0 + nzx;
		ft[18] = ft[17] + rhoBC*(wBC - vBC)/6.0 + nzy;
	}		
	
}