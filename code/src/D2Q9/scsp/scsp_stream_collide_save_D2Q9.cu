
# include "scsp_stream_collide_save_D2Q9.cuh"
# include "../iolets/zou_he_BC_D2Q9.cuh"
# include <stdio.h>



// --------------------------------------------------------
// D2Q9 update kernel.
// This algorithm is based on the optimized "stream-collide-
// save" algorithm recommended by T. Kruger in the 
// textbook: "The Lattice Boltzmann Method: Principles
// and Practice".
// --------------------------------------------------------

__global__ 
void scsp_stream_collide_save_D2Q9(
	float* f1,
    float* f2,
	float* r,
	float* u,
	float* v,
	int* streamIndex,
	int* voxelType,
	iolet2D* iolets,
	float nu,
	int nVoxels,
	bool save)
{

	// -----------------------------------------------
	// define voxel:
	// -----------------------------------------------
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
		
	if (i < nVoxels) {
		
		// --------------------------------------------------		
		// voxel-specific parameters:
		// --------------------------------------------------
		
		int vtype = voxelType[i];
		int offst = 9*i;	
		float ft[9];
		
		// --------------------------------------------------		
		// STREAMING - load populations from adjacent voxels,
		//             note	that streamIndex[] accounts for
		//             halfway bounceback conditions.
		// --------------------------------------------------
		
		ft[0] = f1[streamIndex[offst+0]];                   
		ft[1] = f1[streamIndex[offst+1]]; 
		ft[2] = f1[streamIndex[offst+2]];  
		ft[3] = f1[streamIndex[offst+3]];  
		ft[4] = f1[streamIndex[offst+4]];  
		ft[5] = f1[streamIndex[offst+5]]; 
		ft[6] = f1[streamIndex[offst+6]];  
		ft[7] = f1[streamIndex[offst+7]];  
		ft[8] = f1[streamIndex[offst+8]]; 		
				
		// --------------------------------------------------
		// COLLISION - this step is done only for fluid voxels;
		//             other treatments are performed for inlet/
		//             outlet voxels.  
		// --------------------------------------------------
		
		float rho = ft[0] + ft[1] + ft[2] + ft[3] + ft[4] + ft[5] + ft[6] + ft[7] + ft[8];
		float rhoinv = 1.0/rho;
		float ux = rhoinv*(ft[1] + ft[5] + ft[8] - (ft[3] + ft[6] + ft[7]));
		float vy = rhoinv*(ft[2] + ft[5] + ft[6] - (ft[4] + ft[7] + ft[8]));
		
		// Fluid voxel...
		if (vtype == 0) {
			
			// useful constants:
			const float tauinv = 2.0/(6.0*nu + 1.0);   // 1/tau
			const float omtauinv = 1.0 - tauinv;       // 1 - 1/tau
			const float tw0r = (4.0/9.0)*rho*tauinv;   // w[0]*rho/tau
			const float twsr = (1.0/9.0)*rho*tauinv;   // w[1-4]*rho/tau
			const float twdr = (1.0/36.0)*rho*tauinv;  // w[5-8]*rho/tau
			const float omusq = 1.0 - 1.5*(ux*ux + vy*vy);
			const float tux = 3.0*ux;
			const float tvy = 3.0*vy;	
			
			// collision calculations:	
			f2[offst+0] = omtauinv*ft[0] + tw0r*(omusq);		                      // dir-0
			float cidot3u = tux;
			f2[offst+1] = omtauinv*ft[1] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-1
			cidot3u = tvy;
			f2[offst+2] = omtauinv*ft[2] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-2
			cidot3u = -tux;
			f2[offst+3] = omtauinv*ft[3] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-3
			cidot3u = -tvy;
			f2[offst+4] = omtauinv*ft[4] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-4
			cidot3u = tux+tvy;
			f2[offst+5] = omtauinv*ft[5] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-5
			cidot3u = tvy-tux;
			f2[offst+6] = omtauinv*ft[6] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-6
			cidot3u = -(tux+tvy);
			f2[offst+7] = omtauinv*ft[7] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-7
			cidot3u = tux-tvy;
			f2[offst+8] = omtauinv*ft[8] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));  // dir-8			
		
		}
		
		// Boundary Conditions:
		else if (vtype > 0) {
			
			// decide the type of iolet:
			const int ioi = vtype - 1;  // iolet index
			const int iotype = iolets[ioi].type;
			
			// Zou-He velocity boundary (East)...
			if (iotype == 1) {
				ux = iolets[ioi].uBC;
				vy = iolets[ioi].vBC;
				rho = (ft[0]+ft[2]+ft[4] + 2.0*(ft[1]+ft[5]+ft[8])) / (1.0 + ux);
				zou_he_velo_east_D2Q9(offst,f2,ft,ux,vy,rho);
			}
			
			// Zou-He velocity boundary (West)...
			else if (iotype == 2) {
				ux = iolets[ioi].uBC;
				vy = iolets[ioi].vBC;
				rho = (ft[0]+ft[2]+ft[4] + 2.0*(ft[3]+ft[7]+ft[6])) / (1.0 - ux);
				zou_he_velo_west_D2Q9(offst,f2,ft,ux,vy,rho);
			}
			
			// Zou-He velocity boundary (North)...
			else if (iotype == 3) {
				ux = iolets[ioi].uBC;
				vy = iolets[ioi].vBC;
				rho = (ft[0]+ft[1]+ft[3] + 2.0*(ft[2]+ft[5]+ft[6])) / (1.0 + vy);
				zou_he_velo_north_D2Q9(offst,f2,ft,ux,vy,rho);
			}	
			
			// Zou-He velocity boundary (South)...
			else if (iotype == 4) {
				ux = iolets[ioi].uBC;
				vy = iolets[ioi].vBC;
				rho = (ft[0]+ft[1]+ft[3] + 2.0*(ft[4]+ft[7]+ft[8])) / (1.0 - vy);
				zou_he_velo_south_D2Q9(offst,f2,ft,ux,vy,rho);
			}	
									
			// Zou-He pressure boundary (East)...
			else if (iotype == 11) {
				vy = iolets[ioi].vBC;
				rho = iolets[ioi].rBC;
				ux = (ft[0]+ft[2]+ft[4] + 2.0*(ft[1]+ft[5]+ft[8]))/rho - 1.0;				
				zou_he_pres_east_D2Q9(offst,f2,ft,ux,vy,rho);
			}	
			
			// Zou-He pressure boundary (West)...
			else if (iotype == 12) {			
				vy = iolets[ioi].vBC;
				rho = iolets[ioi].rBC;
				ux = (ft[0]+ft[2]+ft[4] + 2.0*(ft[3]+ft[7]+ft[6]))/rho - 1.0;
				zou_he_pres_west_D2Q9(offst,f2,ft,ux,vy,rho);
			}	
			
			// Zou-He pressure boundary (North)...
			else if (iotype == 13) {
				ux = iolets[ioi].uBC;
				rho = iolets[ioi].rBC;
				vy = (ft[0]+ft[1]+ft[3] + 2.0*(ft[2]+ft[5]+ft[6]))/rho - 1.0;
				zou_he_pres_north_D2Q9(offst,f2,ft,ux,vy,rho);
			}	
			
			// Zou-He pressure boundary (South)...
			else if (iotype == 14) {
				ux = iolets[ioi].uBC;
				rho = iolets[ioi].rBC;
				vy = (ft[0]+ft[1]+ft[3] + 2.0*(ft[4]+ft[7]+ft[8]))/rho - 1.0;
				zou_he_pres_south_D2Q9(offst,f2,ft,ux,vy,rho);
			}
									
		}
	
		// --------------------------------------------------		
		// SAVE - write macros to arrays 
		// --------------------------------------------------
		
		if (save) {
			r[i] = rho;
			u[i] = ux;
			v[i] = vy;
		}
							
	}
}
