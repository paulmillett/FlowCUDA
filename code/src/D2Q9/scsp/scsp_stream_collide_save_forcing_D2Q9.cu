
# include "scsp_stream_collide_save_forcing_D2Q9.cuh"
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
void scsp_stream_collide_save_forcing_D2Q9(
	float* f1,
    float* f2,
	float* r,
	float* u,
	float* v,
	float* Fx,
	float* Fy,
	int* streamIndex,
	int* voxelType,
	iolet2D* iolets,
	float nu,
	int nVoxels)
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
		//             outlet voxels.  Here, we include Guo
		//             forcing in the collision term.
		// --------------------------------------------------
		
		float rho = ft[0] + ft[1] + ft[2] + ft[3] + ft[4] + ft[5] + ft[6] + ft[7] + ft[8];
		float rhoinv = 1.0/rho;
		float ux = rhoinv*(ft[1] + ft[5] + ft[8] - (ft[3] + ft[6] + ft[7]) + 0.5*Fx[i]);
		float vy = rhoinv*(ft[2] + ft[5] + ft[6] - (ft[4] + ft[7] + ft[8]) + 0.5*Fy[i]);
		
		// Fluid voxel...
		if (vtype == 0) {
			
			// useful constants:
			const float w0 = 4.0/9.0;
			const float ws = 1.0/9.0;
			const float wd = 1.0/36.0;			
			const float omega = 2.0/(6.0*nu + 1.0);   // 1/tau
			const float omomega = 1.0 - omega;        // 1 - 1/tau
			const float omomega2 = 1.0 - 0.5*omega;   // 1 - 1/(2tau)
			const float omusq = 1.0 - 1.5*(ux*ux + vy*vy);
						
			// direction 0
			float evel = 0.0;       // e dot velocity
			float emiu = 0.0-ux;    // e minus u
			float emiv = 0.0-vy;    // e minus v
			float feq = w0*rho*omusq;
			float frc = w0*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv));
			f2[offst+0] = omomega*ft[0] + omega*feq + omomega2*frc;
			
			// direction 1
			evel = ux;
			emiu = 1.0-ux;
			emiv = 0.0-vy;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv));
			f2[offst+1] = omomega*ft[1] + omega*feq + omomega2*frc;
			
			// direction 2
			evel = vy; 
			emiu = 0.0-ux;
			emiv = 1.0-vy;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv + 9.0*evel));
			f2[offst+2] = omomega*ft[2] + omega*feq + omomega2*frc;
			
			// direction 3
			evel = -ux;
			emiu = -1.0-ux;
			emiv =  0.0-vy;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv));
			f2[offst+3] = omomega*ft[3] + omega*feq + omomega2*frc;
			
			// direction 4
			evel = -vy;
			emiu =  0.0-ux;
			emiv = -1.0-vy;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx[i]*(3.0*emiu) + Fy[i]*(3.0*emiv - 9.0*evel));
			f2[offst+4] = omomega*ft[4] + omega*feq + omomega2*frc;
			
			// direction 5
			evel = ux + vy;
			emiu = 1.0-ux;
			emiv = 1.0-vy;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv + 9.0*evel));
			f2[offst+5] = omomega*ft[5] + omega*feq + omomega2*frc;
			
			// direction 6
			evel = -ux + vy;
			emiu = -1.0-ux;
			emiv =  1.0-vy;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv + 9.0*evel));
			f2[offst+6] = omomega*ft[6] + omega*feq + omomega2*frc;
			
			// direction 7
			evel = -ux - vy;
			emiu = -1.0-ux;
			emiv = -1.0-vy;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu - 9.0*evel) + Fy[i]*(3.0*emiv - 9.0*evel));
			f2[offst+7] = omomega*ft[7] + omega*feq + omomega2*frc;
			
			// direction 8
			evel = ux - vy;
			emiu =  1.0-ux;
			emiv = -1.0-vy;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx[i]*(3.0*emiu + 9.0*evel) + Fy[i]*(3.0*emiv - 9.0*evel));
			f2[offst+8] = omomega*ft[8] + omega*feq + omomega2*frc;
									
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
		
		r[i] = rho;
		u[i] = ux;
		v[i] = vy;
					
	}
}
