
# include "scsp_stream_collide_save_IBforcing_D3Q19.cuh"
# include "../iolets/zou_he_BC_D3Q19.cuh"
# include "../iolets/boundary_condition_iolet.cuh"
# include <stdio.h>



// --------------------------------------------------------
// D3Q19 update kernel.
// This algorithm is based on the optimized "stream-collide-
// save" algorithm recommended by T. Kruger in the 
// textbook: "The Lattice Boltzmann Method: Principles
// and Practice".
// --------------------------------------------------------

__global__ void scsp_stream_collide_save_IBforcing_D3Q19(float* f1,
                                                         float* f2,
										                 float* r,
										                 float* u,
										                 float* v,
										                 float* w,
													     float* uIBvox,
													     float* vIBvox,
													     float* wIBvox,
														 float* weights,
										                 int* streamIndex,
										                 int* voxelType, 
										                 iolet* iolets,
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
		int offst = 19*i;	
		float ft[19];
		
		// --------------------------------------------------		
		// STREAMING - load populations from adjacent voxels,
		//             note	that streamIndex[] accounts for
		//             halfway bounceback conditions.
		// --------------------------------------------------
				
		ft[0]  = f1[streamIndex[offst+0]];                   
		ft[1]  = f1[streamIndex[offst+1]]; 
		ft[2]  = f1[streamIndex[offst+2]];  
		ft[3]  = f1[streamIndex[offst+3]];  
		ft[4]  = f1[streamIndex[offst+4]];  
		ft[5]  = f1[streamIndex[offst+5]]; 
		ft[6]  = f1[streamIndex[offst+6]];  
		ft[7]  = f1[streamIndex[offst+7]];  
		ft[8]  = f1[streamIndex[offst+8]]; 
		ft[9]  = f1[streamIndex[offst+9]]; 
		ft[10] = f1[streamIndex[offst+10]]; 	
		ft[11] = f1[streamIndex[offst+11]]; 	
		ft[12] = f1[streamIndex[offst+12]]; 	
		ft[13] = f1[streamIndex[offst+13]]; 	
		ft[14] = f1[streamIndex[offst+14]]; 	
		ft[15] = f1[streamIndex[offst+15]]; 	
		ft[16] = f1[streamIndex[offst+16]]; 	
		ft[17] = f1[streamIndex[offst+17]]; 	
		ft[18] = f1[streamIndex[offst+18]]; 	
		
		// --------------------------------------------------
		// COLLISION - this step is done only for fluid voxels;
		//             other treatments are performed for inlet/
		//             outlet voxels.
		// --------------------------------------------------
		
		float rho = ft[0]+ft[1]+ft[2]+ft[3]+ft[4]+ft[5]+ft[6]+ft[7]+ft[8]+ft[9]+ft[10]+ft[11]+
			        ft[12]+ft[13]+ft[14]+ft[15]+ft[16]+ft[17]+ft[18];
		float rhoinv = 1.0/rho;
		float ux = rhoinv*(ft[1] + ft[7] + ft[9]  + ft[13] + ft[15] - (ft[2] + ft[8]  + ft[10] + ft[14] + ft[16]));
		float vy = rhoinv*(ft[3] + ft[7] + ft[11] + ft[14] + ft[17] - (ft[4] + ft[8]  + ft[12] + ft[13] + ft[18]));
		float wz = rhoinv*(ft[5] + ft[9] + ft[11] + ft[16] + ft[18] - (ft[6] + ft[10] + ft[12] + ft[15] + ft[17]));
		
		// IB force correction:
		float Fx = 0.0;
		float Fy = 0.0;
		float Fz = 0.0;
		if (weights[i] > 0.0){
			float uxIB = uIBvox[i]/weights[i];  // weighted average
			float vyIB = vIBvox[i]/weights[i];  // "              "	
			float wzIB = wIBvox[i]/weights[i];  // "              "		
			Fx = (uxIB - ux)*2.0*rho;
			Fy = (vyIB - vy)*2.0*rho;
			Fz = (wzIB - wz)*2.0*rho;
			ux += 0.5*Fx*rhoinv;
			vy += 0.5*Fy*rhoinv;
			wz += 0.5*Fz*rhoinv;
		}		
		
		
		// Fluid voxel...
		if (vtype == 0) {
			
			// useful constants:
			const float w0 = 1.0/3.0;
			const float ws = 1.0/18.0;
			const float wd = 1.0/36.0;			
			const float omega = 2.0/(6.0*nu + 1.0);   // 1/tau
			const float omomega = 1.0 - omega;        // 1 - 1/tau
			const float omomega2 = 1.0 - 0.5*omega;   // 1 - 1/(2tau)
			const float omusq = 1.0 - 1.5*(ux*ux + vy*vy + wz*wz);
			
			// direction 0
			float evel = 0.0;       // e dot velocity
			float emiu = 0.0-ux;    // e minus u
			float emiv = 0.0-vy;    // e minus v
			float emiw = 0.0-wz;    // e minus w
			float feq = w0*rho*omusq;
			float frc = w0*(Fx*(3.0*emiu) + Fy*(3.0*emiv) + Fz*(3.0*emiw));
			f2[offst+0] = omomega*ft[0] + omega*feq + omomega2*frc;
			
			// direction 1
			evel = ux;
			emiu = 1.0-ux;
			emiv = 0.0-vy;
			emiw = 0.0-wz;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx*(3.0*emiu + 9.0*evel) + Fy*(3.0*emiv) + Fz*(3.0*emiw));
			f2[offst+1] = omomega*ft[1] + omega*feq + omomega2*frc;
			
			// direction 2
			evel = -ux;
			emiu = -1.0-ux;
			emiv = 0.0-vy;
			emiw = 0.0-wz;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx*(3.0*emiu - 9.0*evel) + Fy*(3.0*emiv) + Fz*(3.0*emiw));
			f2[offst+2] = omomega*ft[2] + omega*feq + omomega2*frc;
			
			// direction 3
			evel = vy;
			emiu = 0.0-ux;
			emiv = 1.0-vy;
			emiw = 0.0-wz;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx*(3.0*emiu) + Fy*(3.0*emiv + 9.0*evel) + Fz*(3.0*emiw));
			f2[offst+3] = omomega*ft[3] + omega*feq + omomega2*frc;
			
			// direction 4
			evel = -vy;
			emiu = 0.0-ux;
			emiv = -1.0-vy;
			emiw = 0.0-wz;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx*(3.0*emiu) + Fy*(3.0*emiv - 9.0*evel) + Fz*(3.0*emiw));
			f2[offst+4] = omomega*ft[4] + omega*feq + omomega2*frc;
			
			// direction 5
			evel = wz;
			emiu = 0.0-ux;
			emiv = 0.0-vy;
			emiw = 1.0-wz;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx*(3.0*emiu) + Fy*(3.0*emiv) + Fz*(3.0*emiw + 9.0*evel));
			f2[offst+5] = omomega*ft[5] + omega*feq + omomega2*frc;
			
			// direction 6
			evel = -wz;
			emiu = 0.0-ux;
			emiv = 0.0-vy;
			emiw = -1.0-wz;
			feq = ws*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = ws*(Fx*(3.0*emiu) + Fy*(3.0*emiv) + Fz*(3.0*emiw - 9.0*evel));
			f2[offst+6] = omomega*ft[6] + omega*feq + omomega2*frc;
			
			// direction 7
			evel = ux+vy;
			emiu = 1.0-ux;
			emiv = 1.0-vy;
			emiw = 0.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx*(3.0*emiu + 9.0*evel) + Fy*(3.0*emiv + 9.0*evel) + Fz*(3.0*emiw));
			f2[offst+7] = omomega*ft[7] + omega*feq + omomega2*frc;
			
			// direction 8
			evel = -ux-vy;
			emiu = -1.0-ux;
			emiv = -1.0-vy;
			emiw = 0.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx*(3.0*emiu - 9.0*evel) + Fy*(3.0*emiv - 9.0*evel) + Fz*(3.0*emiw));
			f2[offst+8] = omomega*ft[8] + omega*feq + omomega2*frc;
			
			// direction 9
			evel = ux+wz;
			emiu = 1.0-ux;
			emiv = 0.0-vy;
			emiw = 1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx*(3.0*emiu + 9.0*evel) + Fy*(3.0*emiv) + Fz*(3.0*emiw + 9.0*evel));
			f2[offst+9] = omomega*ft[9] + omega*feq + omomega2*frc;
			
			// direction 10
			evel = -ux-wz;
			emiu = -1.0-ux;
			emiv = 0.0-vy;
			emiw = -1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx*(3.0*emiu - 9.0*evel) + Fy*(3.0*emiv) + Fz*(3.0*emiw - 9.0*evel));
			f2[offst+10] = omomega*ft[10] + omega*feq + omomega2*frc;
			
			// direction 11
			evel = vy+wz;
			emiu = 0.0-ux;
			emiv = 1.0-vy;
			emiw = 1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx*(3.0*emiu) + Fy*(3.0*emiv + 9.0*evel) + Fz*(3.0*emiw + 9.0*evel));
			f2[offst+11] = omomega*ft[11] + omega*feq + omomega2*frc;
			
			// direction 12
			evel = -vy-wz;
			emiu = 0.0-ux;
			emiv = -1.0-vy;
			emiw = -1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx*(3.0*emiu) + Fy*(3.0*emiv - 9.0*evel) + Fz*(3.0*emiw - 9.0*evel));
			f2[offst+12] = omomega*ft[12] + omega*feq + omomega2*frc;
			
			// direction 13
			evel = ux-vy;
			emiu = 1.0-ux;
			emiv = -1.0-vy;
			emiw = 0.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx*(3.0*emiu + 9.0*evel) + Fy*(3.0*emiv - 9.0*evel) + Fz*(3.0*emiw));
			f2[offst+13] = omomega*ft[13] + omega*feq + omomega2*frc;
			
			// direction 14
			evel = -ux+vy;
			emiu = -1.0-ux;
			emiv = 1.0-vy;
			emiw = 0.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx*(3.0*emiu - 9.0*evel) + Fy*(3.0*emiv + 9.0*evel) + Fz*(3.0*emiw));
			f2[offst+14] = omomega*ft[14] + omega*feq + omomega2*frc;
			
			// direction 15
			evel = ux-wz;
			emiu = 1.0-ux;
			emiv = 0.0-vy;
			emiw = -1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx*(3.0*emiu + 9.0*evel) + Fy*(3.0*emiv) + Fz*(3.0*emiw - 9.0*evel));
			f2[offst+15] = omomega*ft[15] + omega*feq + omomega2*frc;
			
			// direction 16
			evel = -ux+wz;
			emiu = -1.0-ux;
			emiv = 0.0-vy;
			emiw = 1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx*(3.0*emiu - 9.0*evel) + Fy*(3.0*emiv) + Fz*(3.0*emiw + 9.0*evel));
			f2[offst+16] = omomega*ft[16] + omega*feq + omomega2*frc;
			
			// direction 17
			evel = vy-wz;
			emiu = 0.0-ux;
			emiv = 1.0-vy;
			emiw = -1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx*(3.0*emiu) + Fy*(3.0*emiv + 9.0*evel) + Fz*(3.0*emiw - 9.0*evel));
			f2[offst+17] = omomega*ft[17] + omega*feq + omomega2*frc;
			
			// direction 18
			evel = -vy+wz;
			emiu = 0.0-ux;
			emiv = -1.0-vy;
			emiw = 1.0-wz;
			feq = wd*rho*(omusq + 3.0*evel + 4.5*evel*evel);
			frc = wd*(Fx*(3.0*emiu) + Fy*(3.0*emiv - 9.0*evel) + Fz*(3.0*emiw + 9.0*evel));
			f2[offst+18] = omomega*ft[18] + omega*feq + omomega2*frc;
						
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
				wz = iolets[ioi].wBC;
				rho = (ft[0]+ft[3]+ft[4]+ft[5]+ft[6]+ft[11]+ft[12]+ft[17]+ft[18]+
				       2.0*(ft[1]+ft[7]+ft[9]+ft[13]+ft[15]))/(1.0 + ux);
				zou_he_velo_east_D3Q19(offst,f2,ft,ux,vy,wz,rho);
			}
			
			// Zou-He velocity boundary (West)...
			else if (iotype == 2) {
				ux = iolets[ioi].uBC;
				vy = iolets[ioi].vBC;
				wz = iolets[ioi].wBC;
				rho = (ft[0]+ft[3]+ft[4]+ft[5]+ft[6]+ft[11]+ft[12]+ft[17]+ft[18]+
				       2.0*(ft[2]+ft[8]+ft[10]+ft[14]+ft[16]))/(1.0 - ux);		
				zou_he_velo_west_D3Q19(offst,f2,ft,ux,vy,wz,rho);
			}
			
			// Zou-He velocity boundary (North)...
			else if (iotype == 3) {
				ux = iolets[ioi].uBC;
				vy = iolets[ioi].vBC;
				wz = iolets[ioi].wBC;
				rho = (ft[0]+ft[1]+ft[2]+ft[5]+ft[6]+ft[9]+ft[10]+ft[15]+ft[16]+
				       2.0*(ft[3]+ft[7]+ft[11]+ft[14]+ft[17]))/(1.0 + vy);
				zou_he_velo_north_D3Q19(offst,f2,ft,ux,vy,wz,rho);
			}	
			
			// Zou-He velocity boundary (South)...
			else if (iotype == 4) {
				ux = iolets[ioi].uBC;
				vy = iolets[ioi].vBC;
				wz = iolets[ioi].wBC;
				rho = (ft[0]+ft[1]+ft[2]+ft[5]+ft[6]+ft[9]+ft[10]+ft[15]+ft[16]+
				       2.0*(ft[4]+ft[8]+ft[12]+ft[13]+ft[18]))/(1.0 - vy);	
				zou_he_velo_south_D3Q19(offst,f2,ft,ux,vy,wz,rho);
			}	
			
			// Zou-He velocity boundary (Top)...
			else if (iotype == 5) {
				ux = iolets[ioi].uBC;
				vy = iolets[ioi].vBC;
				wz = iolets[ioi].wBC;
				rho = (ft[0]+ft[1]+ft[2]+ft[3]+ft[4]+ft[7]+ft[8]+ft[13]+ft[14]+
				       2.0*(ft[5]+ft[9]+ft[11]+ft[16]+ft[18]))/(1.0 + wz);
				zou_he_velo_top_D3Q19(offst,f2,ft,ux,vy,wz,rho);
			}	
			
			// Zou-He velocity boundary (Bottom)...
			else if (iotype == 6) {
				ux = iolets[ioi].uBC;
				vy = iolets[ioi].vBC;
				wz = iolets[ioi].wBC;
				rho = (ft[0]+ft[1]+ft[2]+ft[3]+ft[4]+ft[7]+ft[8]+ft[13]+ft[14]+
				       2.0*(ft[6]+ft[10]+ft[12]+ft[15]+ft[17]))/(1.0 - wz);	
				zou_he_velo_bottom_D3Q19(offst,f2,ft,ux,vy,wz,rho);
			}	
			
			// Zou-He pressure boundary (East)...
			else if (iotype == 11) {
				vy = iolets[ioi].vBC;
				wz = iolets[ioi].wBC;
				rho = iolets[ioi].rBC;
				ux = -1.0 + (ft[0]+ft[3]+ft[4]+ft[5]+ft[6]+ft[11]+ft[12]+ft[17]+ft[18]+
				             2.0*(ft[1]+ft[7]+ft[9]+ft[13]+ft[15]))/rho;				
				zou_he_pres_east_D3Q19(offst,f2,ft,ux,vy,wz,rho);
			}	
			
			// Zou-He pressure boundary (West)...
			else if (iotype == 12) {			
				vy = iolets[ioi].vBC;
				wz = iolets[ioi].wBC;
				rho = iolets[ioi].rBC;
				ux = 1.0 - (ft[0]+ft[3]+ft[4]+ft[5]+ft[6]+ft[11]+ft[12]+ft[17]+ft[18]+
				            2.0*(ft[2]+ft[8]+ft[10]+ft[14]+ft[16]))/rho;	
				zou_he_pres_west_D3Q19(offst,f2,ft,ux,vy,wz,rho);
			}	
			
			// Zou-He pressure boundary (North)...
			else if (iotype == 13) {
				ux = iolets[ioi].uBC;
				wz = iolets[ioi].wBC;
				rho = iolets[ioi].rBC;
				vy = -1.0 + (ft[0]+ft[1]+ft[2]+ft[5]+ft[6]+ft[9]+ft[10]+ft[15]+ft[16]+
						     2.0*(ft[3]+ft[7]+ft[11]+ft[14]+ft[17]))/rho;
				zou_he_pres_north_D3Q19(offst,f2,ft,ux,vy,wz,rho);
			}	
			
			// Zou-He pressure boundary (South)...
			else if (iotype == 14) {
				ux = iolets[ioi].uBC;
				wz = iolets[ioi].wBC;
				rho = iolets[ioi].rBC;
				vy = 1.0 - (ft[0]+ft[1]+ft[2]+ft[5]+ft[6]+ft[9]+ft[10]+ft[15]+ft[16]+
						    2.0*(ft[4]+ft[8]+ft[12]+ft[13]+ft[18]))/rho;		
				zou_he_pres_south_D3Q19(offst,f2,ft,ux,vy,wz,rho);
			}	
			
			// Zou-He pressure boundary (Top)...
			else if (iotype == 15) {
				ux = iolets[ioi].uBC;
				vy = iolets[ioi].vBC;
				rho = iolets[ioi].rBC;
				wz = -1.0 + (ft[0]+ft[1]+ft[2]+ft[3]+ft[4]+ft[7]+ft[8]+ft[13]+ft[14]+
						     2.0*(ft[5]+ft[9]+ft[11]+ft[16]+ft[18]))/rho;	
				zou_he_pres_top_D3Q19(offst,f2,ft,ux,vy,wz,rho);
			}	
			
			// Zou-He pressure boundary (Bottom)...
			else if (iotype == 16) {
				ux = iolets[ioi].uBC;
				vy = iolets[ioi].vBC;
				rho = iolets[ioi].rBC;
				wz = 1.0 - (ft[0]+ft[1]+ft[2]+ft[3]+ft[4]+ft[7]+ft[8]+ft[13]+ft[14]+
						    2.0*(ft[6]+ft[10]+ft[12]+ft[15]+ft[17]))/rho;
				zou_he_pres_bottom_D3Q19(offst,f2,ft,ux,vy,wz,rho);
			}	
			
		}
	
		// --------------------------------------------------		
		// SAVE - write macros to arrays 
		// --------------------------------------------------
		
		r[i] = rho;
		u[i] = ux;
		v[i] = vy;
		w[i] = wz;
					
	}
}
