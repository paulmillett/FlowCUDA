
# include "scsp_stream_collide_save_D3Q19.cuh"
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

__global__ void scsp_stream_collide_save_D3Q19(float* f1,
                                               float* f2,
										       float* r,
										       float* u,
										       float* v,
										       float* w,
										       int* streamIndex,
										       int* voxelType, 
										       iolet* iolets,
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
		// INLETS/OUTLETS - correct the ft[] values at inlets
		//                  and outlets using Zou-He BC's.
		// --------------------------------------------------
		
		if (vtype > 0) {
			zou_he_BC_D3Q19(vtype,ft,iolets);
		}
		
		// --------------------------------------------------
		// MACROS - calculate the velocity and density.
		// --------------------------------------------------
		
		float rho = ft[0]+ft[1]+ft[2]+ft[3]+ft[4]+ft[5]+ft[6]+ft[7]+ft[8]+ft[9]+ft[10]+ft[11]+
			        ft[12]+ft[13]+ft[14]+ft[15]+ft[16]+ft[17]+ft[18];
		float rhoinv = 1.0/rho;
		float ux = rhoinv*(ft[1] + ft[7] + ft[9]  + ft[13] + ft[15] - (ft[2] + ft[8]  + ft[10] + ft[14] + ft[16]));
		float vy = rhoinv*(ft[3] + ft[7] + ft[11] + ft[14] + ft[17] - (ft[4] + ft[8]  + ft[12] + ft[13] + ft[18]));
		float wz = rhoinv*(ft[5] + ft[9] + ft[11] + ft[16] + ft[18] - (ft[6] + ft[10] + ft[12] + ft[15] + ft[17]));
		
		// --------------------------------------------------
		// COLLISION - perform the BGK collision operator.
		// --------------------------------------------------
		
		// useful constants:
		const float tauinv = 2.0/(6.0*nu + 1.0);   // 1/tau
		const float omtauinv = 1.0 - tauinv;       // 1 - 1/tau
		const float tw0r = (1.0/3.0)*rho*tauinv;   // w[0]*rho/tau
		const float twsr = (1.0/18.0)*rho*tauinv;  // w[1-6]*rho/tau
		const float twdr = (1.0/36.0)*rho*tauinv;  // w[7-18]*rho/tau
		const float omusq = 1.0 - 1.5*(ux*ux + vy*vy + wz*wz);
		const float tux = 3.0*ux;
		const float tvy = 3.0*vy;
		const float twz = 3.0*wz;
		// collision calculations:	
		f2[offst+0] = omtauinv*ft[0] + tw0r*(omusq);
		float cidot3u = tux;
		f2[offst+1] = omtauinv*ft[1] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -tux;
		f2[offst+2] = omtauinv*ft[2] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy;
		f2[offst+3] = omtauinv*ft[3] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -tvy;
		f2[offst+4] = omtauinv*ft[4] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = twz;
		f2[offst+5] = omtauinv*ft[5] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -twz;
		f2[offst+6] = omtauinv*ft[6] + twsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tux+tvy;
		f2[offst+7] = omtauinv*ft[7] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -(tux+tvy);
		f2[offst+8] = omtauinv*ft[8] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tux+twz;
		f2[offst+9] = omtauinv*ft[9] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -(tux+twz);
		f2[offst+10] = omtauinv*ft[10] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy+twz;
		f2[offst+11] = omtauinv*ft[11] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -(tvy+twz);
		f2[offst+12] = omtauinv*ft[12] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tux-tvy;
		f2[offst+13] = omtauinv*ft[13] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy-tux;
		f2[offst+14] = omtauinv*ft[14] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tux-twz;
		f2[offst+15] = omtauinv*ft[15] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = twz-tux;
		f2[offst+16] = omtauinv*ft[16] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy-twz;
		f2[offst+17] = omtauinv*ft[17] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = twz-tvy;
		f2[offst+18] = omtauinv*ft[18] + twdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	
		// --------------------------------------------------		
		// SAVE - write macros to arrays only
		//        when needed.
		// --------------------------------------------------
		
		if (save) {
			r[i] = rho;
			u[i] = ux;
			v[i] = vy;
			w[i] = wz;
		}
					
	}
}
