
# include "scsp_initial_equilibrium_D3Q19.cuh"
# include <stdio.h>



// --------------------------------------------------------
// D3Q19 initialize kernel:
// --------------------------------------------------------

__global__ void scsp_initial_equilibrium_D3Q19(float* f1,
										       float* r,
										       float* u,
										       float* v,
										       float* w,
										       int nVoxels)
{
	
	// -----------------------------------------------
	// define current voxel:
	// -----------------------------------------------
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	// -----------------------------------------------
	// assign equilibrium populations:
	// -----------------------------------------------
	
	if (i < nVoxels) {			
		// useful constants: 
		const int offst = 19*i;
		const float ux = u[i];
		const float vy = v[i];
		const float wz = w[i];
		const float rho = r[i];
		const float w0r = rho*1.0/3.0;
		const float wsr = rho*1.0/18.0;
		const float wdr = rho*1.0/36.0;
		const float omusq = 1.0 - 1.5*(ux*ux + vy*vy + wz*wz);	
		const float tux = 3.0*ux;
		const float tvy = 3.0*vy;
		const float twz = 3.0*wz;
		// equilibrium populations:
		f1[offst+0] = w0r*(omusq);				
		float cidot3u = tux;
		f1[offst+1] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -tux;
		f1[offst+2] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy;
		f1[offst+3] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -tvy;
		f1[offst+4] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));	
		cidot3u = twz;
		f1[offst+5] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -twz;
		f1[offst+6] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));		
		cidot3u = tux+tvy;
		f1[offst+7] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -(tux+tvy);
		f1[offst+8] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tux+twz;
		f1[offst+9] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -(tux+twz);
		f1[offst+10] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy+twz;
		f1[offst+11] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -(tvy+twz);
		f1[offst+12] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tux-tvy;
		f1[offst+13] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy-tux;
		f1[offst+14] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tux-twz;
		f1[offst+15] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = twz-tux;
		f1[offst+16] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy-twz;
		f1[offst+17] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = twz-tvy;
		f1[offst+18] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	}	
	
}