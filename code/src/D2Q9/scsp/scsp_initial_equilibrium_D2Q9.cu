
# include "scsp_initial_equilibrium_D2Q9.cuh"
# include <stdio.h>

// --------------------------------------------------------
// D2Q9 initialize kernel: 
// --------------------------------------------------------

__global__ void scsp_initial_equilibrium_D2Q9(float* f1,
										      float* r,
										      float* u,
										      float* v,
										      int nVoxels)
{
	// define current voxel:
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {			
		// useful constants: 
		const float rho = r[i];
		const float ux = u[i];
		const float vy = v[i];
		const float w0r = rho*4.0/9.0;
		const float wsr = rho*1.0/9.0;
		const float wdr = rho*1.0/36.0;
		const float omusq = 1.0 - 1.5*(ux*ux + vy*vy);	
		const float tux = 3.0*ux;
		const float tvy = 3.0*vy;			
		// equilibrium populations:
		f1[9*i+0] = w0r*(omusq);		
		float cidot3u = tux;
		f1[9*i+1] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy;
		f1[9*i+2] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -tux;
		f1[9*i+3] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -tvy;
		f1[9*i+4] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));	
		cidot3u = tux+tvy;
		f1[9*i+5] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tvy-tux;
		f1[9*i+6] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = -(tux+tvy);
		f1[9*i+7] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
		cidot3u = tux-tvy;
		f1[9*i+8] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	}
}