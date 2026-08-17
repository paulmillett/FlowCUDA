# include "kernels_scsp_psm_D3Q19.cuh"
# include <stdio.h>




// --------------------------------------------------------
// D3Q19 initialize kernel:
// --------------------------------------------------------

__global__ void scsp_psm_initial_equilibrium_D3Q19(
	float* f1,
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
		equilibrium_populations_psm_D3Q19(f1,r[i],u[i],v[i],w[i],offst);
	}		
}



// --------------------------------------------------------
// D3Q19 equilibrium populations:
// --------------------------------------------------------

__device__ void equilibrium_populations_psm_D3Q19(
	float* f1,
	const float r,
	const float u,
	const float v,
	const float w,
	const int offst)
{
	// constants:
	const float w0r = r*1.0/3.0;
	const float wsr = r*1.0/18.0;
	const float wdr = r*1.0/36.0;
	const float omusq = 1.0 - 1.5*(u*u + v*v + w*w);	
	const float tux = 3.0*u;
	const float tvy = 3.0*v;
	const float twz = 3.0*w;
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



// --------------------------------------------------------
// D3Q19 kernel to set shear velocities at the y=0 and
// y=Ny-1 boundaries.  The shear direction is the x-dir.
// NOTE: This should be called AFTER the collide-streaming
//       step.  It should be the last calculation for the 
//       fluid update.  
// --------------------------------------------------------

__global__ void scsp_psm_set_boundary_shear_velocity_D3Q19(
	float uBot,
    float uTop,
	float* f1,													   
	float* u,
	float* v,
	float* w,
	float* r,
	int Nx,
	int Ny,
	int Nz,
	int nVoxels)
{
    // -----------------------------------------------
    // define voxel:
    // -----------------------------------------------
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (i < nVoxels) {		
		int zi = i/(Nx*Ny);    // z-index assuming data is ordered first x, then y, then z
		if (zi == 0) {
			int offst = 19*i;
			u[i] = uBot;
			v[i] = 0.0;
			w[i] = 0.0;
			equilibrium_populations_psm_D3Q19(f1,r[i],u[i],v[i],w[i],offst);
		} 
		if (zi == Nz-1) {
			int offst = 19*i;
			u[i] = uTop;
			v[i] = 0.0;
			w[i] = 0.0;
			equilibrium_populations_psm_D3Q19(f1,r[i],u[i],v[i],w[i],offst);
		}		
	}		
}



// --------------------------------------------------------
// D3Q19 kernel to set shear velocities at the y=0 and
// y=Ny-1 boundaries.  The shear direction is the x-dir.
// NOTE: This should be called AFTER the collide-streaming
//       step.  It should be the last calculation for the 
//       fluid update.  
// --------------------------------------------------------

__global__ void scsp_psm_map_particles_to_lattice_D3Q19(
	float* eps,
	int* pID,
	sphere* spheres,
	int nSpheres,
	int Nx,
	int Ny,
	int Nz,
	int nVoxels)
{
    // -----------------------------------------------
    // define voxel:
    // -----------------------------------------------
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;	
	
	if (i < nVoxels) {		
		
		// --------------------------------------------------		
		// lattice site indices:
		// --------------------------------------------------
		
        int x = i % Nx;
        int y = (i / Nx) % Ny;
        int z = i / (Nx * Ny);
		
		// --------------------------------------------------		
		// set pID[] and eps[] to default values:
		// --------------------------------------------------
		
		eps[i] = 0.0;
		pID[i] = -1;
		const float delta = 0.5;   // interface width
		
		// --------------------------------------------------		
		// loop over particles:
		// --------------------------------------------------
		
		for (int s=0; s<nSpheres; s++) {
			
			// distance between lattice site and sphere coords
            sphere p = spheres[s];
            float rx = (float)x - p.r.x;
            float ry = (float)y - p.r.y;
            float rz = (float)z - p.r.z;
			// need to correct for PBC's
			float dist = sqrt(rx*rx + ry*ry + rz*rz);
			
			// if lattice site is inside sphere, mark it:			
			if (dist <= p.rad - delta) {
				eps[i] = 1.0;
				pID[i] = s;
			}
			else if (dist <= p.rad + delta) {
				eps[i] = 0.5*(1.0 - (dist - p.rad)/delta);
				pID[i] = s;
			}
		}
	}		
}



// --------------------------------------------------------
// D3Q19 update kernel tailored for the partially-saturated
// method for particle suspensions, see: Noble & Torczynski,
// A Lattice-Boltzmann Method for Partially Saturated
// Computational Cells,
// International Journal of Modern Physics C, 09(08):1189–1201, 1998.
//
// This algorithm is based on the optimized "stream-collide-
// save" algorithm recommended by T. Kruger in the 
// textbook: "The Lattice Boltzmann Method: Principles
// and Practice".
// --------------------------------------------------------

__global__ void scsp_psm_stream_collide_save_D3Q19(
	float* f1,
    float* f2,
    float* r,
    float* u,
    float* v,
    float* w,
    float* eps,
    int* pID,
    int* streamIndex,
    sphere* spheres,
    float nu,
    int Nx,
    int Ny,
    int Nz,
    int nVoxels)
{
    
	// -----------------------------------------------
    // define voxel:
    // -----------------------------------------------
    
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < nVoxels) {
        
        // --------------------------------------------------        
        // voxel-specific parameters:
        // --------------------------------------------------
		
        const int offst = 19 * i;    
        const float epsilon = eps[i];
		const float tauinv = 2.0/(6.0*nu + 1.0);   // 1/tau
        
        // --------------------------------------------------        
        // STREAMING - load populations from adjacent voxels
        // --------------------------------------------------
		
		float ft[19];
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
        // MACROS - calculate the velocity and density
        // --------------------------------------------------
		
        float rho = ft[0]+ft[1]+ft[2]+ft[3]+ft[4]+ft[5]+ft[6]+ft[7]+ft[8]+ft[9]+ft[10]+ft[11]+
                    ft[12]+ft[13]+ft[14]+ft[15]+ft[16]+ft[17]+ft[18];
        float rhoinv = 1.0f / rho;
        float ux = rhoinv*(ft[1] + ft[7] + ft[9]  + ft[13] + ft[15] - (ft[2] + ft[8]  + ft[10] + ft[14] + ft[16]));
        float vy = rhoinv*(ft[3] + ft[7] + ft[11] + ft[14] + ft[17] - (ft[4] + ft[8]  + ft[12] + ft[13] + ft[18]));
        float wz = rhoinv*(ft[5] + ft[9] + ft[11] + ft[16] + ft[18] - (ft[6] + ft[10] + ft[12] + ft[15] + ft[17]));
        
        // --------------------------------------------------
        // COLLISION - Standard equilibrium populations
        // --------------------------------------------------
		
		float feq[19];
		equilibrium_populations_psm_D3Q19(feq,rho,ux,vy,wz);
		
        // --------------------------------------------------
        // PSM MODIFICATION (Noble & Torczynski, 1998)
		// for fully or partially solid cells:
        // --------------------------------------------------
		
        if (epsilon > 0.0f) {
			
			// Calculate weighting factor B(eps, tau) where (tau - 0.5) = 3*nu
            float tau_minus_half = 3.0f * nu;
            float B = (epsilon * tau_minus_half) / ((1.0f - epsilon) + tau_minus_half);

            // Unravel 1D index into 3D grid spatial coordinates (x, y, z)
            int x = i % Nx;
            int y = (i / Nx) % Ny;
            int z = i / (Nx * Ny);

            // Compute local solid velocity: u_s = V_p + omega_p x (x - X_p)
            int id = pID[i];
            sphere p = spheres[id];
            float rx = (float)x - p.r.x;
            float ry = (float)y - p.r.y;
            float rz = (float)z - p.r.z;
            float usx = p.v.x + (p.w.y * rz - p.w.z * ry);
            float usy = p.v.y + (p.w.z * rx - p.w.x * rz);
            float usz = p.v.z + (p.w.x * ry - p.w.y * rx);
			
			// Equilibrium populations based on solid velocity
			float feqS[19];
			equilibrium_populations_psm_D3Q19(feqS,rho,usx,usy,usz);
			
            // Direction opposite lookup map for D3Q19
            const int opp[19] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};

            // Blended collision operator
            for (int q=0; q<19; q++) {
				int qopp = opp[q];
                float omegaF = ft[q] - feq[q];
				//float omegaS = ft[qopp] - feq[q] + feqS[q] - feqS[qopp];  // standard solid collision
				float omegaS = ft[qopp] - ft[q] + feqS[q] - feq[qopp];      // non-equil BB solid collision
				f2[offst + q] = ft[q] + (1.0 - B)*tauinv*omegaF + B*omegaS;
            }
        } 
		
        // --------------------------------------------------
        // Standard fluid collision:
        // --------------------------------------------------
		
		else {
            for (int q=0; q<19; q++) {
				f2[offst + q] = ft[q] - tauinv*(ft[q] - feq[q]);
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



// --------------------------------------------------------
// D3Q19 equilibrium populations:
// NOTE: the f[19] array here is a local array
// --------------------------------------------------------

__device__ void equilibrium_populations_psm_D3Q19(
	float* f,
	const float r,
	const float u,
	const float v,
	const float w)
{
	// constants:
	const float w0r = r*1.0/3.0;
	const float wsr = r*1.0/18.0;
	const float wdr = r*1.0/36.0;
	const float omusq = 1.0 - 1.5*(u*u + v*v + w*w);	
	const float tux = 3.0*u;
	const float tvy = 3.0*v;
	const float twz = 3.0*w;
	// equilibrium populations:
	f[0] = w0r*(omusq);				
	float cidot3u = tux;
	f[1] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = -tux;
	f[2] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tvy;
	f[3] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = -tvy;
	f[4] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));	
	cidot3u = twz;
	f[5] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = -twz;
	f[6] = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));		
	cidot3u = tux+tvy;
	f[7] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = -(tux+tvy);
	f[8] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tux+twz;
	f[9] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = -(tux+twz);
	f[10] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tvy+twz;
	f[11] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = -(tvy+twz);
	f[12] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tux-tvy;
	f[13] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tvy-tux;
	f[14] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tux-twz;
	f[15] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = twz-tux;
	f[16] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = tvy-twz;
	f[17] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	cidot3u = twz-tvy;
	f[18] = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
}











// ******************************************************************************
//	Rigid sphere dynamics...
// ******************************************************************************











// --------------------------------------------------------
// kernel to zero sphere forces & torques:
// --------------------------------------------------------

__global__ void zero_sphere_forces_torques(
	sphere* spheres,	
	int nSpheres)
{
	// define sphere:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nSpheres) {
		spheres[i].f = make_float3(0.0f,0.0f,0.0f);
		spheres[i].t = make_float3(0.0f,0.0f,0.0f);
	}
}



// --------------------------------------------------------
// kernel to update sphere position and orientation:
// --------------------------------------------------------

__global__ void update_sphere_position_orientation(
	sphere* spheres,
	float dt,
	int nSpheres)
{
	// define sphere:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nSpheres) {
		
	}
}



/*

// Kernel 1: Calculate pairwise particle interactions & wall forces
__global__ void computeForcesKernel(
	Particle* particles,
	Vec3* contactHistory, 
    Vec3* wallContactHistory,
	int numParticles,
	DEMParams params,
	double dt)
{
    // define sphere:
	int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= numParticles) return;

    Particle p_i = particles[i];
    Vec3 total_force = params.gravity * p_i.mass;
    Vec3 total_torque = {0.0, 0.0, 0.0};

    // 1. Particle-Particle Contacts
    for (int j = 0; j < numParticles; ++j) {
        int pair_idx = i * numParticles + j;

        if (i == j) {
            contactHistory[pair_idx] = Vec3{0, 0, 0};
            continue;
        }

        Particle p_j = particles[j];
        Vec3 r12 = p_j.pos - p_i.pos;
        double dist = r12.norm();
        double overlap = p_i.radius + p_j.radius - dist;

        if (overlap > 0.0 && dist > 1e-12) {
            Vec3 normal = r12 / dist; // Unit vector from i to j

            Vec3 c1 = normal * p_i.radius;
            Vec3 c2 = normal * (-p_j.radius);

            Vec3 v1_c = p_i.vel + p_i.omega.cross(c1);
            Vec3 v2_c = p_j.vel + p_j.omega.cross(c2);
            Vec3 v_rel = v1_c - v2_c; // Relative velocity of i w.r.t j

            double v_n = v_rel.dot(normal);
            Vec3 v_t = v_rel - normal * v_n;

            // Accumulated tangential displacement from global memory
            Vec3 delta_t = contactHistory[pair_idx];
            delta_t = delta_t + v_t * dt;
            delta_t = delta_t - normal * delta_t.dot(normal); // Project onto tangent plane

            // Normal Force
            double F_n_mag = params.kn * overlap - params.gn * v_n;
            if (F_n_mag < 0.0) F_n_mag = 0.0;
            Vec3 F_n = normal * (-F_n_mag);

            // Tangential Force (Trial)
            Vec3 F_t_trial = delta_t * (-params.kt) - v_t * params.gt;

            // Coulomb Friction
            Vec3 F_t;
            double max_friction = params.mu * F_n_mag;
            double F_t_norm = F_t_trial.norm();

            if (F_t_norm > max_friction && F_t_norm > 1e-12) {
                F_t = F_t_trial.normalized() * max_friction;
                if (params.kt > 0.0) {
                    delta_t = (F_t * (-1.0) - v_t * params.gt) / params.kt;
                }
            } else {
                F_t = F_t_trial;
            }

            contactHistory[pair_idx] = delta_t;

            Vec3 F_total = F_n + F_t;
            total_force += F_total;
            total_torque += c1.cross(F_total);
        } else {
            contactHistory[pair_idx] = Vec3{0, 0, 0}; // Reset when separated
        }
    }

    // 2. Ground Wall Collision (z = 0)
    double overlap_wall = p_i.radius - p_i.pos.z;
    if (overlap_wall > 0.0) {
        Vec3 normal = {0.0, 0.0, 1.0};
        Vec3 c = {0.0, 0.0, -p_i.radius};

        Vec3 v_c = p_i.vel + p_i.omega.cross(c);
        double v_n = v_c.dot(normal);
        Vec3 v_t = v_c - normal * v_n;

        Vec3 delta_t = wallContactHistory[i];
        delta_t = delta_t + v_t * dt;
        delta_t = delta_t - normal * delta_t.dot(normal);

        double F_n_mag = params.kn * overlap_wall - params.gn * v_n;
        if (F_n_mag < 0.0) F_n_mag = 0.0;
        Vec3 F_n = normal * F_n_mag;

        Vec3 F_t_trial = delta_t * (-params.kt) - v_t * params.gt;
        Vec3 F_t;
        double max_friction = params.mu * F_n_mag;
        double F_t_norm = F_t_trial.norm();

        if (F_t_norm > max_friction && F_t_norm > 1e-12) {
            F_t = F_t_trial.normalized() * max_friction;
            if (params.kt > 0.0) {
                delta_t = (F_t * (-1.0) - v_t * params.gt) / params.kt;
            }
        } else {
            F_t = F_t_trial;
        }

        wallContactHistory[i] = delta_t;

        Vec3 F_total = F_n + F_t;
        total_force += F_total;
        total_torque += c.cross(F_total);
    } else {
        wallContactHistory[i] = Vec3{0, 0, 0};
    }

    particles[i].force = total_force;
    particles[i].torque = total_torque;
}

*/


