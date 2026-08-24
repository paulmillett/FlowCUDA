# include "kernels_discs_ibm3D.cuh"
# include <stdio.h>






// --------------------------------------------------------
// 1. Initialization Kernel: Sets up the PRNG state for each thread
// --------------------------------------------------------

__global__ void init_rand_kernel_discs_IBM3D(
	curandState *state,
	unsigned long seed,
	int nDiscs)
{
    // Each thread gets same seed, a unique sequence number (i), and no offset
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < nDiscs) curand_init(seed,i,0,&state[i]);
}



// --------------------------------------------------------
// IBM3D kernel to zero disc forces, torques, moment of
// inertia:
// --------------------------------------------------------

__global__ void zero_disc_forces_torques_moments_IBM3D(
	disc* discs,	
	int nDiscs)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nDiscs) {
		discs[i].f = make_float3(0.0f,0.0f,0.0f);
		discs[i].t = make_float3(0.0f,0.0f,0.0f);
		discs[i].uf = make_float3(0.0f,0.0f,0.0f);
		discs[i].gradu.xx = 0.0;
		discs[i].gradu.xy = 0.0;
		discs[i].gradu.xz = 0.0;
		discs[i].gradu.yx = 0.0;
		discs[i].gradu.yy = 0.0;
		discs[i].gradu.yz = 0.0;
		discs[i].gradu.zx = 0.0;
		discs[i].gradu.zy = 0.0;
		discs[i].gradu.zz = 0.0;
	}
}



// --------------------------------------------------------
// IBM3D kernel to zero bead forces:
// --------------------------------------------------------

__global__ void zero_bead_forces_IBM3D(
	beaddisc* beads,	
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		beads[i].f = make_float3(0.0f,0.0f,0.0f);
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate disc orientation:
// --------------------------------------------------------

__global__ void set_disc_position_orientation_IBM3D(
	beaddisc* beads,
	disc* discs,	
	int nDiscs)
{
	// define disc:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nDiscs) {
		int BC = discs[i].centerBead;
		discs[i].r = beads[BC].r;
	}
}



// --------------------------------------------------------
// IBM3D enforce a maximum bead force:
// --------------------------------------------------------

__global__ void enforce_max_bead_force_IBM3D(
	beaddisc* beads,
	float fmax,
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		float fi = length(beads[i].f);
		if (fi > fmax) beads[i].f *= (fmax/fi);
	}
}



// --------------------------------------------------------
// IBM3D enforce a maximum disc force & torque:
// --------------------------------------------------------

__global__ void enforce_max_disc_force_torque_IBM3D(
	disc* discs,
	float fmax,
	float tmax,
	int nDiscs)
{
	// define disc:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nDiscs) {
		float fi = length(discs[i].f);
		float ti = length(discs[i].t);
		if (fi > fmax) discs[i].f *= (fmax/fi);
		if (ti > tmax) discs[i].t *= (tmax/ti);
	}
}



// --------------------------------------------------------
// IBM3D bead update kernel:
// --------------------------------------------------------

__global__ void update_bead_positions_discs_IBM3D(
	beaddisc* beads,
	disc* discs,
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		int discID = beads[i].discID;
		quaternion q = discs[discID].q;
		float3 rrel = beads[i].rrel;		
		beads[i].r = q*rrel + discs[discID].r;
	}
}



// --------------------------------------------------------
// IBM3D bead update kernel:
// --------------------------------------------------------

__global__ void update_bead_velocity_discs_IBM3D(
	beaddisc* beads,
	float3 Box,
	int3 pbcFlag,
	float dt,
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {		
		float3 dr = beads[i].r - beads[i].rm1;
		dr -= roundf(dr/Box)*Box*pbcFlag;  // PBC's	
		beads[i].v = dr/dt;			
	}
}



// --------------------------------------------------------
// IBM3D bead update kernel:
// --------------------------------------------------------

__global__ void add_gravity_force_to_beads_IBM3D(
	beaddisc* beads,
	float Fzgrav,
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {		
		beads[i].f.x -= Fzgrav;			
	}
}



// --------------------------------------------------------
// IBM3D disc update kernel:
// --------------------------------------------------------

__global__ void update_disc_position_orientation_fluid_IBM3D(
	disc* discs,
	float dt,
	int nDiscs)
{
	// define disc:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nDiscs) {
				
		// mobility coefficients
		tensor ppT = dyadic(discs[i].p);
		tensor Imppt = identity() - ppT;
		tensor mobTensor = discs[i].mobPar*ppT + discs[i].mobPer*Imppt;
	
		// disc translation:	
		discs[i].r += dt*(discs[i].uf + mobTensor*discs[i].f);
		
		// disc shape factor for rotation:
		float ar = discs[i].ar;
		//if (ar > 5.0) ar = (1.24*ar)/sqrt(log(ar));  // correction factor for discs vs spheroids: Cox, JFM (1971) 45:625-657 
		float shape = (ar*ar - 1.0)/(ar*ar + 1.0);   // Bretherton constant
		
		// fluid strain rate tensor (E) and vorticity tensor (W):
		tensor E = 0.5*(discs[i].gradu + transpose(discs[i].gradu));
		tensor W = 0.5*(discs[i].gradu - transpose(discs[i].gradu));
		float3 Wvec = make_float3(-W.yz,W.xz,-W.xy);  // vorticity vector
		
		// angular velocity of disc:
		float3 p = discs[i].p;
		float3 omegaDisc = Wvec + shape*(cross(p,E*p));   // WHAT ABOUT TORQUES...!
		
		// update disc quaternion:
		discs[i].q.update(dt,omegaDisc);
		discs[i].q.normalize();
		
		// update disc orientation vector:
		discs[i].p = discs[i].q.orientation_vec();
		discs[i].p = normalize(discs[i].p);
									
	}
}



// --------------------------------------------------------
// IBM3D disc update kernel:
// --------------------------------------------------------

__global__ void update_disc_position_orientation_no_fluid_IBM3D(
	disc* discs,
	float dt,
	int nDiscs)
{
	// define disc:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nDiscs) {
		/*	
		// mobility coefficients
		tensor ppT = dyadic(rods[i].p);
		tensor Imppt = identity() - ppT;
		tensor mobTensor = rods[i].mobPar*ppT + rods[i].mobPer*Imppt;
				
		// rod translation:		
		rods[i].r += dt*(mobTensor*rods[i].f);
		
		// rod rotation:
		rods[i].p += dt*(rods[i].mobRot*cross(rods[i].t,rods[i].p));		
		rods[i].p = normalize(rods[i].p);
		*/			
	}
}



// --------------------------------------------------------
// IBM3D kernel to assign discs in the backfill zone a set
// velocity:
// --------------------------------------------------------

__global__ void assign_velocity_to_backfill_discs_IBM3D(
	disc* discs,
	float ufx,
	int nDiscs)
{
	// define disc:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nDiscs) {
		// if rod is in backfill zone:
		if (discs[i].r.x < 0.0f) {
			discs[i].uf.x = ufx;
			discs[i].uf.y = 0.0;
			discs[i].uf.z = 0.0;
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to move disc back to inlet if too close to
// outlet:
// --------------------------------------------------------

__global__ void move_disc_back_to_inlet_random_IBM3D(
	disc* discs,
	float3 Box,
	float lenI,
	float radI,
	float radO,
	int nDiscs,
	curandState* state)
{
	// define disc:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nDiscs) {
		
		/*
		float Lrod = float(rods[i].nBeads-1)*L0;  //float(nBeadsPerRod-1)*L0;
		float offset = Lrod/2.0 + 1.0;
		
		// check if rod is too close to outlet (x-dir):
		if (rods[i].r.x > Box.x - offset) {
			
			// Copy random state to local registers:
		    curandState localState = state[i];
			
			// find new rod position & orientation in loading zone:
			while (true) {
				
				// get random position in loading zone:
				float ran1 = curand_uniform(&localState);
				float ran2 = curand_uniform(&localState);
				float ran3 = curand_uniform(&localState);						
				float xpo = offset + ran1*(lenI - 2*offset);
				float rad = ran2*(radI);
				float ang = ran3*(2*M_PI);
				rods[i].r.x = xpo;
				rods[i].r.y = rad*cos(ang) + (Box.y-1.0)/2.0;
				rods[i].r.z = rad*sin(ang) + (Box.z-1.0)/2.0;
			
				// get random orientation:
				float ran4 = curand_uniform(&localState);
				float ran5 = curand_uniform(&localState);
				float phi = ran4*(2*M_PI);  // azimuthal angle
				float psi = 2.0*ran5 - 1.0; // random num from -1 to 1
				rods[i].p.x = sqrt(1.0-psi*psi)*cos(phi);
				rods[i].p.y = sqrt(1.0-psi*psi)*sin(phi);
				rods[i].p.z = psi;
						
				// radial position of rod head:
				float3 head = rods[i].r + offset*rods[i].p;		
				float ymid = (Box.y-1.0)/2.0;
				float zmid = (Box.z-1.0)/2.0;
				float hyi = head.y - ymid;  // distance to channel centerline
				float hzi = head.z - zmid;  // "                            "
				float hri = sqrt(hyi*hyi + hzi*hzi);
						
				// radial position of rod tail:
				float3 tail = rods[i].r - offset*rods[i].p;			
				float tyi = tail.y - ymid;  // distance to channel centerline
				float tzi = tail.z - zmid;  // "                            "
				float tri = sqrt(tyi*tyi + tzi*tzi);
				
				// if head and tail are inside the radial wall, then exit:
				if (hri < radI && tri < radI) break;
				
			}
			
			// save the final updated state back to global memory
		    state[i] = localState;
		}	
		*/		
	}
}



// --------------------------------------------------------
// IBM3D kernel to sum the forces, torques, and moments of
// inertia for the rods:
// --------------------------------------------------------

__global__ void sum_disc_forces_torques_moments_IBM3D(
	beaddisc* beads,
	disc* discs,
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		int discID = beads[i].discID;
		float3 ricom = beads[i].rrel;
		float3 force = beads[i].f;
		float3 torque = cross(ricom,beads[i].f);
		float nBeadsPerDiscf = float(discs[discID].nBeads);
		// add up forces
		atomicAdd(&discs[discID].f.x, force.x);
		atomicAdd(&discs[discID].f.y, force.y);
		atomicAdd(&discs[discID].f.z, force.z);
		// add up torques
		atomicAdd(&discs[discID].t.x, torque.x);
		atomicAdd(&discs[discID].t.y, torque.y);
		atomicAdd(&discs[discID].t.z, torque.z);
		// add up fluid velocities
		atomicAdd(&discs[discID].uf.x, beads[i].uf.x/nBeadsPerDiscf);
		atomicAdd(&discs[discID].uf.y, beads[i].uf.y/nBeadsPerDiscf);
		atomicAdd(&discs[discID].uf.z, beads[i].uf.z/nBeadsPerDiscf);
		// add up gradient of fluid velocities
		atomicAdd(&discs[discID].gradu.xx, beads[i].gradu.xx/nBeadsPerDiscf);	
		atomicAdd(&discs[discID].gradu.xy, beads[i].gradu.xy/nBeadsPerDiscf);	
		atomicAdd(&discs[discID].gradu.xz, beads[i].gradu.xz/nBeadsPerDiscf);	
		atomicAdd(&discs[discID].gradu.yx, beads[i].gradu.yx/nBeadsPerDiscf);	
		atomicAdd(&discs[discID].gradu.yy, beads[i].gradu.yy/nBeadsPerDiscf);	
		atomicAdd(&discs[discID].gradu.yz, beads[i].gradu.yz/nBeadsPerDiscf);	
		atomicAdd(&discs[discID].gradu.zx, beads[i].gradu.zx/nBeadsPerDiscf);	
		atomicAdd(&discs[discID].gradu.zy, beads[i].gradu.zy/nBeadsPerDiscf);	
		atomicAdd(&discs[discID].gradu.zz, beads[i].gradu.zz/nBeadsPerDiscf);			
	}
}



// --------------------------------------------------------
// IBM3D kernel to unwrap bead coordinates.  Here, the
// beads of a disc are brought back close to the disc's 
// centerBead.  This is done to avoid complications with
// PBCs:
// --------------------------------------------------------

__global__ void unwrap_bead_coordinates_discs_IBM3D(
	beaddisc* beads,
	disc* discs,
	float3 Box,
	int3 pbcFlag,
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		int f = beads[i].discID;
		int j = discs[f].centerBead;
		float3 rij = beads[j].r - beads[i].r;
		float3 adjust = roundf(rij/Box)*Box*pbcFlag;
		beads[i].r = beads[i].r +  adjust;    // PBC's
		beads[i].rm1 = beads[i].rm1 + adjust; // PBC's	
	}
}



// --------------------------------------------------------
// IBM3D kernel to wrap bead coordinates for PBCs:
// --------------------------------------------------------

__global__ void wrap_bead_coordinates_IBM3D(
	beaddisc* beads,
	float3 Box,
	int3 pbcFlag,
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {	
		float3 adjust = floorf(beads[i].r/Box)*Box*pbcFlag;
		beads[i].r = beads[i].r - adjust;      // PBC's
		beads[i].rm1 = beads[i].rm1 - adjust;  // PBC's 
	}
}



// --------------------------------------------------------
// IBM3D kernel to wrap disc coordinates for PBCs:
// --------------------------------------------------------

__global__ void wrap_disc_coordinates_IBM3D(
	disc* discs,
	float3 Box,
	int3 pbcFlag,
	int nDiscs)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nDiscs) {	
		discs[i].r = discs[i].r - floorf(discs[i].r/Box)*Box*pbcFlag;			
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void bead_wall_forces_ydir_IBM3D(
	beaddisc* beads,
	float3 Box,
	float repA,
	float repD,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		const float d = repD;
		const float A = repA;
		const float yi = beads[i].r.y;
		// bottom wall
		if (yi < d) {
			const float force = A/pow(yi,2) - A/pow(d,2);
			beads[i].f.y += force;
			if (yi < 0.0001) beads[i].r.y = 0.0001;
		}
		// top wall
		else if (yi > (Box.y-1.0)-d) {
			const float bmyi = (Box.y-1.0) - yi;
			const float force = A/pow(bmyi,2) - A/pow(d,2);
			beads[i].f.y -= force;
			if (yi > Box.y-1.0001) beads[i].r.y = Box.y-1.0001;
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void bead_wall_forces_zdir_IBM3D(
	beaddisc* beads,
	float3 Box,
	float repA,
	float repD,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		const float d = repD;
		const float A = repA;
		const float zi = beads[i].r.z;
		// bottom wall
		if (zi < d) {
			const float force = A/pow(zi,2) - A/pow(d,2);
			beads[i].f.z += force;
			if (zi < 0.0001) beads[i].r.z = 0.0001;
		}
		// top wall
		else if (zi > (Box.z-1.0)-d) {
			const float bmzi = (Box.z-1.0) - zi;
			const float force = A/pow(bmzi,2) - A/pow(d,2);
			beads[i].f.z -= force;
			if (zi > Box.z-1.0001) beads[i].r.z = Box.z-1.0001;
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void bead_wall_forces_ydir_zdir_IBM3D(
	beaddisc* beads,
	float3 Box,
	float repA,
	float repD,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		const float d = repD;
		const float A = repA;
		const float yi = beads[i].r.y;
		const float zi = beads[i].r.z;
		// bottom wall
		if (yi < d) {
			const float force = A/pow(yi,2) - A/pow(d,2);
			beads[i].f.y += force;
			if (yi < 0.0001) beads[i].r.y = 0.0001;
		}
		// top wall
		else if (yi > (Box.y-1.0)-d) {
			const float bmyi = (Box.y-1.0) - yi;
			const float force = A/pow(bmyi,2) - A/pow(d,2);
			beads[i].f.y -= force;
			if (yi > Box.y-1.0001) beads[i].r.y = Box.y-1.0001;
		}
		// back wall
		if (zi < d) {
			const float force = A/pow(zi,2) - A/pow(d,2);
			beads[i].f.z += force;
			if (zi < 0.0001) beads[i].r.z = 0.0001;
		}
		// front wall
		else if (zi > (Box.z-1.0)-d) {
			const float bmzi = (Box.z-1.0) - zi;
			const float force = A/pow(bmzi,2) - A/pow(d,2);
			beads[i].f.z -= force;
			if (zi > Box.z-1.0001) beads[i].r.z = Box.z-1.0001;
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void bead_wall_forces_cylinder_IBM3D(
	beaddisc* beads,
	float3 Box,
	float Rad,
	float repA,
	float repD,
	float fric,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		const float d = repD;
		const float A = repA;
		const float ymid = (Box.y-1.0)/2.0;
		const float zmid = (Box.z-1.0)/2.0;
		const float yi = beads[i].r.y - ymid;  // distance to channel centerline
		const float zi = beads[i].r.z - zmid;  // "                            "
		const float ri = sqrt(yi*yi + zi*zi);
		// radial wall		
		if (ri > Rad - d) {
			// parabolic normal force:
			//const float bmri = Rad - ri;
			//const float force = A/pow(bmri,2) - A/pow(d,2);
			//beads[i].f.y -= force*(yi/ri);
			//beads[i].f.z -= force*(zi/ri);	
			
			// linear normal force:
			const float delta = (ri + d) - Rad;  // distance protruding into wall
			const float force = A*delta;
			beads[i].f.y -= force*(yi/ri);
			beads[i].f.z -= force*(zi/ri);
			
			// friction force:
			beads[i].f.x -= fric*force;	
		}				
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces (cone-shaped nozzle):
// --------------------------------------------------------

__global__ void bead_wall_forces_nozzle_IBM3D(
	beaddisc* beads,
	float3 Box,
	float lenCyl,
	float radIn,
	float radOut,
	float repA,
	float repD,
	float fric,
	float lubforceMax,
	int nBeads)
{	
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		
		// parameters:
		const float d = repD;   // this should be = bead radius
		const float A = repA;
		
		// distance to channel centerline 
		const float ymid = (Box.y-1.0)/2.0;
		const float zmid = (Box.z-1.0)/2.0;
		const float yi = beads[i].r.y - ymid;  
		const float zi = beads[i].r.z - zmid;
		const float ri = sqrt(yi*yi + zi*zi);
		
		// nozzle half-angle alpha:
		const float dR = radIn - radOut;
		const float Lnoz = Box.x - lenCyl;
		const float inv_hyp = 1.0f / sqrtf(Lnoz*Lnoz + dR*dR);
		float cos_alpha = inv_hyp*Lnoz;
		float sin_alpha = inv_hyp*dR;
		if (beads[i].r.x < lenCyl) {   // if we're in the straight cylinder section
			cos_alpha = 1.0f;
			sin_alpha = 0.0f;
		}
		
		// nozzle radius at bead's position:
		float Rad = radIn;
		if (beads[i].r.x > lenCyl) Rad = radIn + (radOut - radIn)*(beads[i].r.x-lenCyl)/(Box.x-lenCyl);
		
		// distance from bead center to wall along direction normal to wall:
		const float gap = (Rad - ri)*cos_alpha - d;
						
		// lubrication force with wall
		if (gap > 0.0 && gap < d) {
			// fluid kinematic viscosity:
			float nu = 0.1666666667;
			// outward unit normal vector pointing into the wall:
			float3 n = make_float3(sin_alpha, cos_alpha*(yi/ri), cos_alpha*(zi/ri));			
			// normal lubrication force:
			float gapMax = d;
			float invgap = 1.0/gap - 1.0/gapMax;
			float velN = dot(beads[i].v,n);						
			float lubforceN = 6.0*M_PI*nu*d*d*invgap*velN;
			float lubforceNmag = abs(lubforceN);
			if (lubforceNmag > lubforceMax) lubforceN *= (lubforceMax/lubforceNmag);
			beads[i].f -= lubforceN*n;
			// tangential lubrication force:
			float3 velT = beads[i].v - velN*n;
			float velTmag = length(velT);
			if (velTmag > 1.0e-9){
				float lubforceT = 6.0*M_PI*nu*d*log(gapMax/gap)*velTmag;
				float lubforceTmag = abs(lubforceT);
				if (lubforceTmag > lubforceMax) lubforceT *= (lubforceMax/lubforceTmag);
				beads[i].f -= lubforceT*(velT/velTmag);
			}			
		}
		
		// contact force with wall		
		if (gap < 0.0) {
			// outward unit normal vector pointing into the wall:
			float3 n = make_float3(sin_alpha, cos_alpha*(yi/ri), cos_alpha*(zi/ri));		
			// linear normal force:
			const float deltaN = abs(gap);
			const float forceN = A*deltaN;
			beads[i].f -= forceN*n;		
			// tangential friction force:
			float velN = dot(beads[i].v,n);
			float3 velT = beads[i].v - velN*n;
			float velTmag = length(velT);
			float3 deltaT = beads[i].wallContactHist;
			deltaT += velT*1.0;      // assume timestep dt = 1.0
			beads[i].f -= A*deltaT;  // may want to cap forceT by fric*forceN
			beads[i].wallContactHist = deltaT;			
		}
		// no contact with wall
		else {
			// reset contact wall history:
			beads[i].wallContactHist = make_float3(0.0f);
		}		
					
	}

}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void push_beads_into_sphere_IBM3D(
	beaddisc* beads,
	float xs,
	float ys,
	float zs,
	float rs,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		float3 sphere = make_float3(xs,ys,zs);
		float3 ris = beads[i].r - sphere;
		float r = length(ris);
		if (r > (rs-1.5)) {
			ris /= r;
			beads[i].f -= 0.0005*ris;
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void push_beads_into_cylinder_IBM3D(
	beaddisc* beads,
	float3 Box,
	float Rad,
	float repA,
	float repD,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		const float d = repD;
		const float A = repA;
		const float ymid = (Box.y-1.0)/2.0;
		const float zmid = (Box.z-1.0)/2.0;
		const float yi = beads[i].r.y - ymid;  // distance to channel centerline
		const float zi = beads[i].r.z - zmid;  // "                            "
		const float ri = sqrt(yi*yi + zi*zi);
		// radial wall		
		if (ri > Rad - d) {
			const float outside_dist = ri - (Rad - d);
			const float force = 0.01*outside_dist;
			beads[i].f.y -= force*(yi/ri);
			beads[i].f.z -= force*(zi/ri);			
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void push_beads_into_duct_IBM3D(
	beaddisc* beads,
	float3 Box,
	float repA,
	float repD,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		const float d = repD;
		const float A = repA;
		const float yi = beads[i].r.y;
		const float zi = beads[i].r.z;
		// bottom wall
		if (yi < d) {
			const float outside_dist = d + (0.0 - yi);
			const float force = 0.01*outside_dist;
			beads[i].f.y += force;
		}
		// top wall
		else if (yi > Box.y-1.0-d) {
			const float outside_dist = yi - (Box.y-1.0-d);
			const float force = 0.01*outside_dist;
			beads[i].f.y -= force;
		}
		// back wall
		if (zi < d) {
			const float outside_dist = d + (0.0 - zi);
			const float force = 0.01*outside_dist;
			beads[i].f.z += force;
		}
		// front wall
		else if (zi > Box.z-1.0-d) {
			const float outside_dist = zi - (Box.z-1.0-d);
			const float force = 0.01*outside_dist;
			beads[i].f.z -= force;
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void push_beads_into_slit_IBM3D(
	beaddisc* beads,
	float3 Box,
	float repA,
	float repD,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		const float d = repD;
		const float A = repA;
		const float zi = beads[i].r.z;
		// bottom wall
		if (zi < d) {
			const float outside_dist = d + (0.0 - zi);
			const float force = 0.01*outside_dist;
			beads[i].f.z += force;
		}
		// top wall
		else if (zi > Box.z-1.0-d) {
			const float outside_dist = zi - (Box.z-1.0-d);
			const float force = 0.01*outside_dist;
			beads[i].f.z -= force;
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate wall forces:
// --------------------------------------------------------

__global__ void push_beads_into_nozzle_IBM3D(
	beaddisc* beads,
	float3 Box,
	float lenCyl,
	float radIn,
	float radOut,
	float repA,
	float repD,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {
		const float d = repD;
		const float A = repA;
		const float ymid = (Box.y-1.0)/2.0;
		const float zmid = (Box.z-1.0)/2.0;
		const float yi = beads[i].r.y - ymid;  // distance to channel centerline
		const float zi = beads[i].r.z - zmid;  // "                            "
		const float ri = sqrt(yi*yi + zi*zi);
		
		// nozzle radius at bead's position:
		float Rad = radIn;
		if (beads[i].r.x > lenCyl) Rad = radIn + (radOut - radIn)*(beads[i].r.x-lenCyl)/(Box.x-lenCyl);
		
		// radial wall		
		if (ri > Rad - d) {
			const float outside_dist = ri - (Rad - d);
			const float force = 0.01*outside_dist;
			beads[i].f.y -= force*(yi/ri);
			beads[i].f.z -= force*(zi/ri);			
		}
	}
}



// --------------------------------------------------------
// IBM3D kernel to determine the hydrodynamic force at a 
// bead, then extrapolate it to the LBM lattice. 
// --------------------------------------------------------

__global__ void hydrodynamic_force_bead_disc_IBM3D(
	beaddisc* beads,
	float* fxLBM,
	float* fyLBM,
	float* fzLBM,
	float* uLBM,
	float* vLBM,
	float* wLBM,
	float dt,
	int nBeadsPerRod,
	int Nx,
	int Ny,
	int Nz,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nBeads) {
			
		// --------------------------------------
		// find nearest LBM voxel (rounded down)
		// --------------------------------------
		
		int i0 = int(floor(beads[i].r.x));
		int j0 = int(floor(beads[i].r.y));
		int k0 = int(floor(beads[i].r.z));
		
		// --------------------------------------
		// only do this if bead is in LBM domain:
		// --------------------------------------
		
		if (i0 >= 0 && j0 >= 0 && k0 >= 0) {
			
			// --------------------------------------
			// loop over footprint to get 
			// interpolated LBM velocity:
			// --------------------------------------
				
			float vxLBMi = 0.0;
			float vyLBMi = 0.0;
			float vzLBMi = 0.0;
			
			for (int kk=k0; kk<=k0+1; kk++) {
				for (int jj=j0; jj<=j0+1; jj++) {
					for (int ii=i0; ii<=i0+1; ii++) {				
						int ndx = disc_voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
						float rx = beads[i].r.x - float(ii);
						float ry = beads[i].r.y - float(jj);
						float rz = beads[i].r.z - float(kk);
						float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
						vxLBMi += del*uLBM[ndx];
						vyLBMi += del*vLBM[ndx];
						vzLBMi += del*wLBM[ndx];			
					}
				}
			}
				
			// --------------------------------------
			// calculate hydrodynamic forces 
			// --------------------------------------
		
			float fx = (vxLBMi - beads[i].v.x)/dt/100.0; // /float(nBeadsPerRod);
			float fy = (vyLBMi - beads[i].v.y)/dt/100.0; // /float(nBeadsPerRod);
			float fz = (vzLBMi - beads[i].v.z)/dt/100.0; // /float(nBeadsPerRod);
						
			// --------------------------------------
			// distribute the !negative! of the 
			// hydrodynamic bead force to the LBM
			// fluid:
			// --------------------------------------
		
			for (int kk=k0; kk<=k0+1; kk++) {
				for (int jj=j0; jj<=j0+1; jj++) {
					for (int ii=i0; ii<=i0+1; ii++) {				
						int ndx = disc_voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
						float rx = beads[i].r.x - float(ii);
						float ry = beads[i].r.y - float(jj);
						float rz = beads[i].r.z - float(kk);
						float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
						atomicAdd(&fxLBM[ndx],-del*fx);
						atomicAdd(&fyLBM[ndx],-del*fy);
						atomicAdd(&fzLBM[ndx],-del*fz);				
					}
				}
			}		
			
		}		
	}	
}



// --------------------------------------------------------
// IBM3D kernel to extrapolate the disc's force to the LBM
// lattice.  This is done by averaging the disc's force to
// every bead. 
// --------------------------------------------------------

__global__ void extrapolate_force_bead_disc_IBM3D(
	beaddisc* beads,
	disc* discs,
	float* fxLBM,
	float* fyLBM,
	float* fzLBM,
	int Nx,
	int Ny,
	int Nz,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nBeads) {
		
		// --------------------------------------
		// find the averaged disc force for each
		// bead
		// --------------------------------------
		
		int discID = beads[i].discID;
		float3 beadForce = discs[discID].f/discs[discID].nBeads;
		
		// --------------------------------------
		// the rod torque is converted to a force
		// by calculating the force couple that
		// produces the same torque, assuming the
		// separation distance is 1/2 the rod length.
		// The force couple is then spread across
		// the rod
		// --------------------------------------
		
		/*
		float Lrod = float(discs[discID].nBeads-1)*L0;
		float Lrod2 = Lrod/2.0;
		float3 FcoupleSep = Lrod2*rods[rodID].p;
		float3 Fcouple = cross(rods[rodID].t,FcoupleSep)/(Lrod2*Lrod2);
		float nBeadsPerRod2 = float(rods[rodID].nBeads-1)/2.0;  //float(nBeadsPerRod-1)/2.0;
		Fcouple /= nBeadsPerRod2;
		if (i > rods[rodID].centerBead) Fcouple *= -1.0;
		beadForce += Fcouple;
		*/
				
		// --------------------------------------
		// find nearest LBM voxel (rounded down)
		// --------------------------------------
		
		int i0 = int(floor(beads[i].r.x));
		int j0 = int(floor(beads[i].r.y));
		int k0 = int(floor(beads[i].r.z));
		
		// --------------------------------------
		// loop over footprint (only if bead is in
		// LBM domain)
		// --------------------------------------
		
		if (i0 >= 0 && j0 >= 0 && k0 >= 0) {
		
			for (int kk=k0; kk<=k0+1; kk++) {
				for (int jj=j0; jj<=j0+1; jj++) {
					for (int ii=i0; ii<=i0+1; ii++) {				
						int ndx = disc_voxel_ndx(ii,jj,kk,Nx,Ny,Nz);
						float rx = beads[i].r.x - float(ii);
						float ry = beads[i].r.y - float(jj);
						float rz = beads[i].r.z - float(kk);
						float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));
						atomicAdd(&fxLBM[ndx],del*beadForce.x);
						atomicAdd(&fyLBM[ndx],del*beadForce.y);
						atomicAdd(&fzLBM[ndx],del*beadForce.z);
					}
				}		
			}
			
		}	
	}	
}



// --------------------------------------------------------
// IBM3D kernel to interpolate the fluid velocity and 
// gradient of the fluid velocity at each bead position. 
// --------------------------------------------------------

__global__ void interpolate_gradient_of_velocity_bead_IBM3D(
	beaddisc* beads,
	float* uLBM,
	float* vLBM,
	float* wLBM,
	int Nx,
	int Ny,
	int Nz,
	int nBeads)
{
	// define node:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	
	if (i < nBeads) {
				
		// --------------------------------------
		// find nearest LBM voxel (rounded down)
		// --------------------------------------
		
		int i0 = int(floor(beads[i].r.x));
		int j0 = int(floor(beads[i].r.y));
		int k0 = int(floor(beads[i].r.z));
		
		// --------------------------------------
		// loop over footprint to get 
		// interpolated LBM velocity:
		// --------------------------------------
		
		if (i0 >= 0 && j0 >= 0 && k0 >= 0) {
			
			float dudx = 0.0;
			float dudy = 0.0;
			float dudz = 0.0;
			float dvdx = 0.0;
			float dvdy = 0.0;
			float dvdz = 0.0;
			float dwdx = 0.0;
			float dwdy = 0.0;
			float dwdz = 0.0;
			float uLBMi = 0.0;
			float vLBMi = 0.0;
			float wLBMi = 0.0;
		
			for (int kk=k0; kk<=k0+1; kk++) {
				for (int jj=j0; jj<=j0+1; jj++) {
					for (int ii=i0; ii<=i0+1; ii++) {				
						float rx = beads[i].r.x - float(ii);
						float ry = beads[i].r.y - float(jj);
						float rz = beads[i].r.z - float(kk);
						float del = (1.0-abs(rx))*(1.0-abs(ry))*(1.0-abs(rz));								
						dudx += del*x_deriv(ii,jj,kk,Nx,Ny,Nz,uLBM);
						dudy += del*y_deriv(ii,jj,kk,Nx,Ny,Nz,uLBM);
						dudz += del*z_deriv(ii,jj,kk,Nx,Ny,Nz,uLBM);
						dvdx += del*x_deriv(ii,jj,kk,Nx,Ny,Nz,vLBM);
						dvdy += del*y_deriv(ii,jj,kk,Nx,Ny,Nz,vLBM);
						dvdz += del*z_deriv(ii,jj,kk,Nx,Ny,Nz,vLBM);
						dwdx += del*x_deriv(ii,jj,kk,Nx,Ny,Nz,wLBM);
						dwdy += del*y_deriv(ii,jj,kk,Nx,Ny,Nz,wLBM);
						dwdz += del*z_deriv(ii,jj,kk,Nx,Ny,Nz,wLBM);
						int ndx = disc_voxel_ndx(ii,jj,kk,Nx,Ny,Nz);	
						uLBMi += del*uLBM[ndx];
						vLBMi += del*vLBM[ndx];
						wLBMi += del*wLBM[ndx];					
					}
				}
			}
			
			// --------------------------------------
			// assign grad(u) to bead
			// --------------------------------------
				
			beads[i].uf.x = uLBMi;
			beads[i].uf.y = vLBMi;
			beads[i].uf.z = wLBMi;
			beads[i].gradu.xx = dudx;
			beads[i].gradu.xy = dudy;
			beads[i].gradu.xz = dudz;
			beads[i].gradu.yx = dvdx;
			beads[i].gradu.yy = dvdy;
			beads[i].gradu.yz = dvdz;
			beads[i].gradu.zx = dwdx;
			beads[i].gradu.zy = dwdy;
			beads[i].gradu.zz = dwdz;
			
		}
		
		// --------------------------------------
		// if bead is outside LBM domain, assign
		// zeros to u and gradu:
		// --------------------------------------
			
		else {
			beads[i].uf.x = 0.0;
			beads[i].uf.y = 0.0;
			beads[i].uf.z = 0.0;
			beads[i].gradu.xx = 0.0;
			beads[i].gradu.xy = 0.0;
			beads[i].gradu.xz = 0.0;
			beads[i].gradu.yx = 0.0;
			beads[i].gradu.yy = 0.0;
			beads[i].gradu.yz = 0.0;
			beads[i].gradu.zx = 0.0;
			beads[i].gradu.zy = 0.0;
			beads[i].gradu.zz = 0.0;
		}						
	}	
}



// --------------------------------------------------------
// IBM3D kernel to assign beads to bins:
// --------------------------------------------------------

__global__ void build_bin_lists_for_beads_IBM3D(
	beaddisc* beads,
	bindata bins,
	int nBeads)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {		
		
		// -------------------------------
		// calculate bin ID:
		// -------------------------------
		
		if (beads[i].r.x < 0.0) return;
		
		int binID = int(floor(beads[i].r.x/bins.sizeBins))*bins.numBins.z*bins.numBins.y +  
			        int(floor(beads[i].r.y/bins.sizeBins))*bins.numBins.z +
		            int(floor(beads[i].r.z/bins.sizeBins));		
						
		// -------------------------------
		// update the lists:
		// -------------------------------
		
		if (binID >= 0 && binID < bins.nBins-1) {
			atomicAdd(&bins.binOccupancy[binID],1);
			int offst = binID*bins.binMax;
			for (int k=offst; k<offst+bins.binMax; k++) {
				int flag = atomicCAS(&bins.binMembers[k],-1,i); 
				if (flag == -1) break;  
			}
		}
		
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate nonbonded bead interactions
// using the bin lists:
// --------------------------------------------------------

__global__ void nonbonded_bead_interactions_IBM3D(
	beaddisc* beads,
	bindata bins,
	float repA,
	float repD,
	float lubforceMax,
	int nBeads,
	float3 Box,	
	int3 pbcFlag)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {		
		
		// -------------------------------
		// calculate bin ID:
		// -------------------------------
		
		int binID = int(floor(beads[i].r.x/bins.sizeBins))*bins.numBins.z*bins.numBins.y +  
			        int(floor(beads[i].r.y/bins.sizeBins))*bins.numBins.z +
		            int(floor(beads[i].r.z/bins.sizeBins));		
		
		if (binID < 0 || binID > bins.nBins-1) return; 
		
		// -------------------------------
		// loop over beads in the same bin:
		// -------------------------------
				
		int offst = binID*bins.binMax;
		int occup = bins.binOccupancy[binID];
		if (occup > bins.binMax) {
			printf("occup = %i in Bin %i\n", occup, binID);
			occup = bins.binMax;
		}
								
		for (int k=offst; k<offst+occup; k++) {
			int j = bins.binMembers[k];
			if (i==j) continue;
			if (beads[i].discID == beads[j].discID) continue;
			pairwise_bead_interaction_forces(i,j,repA,repD,lubforceMax,beads,Box,pbcFlag);			
		}
		
		// -------------------------------
		// loop over neighboring bins:
		// -------------------------------
		
        for (int b=0; b<bins.nnbins; b++) {
            // get neighboring bin ID
			int naborbinID = bins.binMap[binID*bins.nnbins + b];
			offst = naborbinID*bins.binMax;
			occup = bins.binOccupancy[naborbinID];
			if (occup > bins.binMax) occup = bins.binMax;
			// loop over beads in this bin:
			for (int k=offst; k<offst+occup; k++) {
				int j = bins.binMembers[k];
				if (beads[i].discID == beads[j].discID) continue;				
				pairwise_bead_interaction_forces(i,j,repA,repD,lubforceMax,beads,Box,pbcFlag);			
			}
		}
				
	}
}



// --------------------------------------------------------
// IBM3D kernel to calculate nonbonded bead interactions
// using the bin lists:
// --------------------------------------------------------

__global__ void nonbonded_bead_interactions_with_friction_IBM3D(
	beaddisc* beads,
	bindata bins,
	float repA,
	float repD,
	float lubforceMax,
	int nBeads,
	float3 Box,	
	int3 pbcFlag)
{
	// define bead:
	int i = blockIdx.x*blockDim.x + threadIdx.x;		
	if (i < nBeads) {		
		
		// -------------------------------
		// calculate bin ID:
		// -------------------------------
		
		int binID = int(floor(beads[i].r.x/bins.sizeBins))*bins.numBins.z*bins.numBins.y +  
			        int(floor(beads[i].r.y/bins.sizeBins))*bins.numBins.z +
		            int(floor(beads[i].r.z/bins.sizeBins));		
		
		if (binID < 0 || binID > bins.nBins-1) return;
		
		// -------------------------------
		// loop over beads in the same bin:
		// -------------------------------
				
		int offst = binID*bins.binMax;
		int occup = bins.binOccupancy[binID];
		if (occup > bins.binMax) {
			printf("occup = %i in Bin %i\n", occup, binID);
			occup = bins.binMax;
		}
								
		for (int k=offst; k<offst+occup; k++) {
			int j = bins.binMembers[k];
			if (i==j) continue;
			if (beads[i].discID == beads[j].discID) continue;
			pairwise_bead_interaction_forces_with_friction(i,j,repA,repD,lubforceMax,beads,Box,pbcFlag);			
		}
		
		// -------------------------------
		// loop over neighboring bins:
		// -------------------------------
		
        for (int b=0; b<bins.nnbins; b++) {
            // get neighboring bin ID
			int naborbinID = bins.binMap[binID*bins.nnbins + b];
			offst = naborbinID*bins.binMax;
			occup = bins.binOccupancy[naborbinID];
			if (occup > bins.binMax) occup = bins.binMax;
			// loop over beads in this bin:
			for (int k=offst; k<offst+occup; k++) {
				int j = bins.binMembers[k];
				if (beads[i].discID == beads[j].discID) continue;				
				pairwise_bead_interaction_forces_with_friction(i,j,repA,repD,lubforceMax,beads,Box,pbcFlag);			
			}
		}
				
	}
}













// **********************************************************************************************
// Miscellaneous kernels and functions
// **********************************************************************************************













// --------------------------------------------------------
// IBM3D kernel to calculate i-j lubrication force.  The 
// model comes from Ladd & Verberg, Journal of Statistical
// Physics, 104 (2001) 1191.  See Eq. (74).
// --------------------------------------------------------

__device__ inline void pairwise_bead_interaction_forces(
	const int i, 
	const int j,
	const float repA,
	const float repD,
	const float lubforceMax,
	beaddisc* beads,
	float3 Box,
	int3 pbcFlag)
{
	float3 rij = beads[i].r - beads[j].r;
	rij -= roundf(rij/Box)*Box*pbcFlag;  // PBC's	
	const float r = length(rij);	
	const float Ri = 0.5*repD;  // bead radius
	const float Rj = 0.5*repD;  // bead radius 
	const float gapMax = 0.25;  //0.05;  // max gap for lubrication forces
	const float cutoff = Ri + Rj + gapMax;		
		
	// interaction range:
	if (r < cutoff) {			
		
		float3 uij = rij/r;
		float3 vij = beads[i].v - beads[j].v;
		
		// lubrication force:
		if (r > (Ri+Rj)) {
			const float nu = 0.1666666667;
			// normal lubrication force:
			float coeff = (Ri*Rj*Ri*Rj)/(Ri+Rj)/(Ri+Rj);
			float udotv = dot(uij,vij);
			float gap = r - Ri - Rj;
			if (gap < 0.001) gap = 0.001;
			float invgap = 1.0/gap - 1.0/gapMax;		
			float lubforce = -6.0*M_PI*nu*coeff*udotv*invgap;
			float lubforcemag = abs(lubforce);
			if (lubforcemag > lubforceMax) lubforce *= (lubforceMax/lubforcemag);
			beads[i].f += lubforce*(uij);
			// tangential lubrication force:
			float3 uTanij = vij - udotv*uij;  // tangential relative velocity
			float3 lubforceTan = -6.0*M_PI*nu*Ri*log(gapMax/gap)*uTanij;
			float lubforceTanmag = length(lubforceTan);
			if (lubforceTanmag > lubforceMax) lubforceTan *= (lubforceMax/lubforceTanmag);
			beads[i].f += lubforceTan;					
		}
		
		// contact force:
		if (r < (Ri+Rj)) {
			// normal force
			float force = repA - (repA/repD)*r;
			beads[i].f += force*uij;						
		}
	}	
}



// --------------------------------------------------------
// IBM3D kernel to calculate i-j lubrication force.  The 
// model comes from Ladd & Verberg, Journal of Statistical
// Physics, 104 (2001) 1191.  See Eq. (74).
// --------------------------------------------------------

__device__ inline void pairwise_bead_interaction_forces_with_friction(
	const int i, 
	const int j,
	const float repA,
	const float repD,
	const float lubforceMax,
	beaddisc* beads,
	float3 Box,
	int3 pbcFlag)
{
	float3 rij = beads[i].r - beads[j].r;
	rij -= roundf(rij/Box)*Box*pbcFlag;  // PBC's	
	const float r = length(rij);	
	const float Ri = 0.5*repD;  // bead radius
	const float Rj = 0.5*repD;  // bead radius 
	const float gapMax = 0.25;  // 0.05 (max gap for lubrication forces)
	const float cutoff = Ri + Rj + gapMax;		
		
	// interaction range:
	if (r < cutoff) {			
		
		float3 uij = rij/r;
		float3 vij = beads[i].v - beads[j].v;
		float udotv = dot(uij,vij);
		
		// lubrication force:
		if (r > (Ri+Rj)) {
			const float nu = 0.1666666667;
			// normal lubrication force:
			float coeff = (Ri*Rj*Ri*Rj)/(Ri+Rj)/(Ri+Rj);			
			float gap = r - Ri - Rj;
			if (gap < 0.001) gap = 0.001;
			float invgap = 1.0/gap - 1.0/gapMax;		
			float lubforce = -6.0*M_PI*nu*coeff*udotv*invgap;
			float lubforcemag = abs(lubforce);
			if (lubforcemag > lubforceMax) lubforce *= (lubforceMax/lubforcemag);
			beads[i].f += lubforce*(uij);
			// tangential lubrication force:
			float3 uTanij = vij - udotv*uij;  // tangential relative velocity
			float3 lubforceTan = -6.0*M_PI*nu*Ri*log(gapMax/gap)*uTanij;
			float lubforceTanmag = length(lubforceTan);
			if (lubforceTanmag > lubforceMax) lubforceTan *= (lubforceMax/lubforceTanmag);
			beads[i].f += lubforceTan;
		}
		
		// contact force:
		if (r < (Ri+Rj)) {
			// normal force
			float forceN = repA - (repA/repD)*r;
			beads[i].f += forceN*uij;
			// tangential (friction) force
			float dt = 1.0;  // assumed time step
			float3 uTij = (vij - udotv*uij)*dt;
			float uT = length(uTij);	
			if (uT > 0.0f) {
				float fric = 0.5;
				float forceT = min(0.1*uT,fric*forceN);
				beads[i].f -= forceT*(uTij/uT);		
			}							
		}
	}	
}



// --------------------------------------------------------
// IBM3D kernel to x-derivative of velocity field
// --------------------------------------------------------

__device__ inline float x_deriv(
	const int i,
	const int j,
	const int k,  
	const int Nx,
	const int Ny,
	const int Nz,
	float* u)
{
	int ndx_plus = disc_voxel_ndx(i+1,j,k,Nx,Ny,Nz);
	int ndx_down = disc_voxel_ndx(i-1,j,k,Nx,Ny,Nz);
	return (u[ndx_plus] - u[ndx_down])/2.0;  // assume dx=1
}



// --------------------------------------------------------
// IBM3D kernel to y-derivative of velocity field
// --------------------------------------------------------

__device__ inline float y_deriv(
	const int i,
	const int j,
	const int k,  
	const int Nx,
	const int Ny,
	const int Nz,
	float* u)
{
	int ndx_plus = disc_voxel_ndx(i,j+1,k,Nx,Ny,Nz);
	int ndx_down = disc_voxel_ndx(i,j-1,k,Nx,Ny,Nz);
	return (u[ndx_plus] - u[ndx_down])/2.0;  // assume dx=1
}



// --------------------------------------------------------
// IBM3D kernel to z-derivative of velocity field
// --------------------------------------------------------

__device__ inline float z_deriv(
	const int i,
	const int j,
	const int k,  
	const int Nx,
	const int Ny,
	const int Nz,
	float* u)
{
	int ndx_plus = disc_voxel_ndx(i,j,k+1,Nx,Ny,Nz);
	int ndx_down = disc_voxel_ndx(i,j,k-1,Nx,Ny,Nz);
	return (u[ndx_plus] - u[ndx_down])/2.0;  // assume dx=1
}



// --------------------------------------------------------
// IBM3D kernel to determine 1D index from 3D indices:
// --------------------------------------------------------

__device__ inline int disc_voxel_ndx(
	int i,
	int j,
	int k,
	int Nx,
	int Ny,
	int Nz)
{
    if (i < 0) i += Nx;
    if (i >= Nx) i -= Nx;
    if (j < 0) j += Ny;
    if (j >= Ny) j -= Ny;
    if (k < 0) k += Nz;
    if (k >= Nz) k -= Nz;
    return k*Nx*Ny + j*Nx + i;	
}









