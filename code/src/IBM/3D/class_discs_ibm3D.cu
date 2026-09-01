 
# include "class_discs_ibm3D.cuh"
# include "../../IO/GetPot"
# include "../../Utils/eig3.cuh"
# include <math.h>
# include <iostream>
# include <iomanip>
# include <fstream>
# include <string>
# include <sstream>
# include <stdlib.h>
# include <time.h>
using namespace std;  











// **********************************************************************************************
// Constructor, destructor, and array allocations...
// **********************************************************************************************











// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

class_discs_ibm3D::class_discs_ibm3D()
{
	// get some parameters:
	GetPot inputParams("input.dat");
	
	// mesh attributes
	nDiscs = inputParams("IBM_DISCS/nDiscs",1);
	nBeadsPerDisc = inputParams("IBM_DISCS/nBeadsPerDisc",0);
	nBeads = nBeadsPerDisc*nDiscs;
	
	// mechanical properties	
	repA = inputParams("IBM_DISCS/repA",0.0);
	repD = inputParams("IBM_DISCS/repD",0.0);
	lubforceMax = inputParams("IBM_DISCS/lubforceMax",0.0);
	repWall = inputParams("IBM_DISCS/repWall",0.0);
	fricWall = inputParams("IBM_DISCS/fricWall",0.0);
	beadFmax = inputParams("IBM_DISCS/beadFmax",1000.0);
	discFmax = inputParams("IBM_DISCS/discFmax",1000.0);
	discTmax = inputParams("IBM_DISCS/discTmax",1000.0);
		
	// domain attributes
	dt = inputParams("Time/dt",1.0);	
	N.x = inputParams("Lattice/Nx",1);
	N.y = inputParams("Lattice/Ny",1);
	N.z = inputParams("Lattice/Nz",1);
	Box.x = float(N.x);   // assume dx=1
	Box.y = float(N.y);
	Box.z = float(N.z);
	pbcFlag = make_int3(1,1,1);
	chRad = inputParams("Lattice/chRad",10.0);
			
	// if we need bins, do some calculations:
	binsFlag = false;
	if (nDiscs > 1) binsFlag = true;
	if (binsFlag) {		
		bins.sizeBins = inputParams("IBM_DISCS/sizeBins",2.0);
		bins.binMax = inputParams("IBM_DISCS/binMax",1);			
		bins.numBins.x = int(floor(N.x/bins.sizeBins));
	    bins.numBins.y = int(floor(N.y/bins.sizeBins));
	    bins.numBins.z = int(floor(N.z/bins.sizeBins));
		bins.nBins = bins.numBins.x*bins.numBins.y*bins.numBins.z;
		bins.nnbins = 26;
	}	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

class_discs_ibm3D::~class_discs_ibm3D()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void class_discs_ibm3D::allocate()
{
	// allocate array memory (host):
	beadsH = (beaddisc*)malloc(nBeads*sizeof(beaddisc));
	discsH = (disc*)malloc(nDiscs*sizeof(disc));
							
	// allocate array memory (device):	
	cudaMalloc((void **) &beads, nBeads*sizeof(beaddisc));
	cudaMalloc((void **) &discs, nDiscs*sizeof(disc));
	cudaMalloc((void **) &states, nDiscs*sizeof(curandState));
	if (binsFlag) {		
		cudaMalloc((void **) &bins.binMembers, bins.nBins*bins.binMax*sizeof(int));
		cudaMalloc((void **) &bins.binOccupancy, bins.nBins*sizeof(int));
		cudaMalloc((void **) &bins.binMap, bins.nBins*26*sizeof(int));		
	}	
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void class_discs_ibm3D::deallocate()
{
	// free array memory (host):
	free(beadsH);
	free(discsH);
					
	// free array memory (device):
	cudaFree(beads);
	cudaFree(discs);
	if (binsFlag) {		
		cudaFree(bins.binMembers);
		cudaFree(bins.binOccupancy);
		cudaFree(bins.binMap);				
	}		
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void class_discs_ibm3D::memcopy_host_to_device()
{
	cudaMemcpy(beads, beadsH, sizeof(beaddisc)*nBeads, cudaMemcpyHostToDevice);	
	cudaMemcpy(discs, discsH, sizeof(disc)*nDiscs, cudaMemcpyHostToDevice);	
}
	


// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_discs_ibm3D::memcopy_device_to_host()
{
	cudaMemcpy(beadsH, beads, sizeof(beaddisc)*nBeads, cudaMemcpyDeviceToHost);
	cudaMemcpy(discsH, discs, sizeof(disc)*nDiscs, cudaMemcpyDeviceToHost);
	
	// unwrap coordinate positions:
	unwrap_bead_coordinates(); 
}











// **********************************************************************************************
// Initialization Stuff...
// **********************************************************************************************












// --------------------------------------------------------
// Create the first rod:
// --------------------------------------------------------

void class_discs_ibm3D::create_first_disc()
{
	// read bead information for the first disc:
	std::string fname = "disc.dat";
	ifstream infile;
	infile.open(fname, ios::in);
	int nB,centerBead;
	
	infile >> nB >> centerBead;
	if (nB != nBeadsPerDisc) cout << "number of IBM beads is NOT consistent with input file = " << nB << endl;
	
	for (int i=0; i<nB; i++) {
		infile >> beadsH[i].r.x >> beadsH[i].r.y >> beadsH[i].r.z;
		beadsH[i].rm1 = beadsH[i].r;
		beadsH[i].f = make_float3(0.0f);
		beadsH[i].v = make_float3(0.0f);
		beadsH[i].uf = make_float3(0.0f);
		beadsH[i].wallContactHist = make_float3(0.0f);
		beadsH[i].discID = 0;
	}
	
	// each bead calculates distance to disc center-of-mass:
	for (int i=0; i<nB; i++) {
		beadsH[i].rrel = beadsH[i].r - beadsH[centerBead].r;
	}	
			
	// set up indices for ALL discs:
	for (int i=0; i<nDiscs; i++) {
		discsH[i].discType = 0;
		discsH[i].nBeads = nBeadsPerDisc;
		discsH[i].indxB0 = i*nBeadsPerDisc;                   // start index for beads
		discsH[i].centerBead = i*nBeadsPerDisc + centerBead;  // center-of-mass
		discsH[i].q.set_values(0.0f,0.0f,0.0f,1.0f);          // initial normal vector is along z (0,0,1)
	}
}



// --------------------------------------------------------
// Setters:
// --------------------------------------------------------

void class_discs_ibm3D::set_pbcFlag(int x, int y, int z)
{
	pbcFlag.x = x; pbcFlag.y = y; pbcFlag.z = z;
}

void class_discs_ibm3D::set_discs_radii(float rad)
{
	// set radius for ALL discs:
	for (int i=0; i<nDiscs; i++) discsH[i].rad = rad;
}

void class_discs_ibm3D::set_disc_radius(int dID, float rad)
{
	// set radius for ONE disc:
	discsH[dID].rad = rad;
}

void class_discs_ibm3D::set_discs_half_thickness(float val)
{
	// set h2 for ALL discs:
	for (int i=0; i<nDiscs; i++) discsH[i].h2 = val;
}

void class_discs_ibm3D::set_disc_half_thickness(int dID, float val)
{
	// set h2 for ONE disc:
	discsH[dID].h2 = val;
}

void class_discs_ibm3D::set_discs_types(int val)
{
	// set discType for ALL discs:
	for (int i=0; i<nDiscs; i++) discsH[i].discType = val;
}

void class_discs_ibm3D::set_disc_type(int dID, int val)
{
	// set discType for ONE disc:
	discsH[dID].discType = val;
}

void class_discs_ibm3D::set_aspect_ratio(float val)
{
	// set aspect ratio for ALL discs:
	for (int i=0; i<nDiscs; i++) discsH[i].ar = val;
}

int class_discs_ibm3D::get_max_array_size()
{
	// return the maximum array size:
	int maxSize = nBeads;
	if (binsFlag) {
		if (bins.nBins > maxSize) maxSize = bins.nBins;
	}
	return maxSize;
}



// --------------------------------------------------------
// Assign the disc ID to every bead:
// --------------------------------------------------------

void class_discs_ibm3D::assign_discIDs_to_beads()
{
	for (int d=0; d<nDiscs; d++) {
		int istr = discsH[d].indxB0;
		int iend = istr + discsH[d].nBeads;
		for (int i=istr; i<iend; i++) beadsH[i].discID = d;
	}
}



// --------------------------------------------------------
// Duplicate the first rod information to all discs:
// --------------------------------------------------------

void class_discs_ibm3D::duplicate_discs()
{
	if (nDiscs > 1) {
		for (int d=1; d<nDiscs; d++) {
			// skip if disc 0 is different than disc d:
			if (discsH[0].nBeads != discsH[d].nBeads) {
				cout << "duplicate discs error: discs have different nBeads" << endl;
				continue;
			}
			
			// copy bead information:
			for (int i=0; i<discsH[0].nBeads; i++) {
				int ii = i + discsH[d].indxB0;
				beadsH[ii].r = beadsH[i].r;
				beadsH[ii].f = beadsH[i].f;
				beadsH[ii].v = beadsH[i].v;
				beadsH[ii].uf = beadsH[i].uf;
				beadsH[ii].wallContactHist = beadsH[i].wallContactHist;
				beadsH[ii].rm1 = beadsH[i].rm1;
				beadsH[ii].rrel = beadsH[i].rrel;
				beadsH[ii].discID = d;
			}
		}
	}
}



// --------------------------------------------------------
// Set mobility coefficients based on disc aspect ratio:
// --------------------------------------------------------

void class_discs_ibm3D::set_mobility_coefficients(float nu, float ar, float discR)
{
	// ----------------------------------------------			
	// mobility coefficients.  
	// (note: mobility = diffusivity/kT = (friction coeff)^-1)
	// (assume fluid density = 1)
	// (note: sending ar=1 will cause NaN)
	// ----------------------------------------------
	
	float ar2 = ar*ar;                // aspect ratio squared
	float a = discR;                  // disc radius
	float e = sqrtf(1.0f - ar2);      // eccentricity
	float S = (2.0f/e)*std::asin(e);  // integral shape factor = Perrin shape factor
	
	// friction coefficients:
	float KparT = 16.0f*M_PI*nu*a*(1.0f-ar2)/(S + ar2*S - 2*ar);
	float KperT = 32.0f*M_PI*nu*a*(1.0f-ar2)/(2.0f*ar + (1.0f - 3.0f*ar2)*S);
	float KparR = 32.0f*M_PI*nu*a*a*a*(1.0f-ar2)/(2.0f*ar - ar2*S)/3.0f;
	float KperR = 32.0f*M_PI*nu*a*a*a*(1.0f-ar2*ar2)/((1.0f+ar2)*S - 2.0f*ar)/3.0f;
	
	// mobility coefficients (inverse of friction coefficients):	
	float mobParT = 1.0f/KparT;
	float mobPerT = 1.0f/KperT;
	float mobParR = 1.0f/KparR;
	float mobPerR = 1.0f/KperR;
		
	// set mobility coefficients for ALL discs:
	for (int d=0; d<nDiscs; d++) {
		discsH[d].mobParT = mobParT;
		discsH[d].mobPerT = mobPerT;
		discsH[d].mobParR = mobParR;
		discsH[d].mobPerR = mobPerR;
	}
	
	// output the numbers:
	cout << " " << endl;
	cout << "Disc aspect ratio = " << ar << endl;
	cout << "Disc mobility coeff (Translational parallel) = "      << mobParT << endl;
	cout << "Disc mobility coeff (Translational perpendicular) = " << mobPerT << endl;
	cout << "Disc mobility coeff (Rotational parallel) = "         << mobParR << endl;
	cout << "Disc mobility coeff (Rotational perpendicular) = "    << mobPerR << endl;	
}



// --------------------------------------------------------
// randomize cell positions and orientations:
// --------------------------------------------------------

void class_discs_ibm3D::randomize_discs(float sepWall)
{
	// copy bead positions from device to host:
	cudaMemcpy(beadsH, beads, sizeof(beaddisc)*nBeads, cudaMemcpyDeviceToHost);
	
	// assign random position and orientation to each disc:
	const float sepMin = sepWall;
	float3* discCOM = (float3*)malloc(nDiscs*sizeof(float3));
	
	// loop over discs
	for (int d=0; d<nDiscs; d++) {
		
		// initialize values
		discCOM[d] = make_float3(0.0);
		float3 shift = make_float3(0.0,0.0,0.0);
		bool tooClose = true;
		
		while (tooClose) {
			
			// reset tooClose to false
			tooClose = false;
			
			// get random position
			float ran1 = (float)rand()/RAND_MAX;
			float ran2 = (float)rand()/RAND_MAX;
			float ran3 = (float)rand()/RAND_MAX;
			shift.x = ran1*Box.x;
			shift.y = sepWall + ran2*(Box.y-2.0*sepWall);
			shift.z = sepWall + ran3*(Box.z-2.0*sepWall);
						
			// check with other discs
			for (int e=0; e<d; e++) {
				float sep = calc_separation_pbc(shift,discCOM[e]);
                if (sep < sepMin) {
                    tooClose = true;
                    break;
                }
			}					
		}		
		discCOM[d] = shift;		
		// update bead positions and quaternion:
		rotate_and_shift_bead_positions(d,shift.x,shift.y,shift.z);
	}
	
	// copy bead positions from host to device:
	cudaMemcpy(beads, beadsH, sizeof(beaddisc)*nBeads, cudaMemcpyHostToDevice);	
	cudaMemcpy(discs, discsH, sizeof(disc)*nDiscs,     cudaMemcpyHostToDevice);	
}



// --------------------------------------------------------
// randomize rod positions in cylinder:
// --------------------------------------------------------

void class_discs_ibm3D::randomize_discs_cylinder(float sepWall)
{
	
	// copy bead positions from device to host:
	cudaMemcpy(beadsH, beads, sizeof(beaddisc)*nBeads, cudaMemcpyDeviceToHost);
	
	// assign random position and orientation to each disc:
	const float sepMin = sepWall;
	float3* discCOM = (float3*)malloc(nDiscs*sizeof(float3));
				
	// loop over discs
	for (int d=0; d<nDiscs; d++) {
		
		// initialize values
		discCOM[d] = make_float3(0.0);
		float3 shift = make_float3(0.0,0.0,0.0);
		bool tooClose = true;
		
		while (tooClose) {
			
			// reset tooClose to false
			tooClose = false;
				
			// get random position
			float rad = (float)rand()/RAND_MAX*(chRad - sepWall);
			float ang = (float)rand()/RAND_MAX*(2*M_PI);
			shift.x = (float)rand()/RAND_MAX*Box.x;		
			shift.y = rad*cos(ang) + (Box.y-1.0)/2.0;
			shift.z = rad*sin(ang) + (Box.z-1.0)/2.0;
			
			// check with other discs
			for (int e=0; e<d; e++) {
				float sep = calc_separation_pbc(shift,discCOM[e]);
                if (sep < sepMin) {
                    tooClose = true;
                    break;
                }
			}
		}		
			
		discCOM[d] = shift;		
		// update bead positions and quaternion:
		rotate_and_shift_bead_positions(d,shift.x,shift.y,shift.z);
	}	
	
	// copy bead positions from host to device:
	cudaMemcpy(beads, beadsH, sizeof(beaddisc)*nBeads, cudaMemcpyHostToDevice);
	cudaMemcpy(discs, discsH, sizeof(disc)*nDiscs,     cudaMemcpyHostToDevice);	
		
}



// --------------------------------------------------------
// randomize rod positions in duct:
// --------------------------------------------------------

void class_discs_ibm3D::randomize_discs_duct()
{
	
	// copy bead positions from device to host:
	cudaMemcpy(beadsH, beads, sizeof(beaddisc)*nBeads, cudaMemcpyDeviceToHost);
			
	// assign random position and orientation to each rod:
	for (int f=0; f<nDiscs; f++) {
		float3 shift = make_float3(0.0,0.0,0.0);
		// get random position
		float rad = (float)rand()/RAND_MAX*(chRad);
		float ang = (float)rand()/RAND_MAX*(2*M_PI);
		shift.x = (float)rand()/RAND_MAX*Box.x;		
		shift.y = (float)rand()/RAND_MAX*Box.y;
		shift.z = (float)rand()/RAND_MAX*Box.z;		
		rotate_and_shift_bead_positions(f,shift.x,shift.y,shift.z);
	}	
	
	// copy bead positions from host to device:
	cudaMemcpy(beads, beadsH, sizeof(beaddisc)*nBeads, cudaMemcpyHostToDevice);
	cudaMemcpy(discs, discsH, sizeof(disc)*nDiscs,     cudaMemcpyHostToDevice);	
		
}



// --------------------------------------------------------
// randomize rod positions in nozzle:
// --------------------------------------------------------

void class_discs_ibm3D::randomize_discs_nozzle(float lenCylinder, float radInlet, float radOutlet, float Lrod)
{
	// copy bead positions from device to host:
	cudaMemcpy(beadsH, beads, sizeof(beaddisc)*nBeads, cudaMemcpyDeviceToHost);
			
	// assign random position and orientation to each rod:
	for (int f=0; f<nDiscs; f++) {
		
		float3 shift = make_float3(0.0,0.0,0.0);
		
		// get random x-position, but scale the probability of accepting that
		// position by the ratio of the local radius to the inlet radius
		float xpos,chRadLocal,randNum;
		do {
			xpos = 0.55*Lrod + (float)rand()/RAND_MAX*(Box.x-2.0*Lrod);
			if (xpos <= lenCylinder) chRadLocal = radInlet;
			if (xpos >  lenCylinder) chRadLocal = radInlet + (radOutlet - radInlet)*(xpos-lenCylinder)/(Box.x-lenCylinder);
			randNum = (float)rand()/RAND_MAX;
		} while (randNum > chRadLocal/radInlet);
		
		// once accepted, proceed to calculate radial position:	
		float rad = (float)rand()/RAND_MAX*(chRadLocal);
		float ang = (float)rand()/RAND_MAX*(2*M_PI);		
		shift.x = xpos;
		shift.y = rad*cos(ang) + (Box.y-1.0)/2.0;
		shift.z = rad*sin(ang) + (Box.z-1.0)/2.0;
		rotate_and_shift_bead_positions(f,shift.x,shift.y,shift.z);
	}	
	
	// copy bead positions from host to device:
	cudaMemcpy(beads, beadsH, sizeof(beaddisc)*nBeads, cudaMemcpyHostToDevice);	
	cudaMemcpy(discs, discsH, sizeof(disc)*nDiscs,     cudaMemcpyHostToDevice);	
}



// --------------------------------------------------------
// randomize rod positions for nozzle, but place discs
// in the cylinder and backfill zones only:
// --------------------------------------------------------

void class_discs_ibm3D::randomize_discs_nozzle_backfill(float lenCylinder, float radInlet, float radOutlet, float Lrod)
{
	// copy bead positions from device to host:
	cudaMemcpy(beadsH, beads, sizeof(beaddisc)*nBeads, cudaMemcpyDeviceToHost);
	
	// find length of backfill zone such that volume of backfill zone is equal to volume of nozzle zone:
	float Vnozzle = M_PI*(Box.x-lenCylinder)*(radInlet*radInlet + radOutlet*radOutlet + radInlet*radOutlet)/3.0;	
	float lenBackfill = Vnozzle/(M_PI*radInlet*radInlet);
	float offset = Lrod/2.0 + 1.0;   // rod half-length plus a little

	// loop over discs:
	for (int i=0; i<nDiscs; i++) {
		
		// find new rod position & orientation in loading zone or backfill zone:
		while (true) {
			
			// get random position in loading zone:
			float ran1 = (float)rand()/RAND_MAX;
			float ran2 = (float)rand()/RAND_MAX;
			float ran3 = (float)rand()/RAND_MAX;
			float lenRange = lenBackfill + lenCylinder;
			float xpo = -lenBackfill + offset + ran1*(lenRange - 2.0*offset);
			float rad = ran2*(radInlet);
			float ang = ran3*(2*M_PI);
			discsH[i].r.x = xpo;
			discsH[i].r.y = rad*cos(ang) + (Box.y-1.0)/2.0;
			discsH[i].r.z = rad*sin(ang) + (Box.z-1.0)/2.0;
		
			// get random orientation:
			float ran4 = (float)rand()/RAND_MAX;
			float ran5 = (float)rand()/RAND_MAX;
			float phi = ran4*(2*M_PI);  // azimuthal angle
			float psi = 2.0*ran5 - 1.0; // random num from -1 to 1
			discsH[i].p.x = sqrt(1.0-psi*psi)*cos(phi);
			discsH[i].p.y = sqrt(1.0-psi*psi)*sin(phi);
			discsH[i].p.z = psi;
				
			// radial position of rod head:
			float3 head = discsH[i].r + offset*discsH[i].p;		
			float ymid = (Box.y-1.0)/2.0;
			float zmid = (Box.z-1.0)/2.0;
			float hyi = head.y - ymid;  // distance to channel centerline
			float hzi = head.z - zmid;  // "                            "
			float hri = sqrt(hyi*hyi + hzi*hzi);
					
			// radial position of rod tail:
			float3 tail = discsH[i].r - offset*discsH[i].p;			
			float tyi = tail.y - ymid;  // distance to channel centerline
			float tzi = tail.z - zmid;  // "                            "
			float tri = sqrt(tyi*tyi + tzi*tzi);
			
			// if head and tail are inside the radial wall, then accept and exit:
			if (hri < radInlet && tri < radInlet) {
				rotate_and_shift_bead_positions_using_orientation_vector(i);
				break;
			}
			
		}		
	}
		
	// copy bead positions from host to device:
	cudaMemcpy(beads, beadsH, sizeof(beaddisc)*nBeads, cudaMemcpyHostToDevice);	
	cudaMemcpy(discs, discsH, sizeof(disc)*nDiscs,     cudaMemcpyHostToDevice);	
}



// --------------------------------------------------------
// calculate separation distance using PBCs:
// --------------------------------------------------------

float class_discs_ibm3D::calc_separation_pbc(float3 r1, float3 r2)
{
	float3 dr = r1 - r2;
	dr -= roundf(dr/Box)*Box;
	return length(dr);
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_discs_ibm3D::shift_bead_positions(int dID, float xsh, float ysh, float zsh)
{
	int istr = discsH[dID].indxB0;
	int iend = istr + discsH[dID].nBeads;
	for (int i=istr; i<iend; i++) {
		beadsH[i].r.x += xsh;
		beadsH[i].r.y += ysh;
		beadsH[i].r.z += zsh;
		beadsH[i].rm1 = beadsH[i].r;
	}
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_discs_ibm3D::rotate_and_shift_bead_positions(int dID, float xsh, float ysh, float zsh)
{
	// random rotation angles:
	//float a = 2.0*M_PI*((float)rand()/RAND_MAX - 0.5);  // alpha
	//float b = 2.0*M_PI*((float)rand()/RAND_MAX - 0.5);  // beta
	//float g = 2.0*M_PI*((float)rand()/RAND_MAX - 0.5);  // gamma
	
	/*	
	float a = 2.0*M_PI*((float)rand()/RAND_MAX);    // alpha
	float u = 2.0*((float)rand()/RAND_MAX - 1.0f);  	
	float b = acos(u);                              // beta
	float g = 2.0*M_PI*((float)rand()/RAND_MAX);    // gamma
	*/
	
	float a = 0.0;
	float b = M_PI/2.0;
	float g = 0.0;
	
	// rotation tensor:
	tensor R;
	R.xx = cos(a)*cos(b); R.xy = cos(a)*sin(b)*sin(g)-sin(a)*cos(g); R.xz = cos(a)*sin(b)*cos(g)+sin(a)*sin(g);
	R.yx = sin(a)*cos(b); R.yy = sin(a)*sin(b)*sin(g)+cos(a)*cos(g); R.yz = sin(a)*sin(b)*cos(g)-cos(a)*sin(g);
	R.zx = -sin(b);       R.zy = cos(b)*sin(g);                      R.zz = cos(b)*cos(g);
	
	// translation vector:
	float3 rtrans = make_float3(xsh,ysh,zsh);
	
	// update bead positions:
	int istr = discsH[dID].indxB0;
	int iend = istr + discsH[dID].nBeads;
	for (int i=istr; i<iend; i++) {
		// rotate:
		float3 rrot = R*beadsH[i].r;
		// shift:
		beadsH[i].r = rrot + rtrans;
		beadsH[i].rm1 = beadsH[i].r;
	}
	
	// update disc quaternion by a global rotation: q = dq * q;
	quaternion dq;
	dq.set_values(R);
	discsH[dID].q = discsH[dID].q.premultiply(dq);
}



// --------------------------------------------------------
// Shift bead positions by specified amount:
// --------------------------------------------------------

void class_discs_ibm3D::rotate_and_shift_bead_positions(int dID, float xsh, float ysh, float zsh, float a, float b, float g)
{
	// rotation tensor:
	tensor R;
	R.xx = cos(a)*cos(b); R.xy = cos(a)*sin(b)*sin(g)-sin(a)*cos(g); R.xz = cos(a)*sin(b)*cos(g)+sin(a)*sin(g);
	R.yx = sin(a)*cos(b); R.yy = sin(a)*sin(b)*sin(g)+cos(a)*cos(g); R.yz = sin(a)*sin(b)*cos(g)-cos(a)*sin(g);
	R.zx = -sin(b);       R.zy = cos(b)*sin(g);                      R.zz = cos(b)*cos(g);
	
	// translation vector:
	float3 rtrans = make_float3(xsh,ysh,zsh);
	
	// update bead positions:
	int istr = discsH[dID].indxB0;
	int iend = istr + discsH[dID].nBeads;
	for (int i=istr; i<iend; i++) {
		// rotate:
		float3 rrot = R*beadsH[i].r;
		// shift:
		beadsH[i].r = rrot + rtrans;
		beadsH[i].rm1 = beadsH[i].r;
	}
	
	// update disc quaternion by a global rotation: q = dq * q;
	quaternion dq;
	dq.set_values(R);
	discsH[dID].q = discsH[dID].q.premultiply(dq);
}



// --------------------------------------------------------
// Shift IBM start positions by specified amount:
// --------------------------------------------------------

void class_discs_ibm3D::rotate_and_shift_bead_positions_using_orientation_vector(int dID)
{
	// update node positions:
	int istr = discsH[dID].indxB0;
	int iend = istr + discsH[dID].nBeads;
	
	for (int i=istr; i<iend; i++) {
		quaternion q = discsH[dID].q;
		float3 rrel = beadsH[i].rrel;		
		beadsH[i].r = q*rrel + discsH[dID].r;
	}
}



// --------------------------------------------------------
// Calculate wall forces:
// --------------------------------------------------------

void class_discs_ibm3D::compute_wall_forces(int nBlocks, int nThreads)
{
	if (pbcFlag.y==0 && pbcFlag.z==1) wall_forces_ydir(nBlocks,nThreads);
	if (pbcFlag.y==1 && pbcFlag.z==0) wall_forces_zdir(nBlocks,nThreads);
	if (pbcFlag.y==0 && pbcFlag.z==0) wall_forces_ydir_zdir(nBlocks,nThreads);
} 



// --------------------------------------------------------
// Take step forward for discs IBM:
// --------------------------------------------------------

void class_discs_ibm3D::stepIBM_Euler(class_scsp_D3Q19& lbm, int nBlocks, int nThreads) 
{
		
	// ----------------------------------------------------------
	//  here, the Euler algorithm is used to update the 
	//  rod positions 
	// ----------------------------------------------------------
	
	// zero fluid forces:
	lbm.zero_forces(nBlocks,nThreads);
	
	// re-build bin lists for rod beads:
	if (nDiscs > 1) {
		reset_bin_lists(nBlocks,nThreads);
		build_bin_lists(nBlocks,nThreads);
	}
		
	// calculate IBM forces:
	zero_bead_forces(nBlocks,nThreads);
	zero_disc_forces_torques_moments(nBlocks,nThreads);
	lbm.interpolate_gradient_of_velocity_disc(nBlocks,nThreads,beads,nBeads);
	if (nDiscs > 1) nonbonded_bead_interactions(nBlocks,nThreads);
	compute_wall_forces(nBlocks,nThreads);	
	unwrap_bead_coordinates(nBlocks,nThreads);
	sum_disc_forces_torques_moments(nBlocks,nThreads);
			
	// update IBM positions:
	enforce_max_disc_force_torque(nBlocks,nThreads);
	update_disc_position_orientation_fluid(nBlocks,nThreads);
	update_bead_position_discs(nBlocks,nThreads);
	update_bead_velocity_discs(nBlocks,nThreads);
	
	// extrapolate rod force to fluid lattice (this uses bead positions from before update):
	lbm.extrapolate_force_bead_disc(nBlocks,nThreads,beads,discs,nBeads);

}



// --------------------------------------------------------
// Take step forward for discs IBM in a cylinder channel:
// --------------------------------------------------------

void class_discs_ibm3D::stepIBM_Euler_cylindrical_channel(class_scsp_D3Q19& lbm, float chRad, int nBlocks, int nThreads) 
{
		
	// ----------------------------------------------------------
	//  here, the Euler algorithm is used to update the 
	//  rod positions 
	// ----------------------------------------------------------
	
	// zero fluid forces:
	lbm.zero_forces(nBlocks,nThreads);
	
	// re-build bin lists for rod beads:
	if (nDiscs > 1) {
		reset_bin_lists(nBlocks,nThreads);
		build_bin_lists(nBlocks,nThreads);
	}
		
	// calculate IBM forces:
	zero_bead_forces(nBlocks,nThreads);
	zero_disc_forces_torques_moments(nBlocks,nThreads);
	lbm.interpolate_gradient_of_velocity_disc(nBlocks,nThreads,beads,nBeads);
	if (nDiscs > 1) nonbonded_bead_interactions(nBlocks,nThreads);
	compute_wall_forces_cylinder(chRad,nBlocks,nThreads);	
	unwrap_bead_coordinates(nBlocks,nThreads);
	sum_disc_forces_torques_moments(nBlocks,nThreads);
			
	// update IBM positions:
	enforce_max_disc_force_torque(nBlocks,nThreads);
	update_disc_position_orientation_fluid(nBlocks,nThreads);
	update_bead_position_discs(nBlocks,nThreads);
	update_bead_velocity_discs(nBlocks,nThreads);
	
	// extrapolate rod force to fluid lattice (this uses bead positions from before update):
	lbm.extrapolate_force_bead_disc(nBlocks,nThreads,beads,discs,nBeads);

}













// **********************************************************************************************
// Calls to CUDA kernels for main calculations
// **********************************************************************************************












// --------------------------------------------------------
// Call to "init_rand_kernel_IBM3D" kernel:
// --------------------------------------------------------

void class_discs_ibm3D::init_rand_kernel(int nBlocks, int nThreads)
{
	init_rand_kernel_IBM3D
	<<<nBlocks,nThreads>>> (states,1234ULL,nDiscs);
}



// --------------------------------------------------------
// Call to "zero_disc_forces_torques_moments_IBM3D" kernel:
// --------------------------------------------------------

void class_discs_ibm3D::zero_disc_forces_torques_moments(int nBlocks, int nThreads)
{
	zero_disc_forces_torques_moments_IBM3D
	<<<nBlocks,nThreads>>> (discs,nDiscs);
}



// --------------------------------------------------------
// Call to "set_disc_position_orientation_IBM3D" kernel:
// --------------------------------------------------------

void class_discs_ibm3D::set_disc_position_orientation(int nBlocks, int nThreads)
{
	set_disc_position_orientation_IBM3D
	<<<nBlocks,nThreads>>> (beads,discs,nDiscs);
}



// --------------------------------------------------------
// Call to "update_bead_positions_discs_IBM3D" kernel:
// --------------------------------------------------------

void class_discs_ibm3D::update_bead_position_discs(int nBlocks, int nThreads)
{
	update_bead_positions_discs_IBM3D
	<<<nBlocks,nThreads>>> (beads,discs,nBeads);
	
	wrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,nBeads);	
}



// --------------------------------------------------------
// Call to "update_bead_velocity_discs_IBM3D" kernel:
// --------------------------------------------------------

void class_discs_ibm3D::update_bead_velocity_discs(int nBlocks, int nThreads)
{
	update_bead_velocity_discs_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,dt,nBeads);
}



// --------------------------------------------------------
// Call to "update_disc_position_orientation_fluid_IBM3D" kernel:
// --------------------------------------------------------

void class_discs_ibm3D::update_disc_position_orientation_fluid(int nBlocks, int nThreads)
{
	update_disc_position_orientation_fluid_IBM3D
	<<<nBlocks,nThreads>>> (discs,dt,nDiscs);
	
	wrap_disc_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (discs,Box,pbcFlag,nDiscs);	
}



// --------------------------------------------------------
// Call to "update_disc_position_orientation_no_fluid_IBM3D" kernel:
// --------------------------------------------------------

void class_discs_ibm3D::update_disc_position_orientation_no_fluid(int nBlocks, int nThreads)
{
	update_disc_position_orientation_no_fluid_IBM3D
	<<<nBlocks,nThreads>>> (discs,dt,nDiscs);
		
	wrap_disc_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (discs,Box,pbcFlag,nDiscs);	
}



// --------------------------------------------------------
// Call to "assign_velocity_to_backfill_discs_IBM3D" kernel: 
// --------------------------------------------------------

void class_discs_ibm3D::assign_velocity_to_backfill_discs(float backfillVel, int nBlocks, int nThreads)
{
	assign_velocity_to_backfill_discs_IBM3D
	<<<nBlocks,nThreads>>> (discs,backfillVel,nDiscs);	
}



// --------------------------------------------------------
// Call to "move_disc_back_to_inlet_random_IBM3D" kernel: 
// --------------------------------------------------------

void class_discs_ibm3D::move_disc_back_to_inlet_random(float lenInlet, float radInlet, float radOutlet, int nBlocks, int nThreads)
{
	move_disc_back_to_inlet_random_IBM3D
	<<<nBlocks,nThreads>>> (discs,Box,lenInlet,radInlet,radOutlet,nDiscs,states);	
}



// --------------------------------------------------------
// Call to "zero_bead_forces_IBM3D" kernel:
// --------------------------------------------------------

void class_discs_ibm3D::zero_bead_forces(int nBlocks, int nThreads)
{
	zero_bead_forces_IBM3D
	<<<nBlocks,nThreads>>> (beads,nBeads);
}



// --------------------------------------------------------
// Call to "enforce_max_node_force_IBM3D" kernel:
// --------------------------------------------------------

void class_discs_ibm3D::enforce_max_bead_force(int nBlocks, int nThreads)
{
	enforce_max_bead_force_IBM3D
	<<<nBlocks,nThreads>>> (beads,beadFmax,nBeads);
}



// --------------------------------------------------------
// Call to "enforce_max_disc_force_torque_IBM3D" kernel:
// --------------------------------------------------------

void class_discs_ibm3D::enforce_max_disc_force_torque(int nBlocks, int nThreads)
{
	enforce_max_disc_force_torque_IBM3D
	<<<nBlocks,nThreads>>> (discs,discFmax,discTmax,nDiscs);
}



// --------------------------------------------------------
// Call to "sum_disc_forces_torques_moments_IBM3D" kernel:
// --------------------------------------------------------

void class_discs_ibm3D::sum_disc_forces_torques_moments(int nBlocks, int nThreads)
{
	sum_disc_forces_torques_moments_IBM3D
	<<<nBlocks,nThreads>>> (beads,discs,nBeads);
}



// --------------------------------------------------------
// Call to "add_gravity_force_to_beads_IBM3D" kernel:
// --------------------------------------------------------

void class_discs_ibm3D::add_gravity_force_to_beads(float Fzgrav, int nBlocks, int nThreads)
{
	add_gravity_force_to_beads_IBM3D
	<<<nBlocks,nThreads>>> (beads,Fzgrav,nBeads);
}



// --------------------------------------------------------
// Call to "unwrap_bead_coordinates_IBM3D" kernel:
// --------------------------------------------------------

void class_discs_ibm3D::unwrap_bead_coordinates(int nBlocks, int nThreads)
{
	unwrap_bead_coordinates_discs_IBM3D
	<<<nBlocks,nThreads>>> (beads,discs,Box,pbcFlag,nBeads);
}



// --------------------------------------------------------
// Call to "wrap_bead_coordinates_IBM3D" kernel:
// --------------------------------------------------------

void class_discs_ibm3D::wrap_bead_coordinates(int nBlocks, int nThreads)
{
	wrap_bead_coordinates_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,pbcFlag,nBeads);
}



// --------------------------------------------------------
// Call to kernel that builds the binMap array:
// --------------------------------------------------------

void class_discs_ibm3D::build_binMap(int nBlocks, int nThreads)
{
	if (nDiscs > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;	
		build_binMap_for_beads_IBM3D
		<<<nBlocks,nThreads>>> (bins);			
	}	
}



// --------------------------------------------------------
// Call to kernel that resets bin lists:
// --------------------------------------------------------

void class_discs_ibm3D::reset_bin_lists(int nBlocks, int nThreads)
{
	if (nDiscs > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;		
		reset_bin_lists_for_beads_IBM3D
		<<<nBlocks,nThreads>>> (bins);
	}	
}

 

// --------------------------------------------------------
// Call to kernel that builds bin lists:
// --------------------------------------------------------

void class_discs_ibm3D::build_bin_lists(int nBlocks, int nThreads)
{
	if (nDiscs > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;		
		build_bin_lists_for_beads_IBM3D
		<<<nBlocks,nThreads>>> (beads,bins,nBeads);		
	}	
}



// --------------------------------------------------------
// Call to kernel that calculates nonbonded forces:
// --------------------------------------------------------

void class_discs_ibm3D::nonbonded_bead_interactions(int nBlocks, int nThreads)
{
	if (nDiscs > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;								
		nonbonded_bead_interactions_IBM3D
		<<<nBlocks,nThreads>>> (beads,bins,repA,repD,lubforceMax,nBeads,Box,pbcFlag);
	}	
}



// --------------------------------------------------------
// Call to kernel that calculates nonbonded forces:
// --------------------------------------------------------

void class_discs_ibm3D::nonbonded_bead_interactions_with_friction(int nBlocks, int nThreads)
{
	if (nDiscs > 1) {
		if (!binsFlag) cout << "Warning: IBM bin arrays have not been initialized" << endl;								
		nonbonded_bead_interactions_with_friction_IBM3D
		<<<nBlocks,nThreads>>> (beads,bins,repA,repD,lubforceMax,nBeads,Box,pbcFlag);
	}	
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in y-dir:
// --------------------------------------------------------

void class_discs_ibm3D::wall_forces_ydir(int nBlocks, int nThreads)
{
	bead_wall_forces_ydir_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,repA,repD,nBeads);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in z-dir:
// --------------------------------------------------------

void class_discs_ibm3D::wall_forces_zdir(int nBlocks, int nThreads)
{
	bead_wall_forces_zdir_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,repA,repD,nBeads);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in y-dir
// and z-dir:
// --------------------------------------------------------

void class_discs_ibm3D::wall_forces_ydir_zdir(int nBlocks, int nThreads)
{
	bead_wall_forces_ydir_zdir_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,repA,repD,nBeads);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in radial
// direction for cylindrical channel:
// --------------------------------------------------------

void class_discs_ibm3D::compute_wall_forces_cylinder(float chRad, int nBlocks, int nThreads)
{
	bead_wall_forces_cylinder_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,chRad,repWall,repD/2.0,fricWall,nBeads);
}



// --------------------------------------------------------
// Call to kernel that calculates wall forces in radial
// direction for nozzle channel:
// --------------------------------------------------------

void class_discs_ibm3D::compute_wall_forces_nozzle(float lenCylinder, float radInlet, float radOutlet, int nBlocks, int nThreads)
{
	bead_wall_forces_nozzle_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,lenCylinder,radInlet,radOutlet,repWall,repD/2.0,fricWall,lubforceMax,nBeads);
}



// --------------------------------------------------------
// Call to kernel that pushes discs inside a sphere:
// --------------------------------------------------------

void class_discs_ibm3D::push_beads_inside_sphere(float xs, float ys, float zs, float rs, 
                                                int nBlocks, int nThreads)
{
	push_beads_into_sphere_IBM3D
	<<<nBlocks,nThreads>>> (beads,xs,ys,zs,rs,nBeads);
}



// --------------------------------------------------------
// Call to kernel that pushes discs inside a cylinder:
// --------------------------------------------------------

void class_discs_ibm3D::push_discs_inside_cylinder(float chRad, int nBlocks, int nThreads)
{
	push_beads_into_cylinder_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,chRad,repA,repD,nBeads);
}



// --------------------------------------------------------
// Call to kernel that pushes discs inside a duct:
// --------------------------------------------------------

void class_discs_ibm3D::push_discs_inside_duct(int nBlocks, int nThreads)
{
	push_beads_into_duct_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,repA,repD,nBeads);
}



// --------------------------------------------------------
// Call to kernel that pushes discs inside a slit:
// --------------------------------------------------------

void class_discs_ibm3D::push_discs_inside_slit(int nBlocks, int nThreads)
{
	push_beads_into_slit_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,repA,repD,nBeads);
}



// --------------------------------------------------------
// Call to kernel that pushes discs inside a cylinder:
// --------------------------------------------------------

void class_discs_ibm3D::push_discs_inside_nozzle(float lenCylinder, float radInlet, float radOutlet, int nBlocks, int nThreads)
{
	push_beads_into_nozzle_IBM3D
	<<<nBlocks,nThreads>>> (beads,Box,lenCylinder,radInlet,radOutlet,repA,repD,nBeads);
}











// **********************************************************************************************
// Analysis and Geometry calculations done by the host (CPU)
// **********************************************************************************************












// --------------------------------------------------------
// Write IBM output to file:
// --------------------------------------------------------

void class_discs_ibm3D::write_output(std::string tagname, int tagnum)
{
	write_vtk_immersed_boundary_3D_discs(tagname,tagnum,
	nBeads,nBeadsPerDisc,nDiscs,beadsH,discsH);
	
	
	cout << "quaternion 1: " << discsH[0].q.w << " " << discsH[0].q.x << " " << discsH[0].q.y << " " << discsH[0].q.z << endl;
	
	
}



// --------------------------------------------------------
// Unwrap bead coordinates based on difference between bead
// position and the rod's center bead position:
// --------------------------------------------------------

void class_discs_ibm3D::unwrap_bead_coordinates()
{
	for (int i=0; i<nBeads; i++) {
		int f = beadsH[i].discID;
		int j = discsH[f].centerBead;
		float3 rij = beadsH[j].r - beadsH[i].r;
		beadsH[i].r = beadsH[i].r + roundf(rij/Box)*Box*pbcFlag; // PBC's
		beadsH[i].rm1 = beadsH[i].rm1 + roundf(rij/Box)*Box*pbcFlag; // PBC's	
	}	
}



// --------------------------------------------------------
// Output the rod orientation, position, and radial position
// inside cylindrical channel
// --------------------------------------------------------

void class_discs_ibm3D::orientation_in_cylindrical_channel(int step)
{
	
	// -----------------------------------------
	// Define the file location and name:
	// -----------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << "rod_orientation.dat";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------
	// Loop over the capsules  
	// -----------------------------------------
		
	for (int r=0; r<nDiscs; r++) {		
		// radial distance to channel centerline:
		float ymid = (Box.y-1.0)/2.0;
		float zmid = (Box.z-1.0)/2.0;
		float yi = discsH[r].r.y - ymid;
		float zi = discsH[r].r.z - zmid;
		float ri = sqrt(yi*yi + zi*zi);
		// print data:
		outfile << fixed << setprecision(4) << step << "  " << r << "  " << discsH[r].p.x << "  " 
			                                                             << discsH[r].p.y << "  " 
																		 << discsH[r].p.z << "  "
																		 << discsH[r].r.x << "  "
																		 << discsH[r].r.y << "  " 
																		 << discsH[r].r.z << "  "
																		 << ri << endl;		
	}

}





