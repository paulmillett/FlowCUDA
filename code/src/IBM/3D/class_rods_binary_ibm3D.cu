# include "class_rods_binary_ibm3D.cuh"
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

class_rods_binary_ibm3D::class_rods_binary_ibm3D()
{
	// get some parameters:
	GetPot inputParams("input.dat");
	
	// mesh attributes
	L0 = inputParams("IBM_RODS/L0",0.5);
	nRods1 = inputParams("IBM_RODS/nRods1",1);
	nRods2 = inputParams("IBM_RODS/nRods2",1);
	nBeadsPerRod1 = inputParams("IBM_RODS/nBeadsPerRod1",0);
	nBeadsPerRod2 = inputParams("IBM_RODS/nBeadsPerRod2",0);
	nRods = nRods1 + nRods2;
	nBeads = nBeadsPerRod1*nRods1 + nBeadsPerRod2*nRods2;
	
	// mechanical properties	
	repA = inputParams("IBM_RODS/repA",0.0);
	repD = inputParams("IBM_RODS/repD",0.0);
	lubforceMax = inputParams("IBM_RODS/lubforceMax",0.0);
	repWall = inputParams("IBM_RODS/repWall",0.0);
	fricWall = inputParams("IBM_RODS/fricWall",0.0);
	beadFmax = inputParams("IBM_RODS/beadFmax",1000.0);
	rodFmax = inputParams("IBM_RODS/rodFmax",1000.0);
	rodTmax = inputParams("IBM_RODS/rodTmax",1000.0);
		
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
	if (nRods > 1) binsFlag = true;
	if (binsFlag) {		
		bins.sizeBins = inputParams("IBM_RODS/sizeBins",2.0);
		bins.binMax = inputParams("IBM_RODS/binMax",1);			
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

class_rods_binary_ibm3D::~class_rods_binary_ibm3D()
{
		
}













// **********************************************************************************************
// Initialization Stuff...
// **********************************************************************************************












// --------------------------------------------------------
// Create the first rod:
//
// Override from parent class
//
// --------------------------------------------------------

void class_rods_binary_ibm3D::create_first_rod()
{
	// set up the bead information for first rod of type 1:
	for (int i=0; i<nBeadsPerRod1; i++) {
		beadsH[i].r.x = 0.0 + float(i)*L0;
		beadsH[i].r.y = 0.0;
		beadsH[i].r.z = 0.0;
		beadsH[i].rm1 = beadsH[i].r;
		beadsH[i].f = make_float3(0.0f);
		beadsH[i].rodID = 0;
	}
	
	// set up the bead information for first rod of type 2:
	for (int i=0; i<nBeadsPerRod2; i++) {
		int ii = i + nBeadsPerRod1*nRods1;
		beadsH[ii].r.x = 0.0 + float(i)*L0;
		beadsH[ii].r.y = 0.0;
		beadsH[ii].r.z = 0.0;
		beadsH[ii].rm1 = beadsH[ii].r;
		beadsH[ii].f = make_float3(0.0f);
		beadsH[ii].rodID = 0 + nRods1;
	}
		
	// set up indices for ALL rods:
	for (int f=0; f<nRods; f++) {
		if (f < nRods1) {
			rodsH[f].nBeads = nBeadsPerRod1;
			rodsH[f].indxB0 = f*nBeadsPerRod1;      // start index for beads
			rodsH[f].headBead = f*nBeadsPerRod1;    // head bead (first bead)
			rodsH[f].tailBead = f*nBeadsPerRod1 + nBeadsPerRod1 - 1;  // tail bead (last bead)
			rodsH[f].centerBead = f*nBeadsPerRod1 + nBeadsPerRod1/2;  // center-of-mass, assuming nBeadsPerRod is odd	
		}
		else {
			rodsH[f].nBeads = nBeadsPerRod2;
			rodsH[f].indxB0 = nRods1*nBeadsPerRod1 + (f-nRods1)*nBeadsPerRod2;    // start index for beads
			rodsH[f].headBead = nRods1*nBeadsPerRod1 + (f-nRods1)*nBeadsPerRod2;  // head bead (first bead)
			rodsH[f].tailBead = nRods1*nBeadsPerRod1 + (f-nRods1)*nBeadsPerRod2 + nBeadsPerRod2 - 1;  // tail bead (last bead)
			rodsH[f].centerBead = nRods1*nBeadsPerRod1 + (f-nRods1)*nBeadsPerRod2 + nBeadsPerRod2/2;  // center-of-mass, assuming nBeadsPerRod is odd	
		}
			
	}
}



// --------------------------------------------------------
// Duplicate the first rod information to all rods:
//
// Override from parent class
//
// --------------------------------------------------------

void class_rods_binary_ibm3D::duplicate_rods()
{	
	// Rod population 1:
	if (nRods1 > 1) {
		for (int r=1; r<nRods1; r++) {
			// copy bead information:
			for (int i=0; i<nBeadsPerRod1; i++) {
				int ii = i + rodsH[r].indxB0;
				beadsH[ii].r = beadsH[i].r;
				beadsH[ii].f = beadsH[i].f;
				beadsH[ii].rm1 = beadsH[i].rm1;
				beadsH[ii].rodID = r;
			}
		}
	}
	
	// Rod population 2:
	if (nRods2 > 1) {
		int offsetB = nRods1*nBeadsPerRod1;
		for (int r=1; r<nRods2; r++) {
			// copy bead information:
			for (int i=0; i<nBeadsPerRod2; i++) {
				int ii = i + rodsH[r].indxB0;
				beadsH[ii].r = beadsH[i+offsetB].r;
				beadsH[ii].f = beadsH[i+offsetB].f;
				beadsH[ii].rm1 = beadsH[i+offsetB].rm1;
				beadsH[ii].rodID = r + nRods1;
			}
		}
	}	
}



// --------------------------------------------------------
// Set aspect ratios for rods:
//
// Override from parent class
//
// --------------------------------------------------------

void class_rods_binary_ibm3D::set_aspect_ratios(float ar1, float ar2)
{
	// set aspect ratio for ALL rods:
	for (int r=0; r<nRods; r++) {
		if (r < nRods1) {
			rodsH[r].ar = ar1;
		} else {
			rodsH[r].ar = ar2;
		}
	}
}



// --------------------------------------------------------
// Set mobility coefficients based on rod aspect ratios:
//
// Override from parent class
//
// --------------------------------------------------------

void class_rods_binary_ibm3D::set_mobility_coefficients(float nu, float ar1, float Lrod1, float ar2, float Lrod2)
{
	// ----------------------------------------------			
	// mobility coefficients.  See Luders et al. 
	// J. Chem. Phys. 159:054901 (2023) Eqs. (15-17)
	// (note: mobility = diffusivity/kT)
	// (assume fluid density = 1)
	// ----------------------------------------------
		
	float mobPar1 = (log(ar1) - 0.207 + 0.980/ar1 - 0.133/(ar1*ar1)) / (2.0*M_PI*nu*Lrod1); 
	float mobPer1 = (log(ar1) + 0.839 + 0.185/ar1 + 0.233/(ar1*ar1)) / (4.0*M_PI*nu*Lrod1);
	float mobRot1 = (log(ar1) - 0.662 + 0.917/ar1 - 0.050/(ar1*ar1)) / (M_PI*nu*Lrod1*Lrod1*Lrod1) * 3.0;
	
	float mobPar2 = (log(ar2) - 0.207 + 0.980/ar2 - 0.133/(ar2*ar2)) / (2.0*M_PI*nu*Lrod2); 
	float mobPer2 = (log(ar2) + 0.839 + 0.185/ar2 + 0.233/(ar2*ar2)) / (4.0*M_PI*nu*Lrod2);
	float mobRot2 = (log(ar2) - 0.662 + 0.917/ar2 - 0.050/(ar2*ar2)) / (M_PI*nu*Lrod2*Lrod2*Lrod2) * 3.0;
		
	// set mobility coefficients for ALL rods:
	for (int r=0; r<nRods; r++) {
		if (r < nRods1) {
			rodsH[r].mobPar = mobPar1;
			rodsH[r].mobPer = mobPer1;
			rodsH[r].mobRot = mobRot1;
		} else {
			rodsH[r].mobPar = mobPar2;
			rodsH[r].mobPer = mobPer2;
			rodsH[r].mobRot = mobRot2;
		}
		
	}
	
	// output the numbers:
	cout << " " << endl;
	cout << "Rod #1 aspect ratio = " << ar1 << endl;
	cout << "Rod #1 mobility coeff (parallel) = " << mobPar1 << endl;
	cout << "Rod #1 mobility coeff (perpendicular) = " << mobPer1 << endl;
	cout << "Rod #1 mobility coeff (rotational) = " << mobRot1 << endl;	
	cout << " " << endl;
	cout << "Rod #2 aspect ratio = " << ar2 << endl;
	cout << "Rod #2 mobility coeff (parallel) = " << mobPar2 << endl;
	cout << "Rod #2 mobility coeff (perpendicular) = " << mobPer2 << endl;
	cout << "Rod #2 mobility coeff (rotational) = " << mobRot2 << endl;	
}




