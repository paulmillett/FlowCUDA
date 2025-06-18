
# include "write_vtk_output.cuh"
# include <iostream>
# include <math.h>
# include <iomanip>
# include <fstream>
# include <string>
# include <sstream>
# include <stdlib.h>
using namespace std;  



// -----------------------------------------------------------------------------------------
// Write output in a VTK Structured Grid format (note: this is for 2D simulations):
// -----------------------------------------------------------------------------------------

void write_vtk_structured_grid_2D(std::string tagname, int tagnum, int NX, int NY,
	                              int NZ, float* r, float* u, float* v)
{		
	
	// -----------------------------------------------
	//	Define the file location and name:
	// -----------------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------------------
	
	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing grid data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET STRUCTURED_POINTS" << endl;
	outfile << "DIMENSIONS" << d << NX << d << NY << d << NZ << endl;
	outfile << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
	outfile << "SPACING" << d << 1.0 << d << 1.0 << d << 1.0 << endl;
	outfile << " " << endl;
	outfile << "POINT_DATA " << NX*NY*NZ << endl;
	outfile << "SCALARS " << tagname << " float" << endl;
	outfile << "LOOKUP_TABLE default" << endl;
	
	// -----------------------------------------------
	// Write the 'rho' data:
	// NOTE: x-data increases fastest,
	//       then y-data
	// -----------------------------------------------
	
	for (int k=0; k<NZ; k++) {
		for (int j=0; j<NY; j++) {
			for (int i=0; i<NX; i++) {
				int ndx = k*NX*NY + j*NX + i;
				outfile << fixed << setprecision(5) << r[ndx] << endl;
			}
		}
	}	
	
	// -----------------------------------------------				
	// Write the 'velocity' data:
	// NOTE: x-data increases fastest,
	//       then y-data	
	// -----------------------------------------------
	
	outfile << "   " << endl;
	outfile << "VECTORS Velocity float" << endl;		
	for (int k=0; k<NZ; k++) {
		for (int j=0; j<NY; j++) {
			for (int i=0; i<NX; i++) {
				int ndx = k*NX*NY + j*NX + i;
				outfile << fixed << setprecision(5) << u[ndx] << " "
					                                << v[ndx] << " " 
													<< 0.0 << endl;
			}
		}
	}
	
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
	
	outfile.close();
	
}



// -----------------------------------------------------------------------------------------
// Write output in a VTK Structured Grid format (note: this is for 2D simulations):
// -----------------------------------------------------------------------------------------

void write_vtk_structured_grid_2D(std::string tagname, int tagnum, int NX, int NY,
	                              int NZ, float* rA, float* rB, float* rS, float* u,
								  float* v)
{		
	
	// -----------------------------------------------
	//	Define the file location and name:
	// -----------------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------------------
	
	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing grid data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET STRUCTURED_POINTS" << endl;
	outfile << "DIMENSIONS" << d << NX << d << NY << d << NZ << endl;
	outfile << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
	outfile << "SPACING" << d << 1.0 << d << 1.0 << d << 1.0 << endl;
	outfile << " " << endl;
	outfile << "POINT_DATA " << NX*NY*NZ << endl;
	outfile << "SCALARS " << tagname << " float" << endl;
	outfile << "LOOKUP_TABLE default" << endl;
	
	// -----------------------------------------------
	// Write the 'rho' data:
	// NOTE: x-data increases fastest,
	//       then y-data
	// -----------------------------------------------
	
	for (int k=0; k<NZ; k++) {
		for (int j=0; j<NY; j++) {
			for (int i=0; i<NX; i++) {
				int ndx = k*NX*NY + j*NX + i;
				float op = (1.0 - rS[ndx])*(rA[ndx] - rB[ndx]);
				outfile << fixed << setprecision(5) << op << endl;
			}
		}
	}	
	
	// -----------------------------------------------				
	// Write the 'velocity' data:
	// NOTE: x-data increases fastest,
	//       then y-data	
	// -----------------------------------------------
	
	outfile << "   " << endl;
	outfile << "VECTORS Velocity float" << endl;		
	for (int k=0; k<NZ; k++) {
		for (int j=0; j<NY; j++) {
			for (int i=0; i<NX; i++) {
				int ndx = k*NX*NY + j*NX + i;
				outfile << fixed << setprecision(3) << u[ndx] << " "
					                                << v[ndx] << " " 
													<< 0.0 << endl;
			}
		}
	}
	
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
	
	outfile.close();
	
}



// -----------------------------------------------------------------------------------------
// Write output in a VTK Structured Grid format (note: this is for 2D simulations):
// -----------------------------------------------------------------------------------------

void write_vtk_structured_grid_2D(std::string tagname, int tagnum, int NX, int NY,
	                              int NZ, float* rA, float* rB, float* u,
								  float* v)
{		
	
	// -----------------------------------------------
	//	Define the file location and name:
	// -----------------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------------------
	
	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing grid data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET STRUCTURED_POINTS" << endl;
	outfile << "DIMENSIONS" << d << NX << d << NY << d << NZ << endl;
	outfile << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
	outfile << "SPACING" << d << 1.0 << d << 1.0 << d << 1.0 << endl;
	outfile << " " << endl;
	outfile << "POINT_DATA " << NX*NY*NZ << endl;
	outfile << "SCALARS " << tagname << " float" << endl;
	outfile << "LOOKUP_TABLE default" << endl;
	
	// -----------------------------------------------
	// Write the 'rho' data:
	// NOTE: x-data increases fastest,
	//       then y-data
	// -----------------------------------------------
	
	for (int k=0; k<NZ; k++) {
		for (int j=0; j<NY; j++) {
			for (int i=0; i<NX; i++) {
				int ndx = k*NX*NY + j*NX + i;
				float op = rA[ndx] - rB[ndx];
				outfile << fixed << setprecision(5) << op << endl;
			}
		}
	}	
	
	// -----------------------------------------------				
	// Write the 'velocity' data:
	// NOTE: x-data increases fastest,
	//       then y-data	
	// -----------------------------------------------
	
	outfile << "   " << endl;
	outfile << "VECTORS Velocity float" << endl;		
	for (int k=0; k<NZ; k++) {
		for (int j=0; j<NY; j++) {
			for (int i=0; i<NX; i++) {
				int ndx = k*NX*NY + j*NX + i;
				outfile << fixed << setprecision(3) << u[ndx] << " "
					                                << v[ndx] << " " 
													<< 0.0 << endl;
			}
		}
	}
	
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
	
	outfile.close();
	
}



// -------------------------------------------------------------------------
// Write IBM mesh to 'vtk' file:
// -------------------------------------------------------------------------

void write_vtk_immersed_boundary_2D(std::string tagname, int tagnum, int nNodes,
                                    float* x, float* y)
{
	
	// -----------------------------------
	//	Define the file location and name:
	// -----------------------------------

	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);

	// -----------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------

	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing IBM data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET POLYDATA" << endl;			
	
	// -----------------------------------
	//	Write the node positions:
	// -----------------------------------

	outfile << " " << endl;	
	outfile << "POINTS " << nNodes << " float" << endl;
	for (int n=0; n<nNodes; n++) {
		outfile << fixed << setprecision(3) << x[n] << "  " << y[n] << "  " << 0.0 << endl;
	}

	// -----------------------------------
	//	Write lines between neighboring nodes:
	// -----------------------------------

	outfile << " " << endl;	
	outfile << "LINES " << nNodes << " " << 3*nNodes << endl;
	for (int n=0; n<nNodes; n++) {
		int nplus = n+1;
		if (n == nNodes-1) nplus = 0;
		outfile << "2  " << n << "  " << nplus << endl;
	}
	
	// -----------------------------------
	//	Write vertices for the nodes:
	// -----------------------------------

	outfile << " " << endl;	
	outfile << "VERTICES " << nNodes << " " << 2*nNodes << endl;
	for (int n=0; n<nNodes; n++) {
		outfile << "1 " << n << endl;
	}
	
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
	
	outfile.close();
	
}








// -----------------------------------------------------------------------------------------
// Write output in a VTK Structured Grid format (note: this is for 3D simulations):
// Here, only one output variable (scalar field) is written
// -----------------------------------------------------------------------------------------

void write_vtk_structured_grid(std::string tagname, int tagnum, int NX, int NY,
	                           int NZ, float* r, int iskip, int jskip, int kskip, int prec)
{		
	
	// -----------------------------------------------
	//	Define the file location and name:
	// -----------------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------------
	//	find output dimensions considering
	//  iskip, jskip, kskip:
	// -----------------------------------------------
	
	int Nxs = NX/iskip;
	int Nys = NY/jskip;
	int Nzs = NZ/kskip;
	if (NX%2 && iskip>1) Nxs++;  // if odd, then add 1
	if (NY%2 && jskip>1) Nys++;
	if (NZ%2 && kskip>1) Nzs++;	
	
	// -----------------------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------------------
	
	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing grid data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET STRUCTURED_POINTS" << endl;
	outfile << "DIMENSIONS" << d << Nxs << d << Nys << d << Nzs << endl;	
	outfile << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
	outfile << "SPACING" << d << 1.0*iskip << d << 1.0*jskip << d << 1.0*kskip << endl;	
    outfile << " " << endl;
    outfile << "POINT_DATA " << Nxs*Nys*Nzs << endl;
    outfile << "SCALARS " << tagname << " float" << endl;
    outfile << "LOOKUP_TABLE default" << endl;
		
	// -----------------------------------------------
	// Write the 'rho' data:
	// NOTE: x-data increases fastest,
	//       then y-data
	// -----------------------------------------------
	
	for (int k=0; k<NZ; k+=kskip) {
		for (int j=0; j<NY; j+=jskip) {
			for (int i=0; i<NX; i+=iskip) {
				int ndx = k*NX*NY + j*NX + i;
				outfile << fixed << setprecision(prec) << r[ndx] << endl;
			}
		}
	}	
	
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
	
	outfile.close();
	
}









// -----------------------------------------------------------------------------------------
// Write output in a VTK Structured Grid format (note: this is for 3D simulations):
// -----------------------------------------------------------------------------------------

void write_vtk_structured_grid(std::string tagname, int tagnum, int NX, int NY,
	                           int NZ, float* r, float* u, float* v, float* w)
{		
	
	// -----------------------------------------------
	//	Define the file location and name:
	// -----------------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------------------
	
	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing grid data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET STRUCTURED_POINTS" << endl;
	outfile << "DIMENSIONS" << d << NX << d << NY << d << NZ << endl;
	outfile << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
	outfile << "SPACING" << d << 1.0 << d << 1.0 << d << 1.0 << endl;
	outfile << " " << endl;
	outfile << "POINT_DATA " << NX*NY*NZ << endl;
	outfile << "SCALARS " << tagname << " float" << endl;
	outfile << "LOOKUP_TABLE default" << endl;
	
	// -----------------------------------------------
	// Write the 'rho' data:
	// NOTE: x-data increases fastest,
	//       then y-data
	// -----------------------------------------------
	
	for (int k=0; k<NZ; k++) {
		for (int j=0; j<NY; j++) {
			for (int i=0; i<NX; i++) {
				int ndx = k*NX*NY + j*NX + i;
				outfile << fixed << setprecision(3) << r[ndx] << endl;
			}
		}
	}	
	
	// -----------------------------------------------				
	// Write the 'velocity' data:
	// NOTE: x-data increases fastest,
	//       then y-data	
	// -----------------------------------------------
	
	outfile << "   " << endl;
	outfile << "VECTORS Velocity float" << endl;		
	for (int k=0; k<NZ; k++) {
		for (int j=0; j<NY; j++) {
			for (int i=0; i<NX; i++) {
				int ndx = k*NX*NY + j*NX + i;
				outfile << fixed << setprecision(3) << u[ndx] << " "
					                                << v[ndx] << " " 
													<< w[ndx] << endl;
			}
		}
	}
	
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
	
	outfile.close();
	
}



// -----------------------------------------------------------------------------------------
// Write output in a VTK Structured Grid format (note: this is for 3D simulations):
// -----------------------------------------------------------------------------------------

void write_vtk_structured_grid(std::string tagname, int tagnum, int NX, int NY,
	                           int NZ, float* r, float* u, float* v, float* w,
							   int iskip, int jskip, int kskip, int prec)
{		
	
	// -----------------------------------------------
	//	Define the file location and name:
	// -----------------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------------
	//	find output dimensions considering
	//  iskip, jskip, kskip:
	// -----------------------------------------------
	
	int Nxs = NX/iskip;
	int Nys = NY/jskip;
	int Nzs = NZ/kskip;
	if (NX%2 && iskip>1) Nxs++;  // if odd, then add 1
	if (NY%2 && jskip>1) Nys++;
	if (NZ%2 && kskip>1) Nzs++;	
	
	// -----------------------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------------------
	
	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing grid data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET STRUCTURED_POINTS" << endl;
	outfile << "DIMENSIONS" << d << Nxs << d << Nys << d << Nzs << endl;	
	outfile << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
	outfile << "SPACING" << d << 1.0*iskip << d << 1.0*jskip << d << 1.0*kskip << endl;	
    outfile << " " << endl;
    outfile << "POINT_DATA " << Nxs*Nys*Nzs << endl;
    outfile << "SCALARS " << tagname << " float" << endl;
    outfile << "LOOKUP_TABLE default" << endl;
		
	// -----------------------------------------------
	// Write the 'rho' data:
	// NOTE: x-data increases fastest,
	//       then y-data
	// -----------------------------------------------
	
	for (int k=0; k<NZ; k+=kskip) {
		for (int j=0; j<NY; j+=jskip) {
			for (int i=0; i<NX; i+=iskip) {
				int ndx = k*NX*NY + j*NX + i;
				outfile << fixed << setprecision(prec) << r[ndx] << endl;
			}
		}
	}	
	
	// -----------------------------------------------				
	// Write the 'velocity' data:
	// NOTE: x-data increases fastest,
	//       then y-data	
	// -----------------------------------------------
	
	outfile << "   " << endl;
	outfile << "VECTORS Velocity float" << endl;		
	for (int k=0; k<NZ; k+=kskip) {
		for (int j=0; j<NY; j+=jskip) {
			for (int i=0; i<NX; i+=iskip) {
				int ndx = k*NX*NY + j*NX + i;
				outfile << fixed << setprecision(prec) << u[ndx] << " "
					                                   << v[ndx] << " " 
													   << w[ndx] << endl;
			}
		}
	}
	
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
	
	outfile.close();
	
}



// -----------------------------------------------------------------------------------------
// Write output in a VTK Structured Grid format (note: this is for 3D simulations):
// -----------------------------------------------------------------------------------------

void write_vtk_structured_grid_slit_scaled(std::string tagname, int tagnum, int NX, int NY,
	                                       int NZ, float* r, float* u, float* v, float* w,
							               int iskip, int jskip, int kskip, int prec,
										   float umax, float h, float scale)
{		
	
	// -----------------------------------------------
	//	Define the file location and name:
	// -----------------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------------
	//	find output dimensions considering
	//  iskip, jskip, kskip:
	// -----------------------------------------------
	
	int Nxs = NX/iskip;
	int Nys = NY/jskip;
	int Nzs = NZ/kskip;
	if (NX%2 && iskip>1) Nxs++;  // if odd, then add 1
	if (NY%2 && jskip>1) Nys++;
	if (NZ%2 && kskip>1) Nzs++;	
	
	// -----------------------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------------------
	
	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing grid data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET STRUCTURED_POINTS" << endl;
	outfile << "DIMENSIONS" << d << Nxs << d << Nys << d << Nzs << endl;	
	outfile << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
	outfile << "SPACING" << d << 1.0*iskip << d << 1.0*jskip << d << 1.0*kskip << endl;	
    outfile << " " << endl;
    outfile << "POINT_DATA " << Nxs*Nys*Nzs << endl;
    outfile << "SCALARS " << tagname << " float" << endl;
    outfile << "LOOKUP_TABLE default" << endl;
		
	// -----------------------------------------------
	// Write the 'rho' data:
	// NOTE: x-data increases fastest,
	//       then y-data
	// -----------------------------------------------
	
	for (int k=0; k<NZ; k+=kskip) {
		for (int j=0; j<NY; j+=jskip) {
			for (int i=0; i<NX; i+=iskip) {
				int ndx = k*NX*NY + j*NX + i;
				outfile << fixed << setprecision(prec) << r[ndx] << endl;
			}
		}
	}	
	
	// -----------------------------------------------				
	// Write the 'velocity' data:
	// NOTE: x-data increases fastest,
	//       then y-data	
	// -----------------------------------------------
	
	outfile << "   " << endl;
	outfile << "VECTORS Velocity float" << endl;		
	for (int k=0; k<NZ; k+=kskip) {
		float z = float(k) + 0.5 - h;
		float u0 = umax*(1.0 - pow(z/h,2));
		for (int j=0; j<NY; j+=jskip) {
			for (int i=0; i<NX; i+=iskip) {
				int ndx = k*NX*NY + j*NX + i;				
				outfile << fixed << setprecision(prec) << scale*(u[ndx] - u0) << " "
					                                   << scale*(v[ndx]) << " " 
													   << scale*(w[ndx]) << endl;
			}
		}
	}
	
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
	
	outfile.close();
	
}



// -----------------------------------------------------------------------------------------
// Write output in a VTK Structured Grid format (note: this is for 3D simulations):
// -----------------------------------------------------------------------------------------

void write_vtk_structured_grid(std::string tagname, int tagnum, int NX, int NY, int NZ,
                               float* rA, float* rB, float* u, float* v, float* w,
							   int iskip, int jskip, int kskip)
{		
	
	// -----------------------------------------------
	//	Define the file location and name:
	// -----------------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------------------
	
	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing grid data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET STRUCTURED_POINTS" << endl;
	outfile << "DIMENSIONS" << d << NX/iskip << d << NY/jskip << d << NZ/kskip << endl;	
	outfile << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
	outfile << "SPACING" << d << 1.0*iskip << d << 1.0*jskip << d << 1.0*kskip << endl;	
    outfile << " " << endl;
    outfile << "POINT_DATA " << (NX/iskip)*(NY/jskip)*(NZ/kskip) << endl;
    outfile << "SCALARS " << tagname << " float" << endl;
    outfile << "LOOKUP_TABLE default" << endl;
		
	// -----------------------------------------------
	// Write the 'rho' data:
	// NOTE: x-data increases fastest,
	//       then y-data
	// -----------------------------------------------
	
	for (int k=0; k<NZ; k+=kskip) {
		for (int j=0; j<NY; j+=jskip) {
			for (int i=0; i<NX; i+=iskip) {
				int ndx = k*NX*NY + j*NX + i;
				float op = rA[ndx] - rB[ndx];
				outfile << fixed << setprecision(3) << op << endl;
			}
		}
	}	
	
	// -----------------------------------------------				
	// Write the 'velocity' data:
	// NOTE: x-data increases fastest,
	//       then y-data	
	// -----------------------------------------------
	
	outfile << "   " << endl;
	outfile << "VECTORS Velocity float" << endl;		
	for (int k=0; k<NZ; k+=kskip) {
		for (int j=0; j<NY; j+=jskip) {
			for (int i=0; i<NX; i+=iskip) {
				int ndx = k*NX*NY + j*NX + i;
				outfile << fixed << setprecision(3) << u[ndx] << " "
					                                << v[ndx] << " " 
													<< w[ndx] << endl;
			}
		}
	}
	
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
	
	outfile.close();
	
}




// -----------------------------------------------------------------------------------------
// Write output in a VTK Structured Grid format (note: this is for 3D simulations):
// -----------------------------------------------------------------------------------------

void write_vtk_structured_grid(std::string tagname, int tagnum, int NX, int NY,
	                           int NZ, int* iarray, float* u, float* v, float* w,
							   int iskip, int jskip, int kskip)
{		
	
	// -----------------------------------------------
	//	Define the file location and name:
	// -----------------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------------------
	
	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing grid data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET STRUCTURED_POINTS" << endl;
	outfile << "DIMENSIONS" << d << NX/iskip << d << NY/jskip << d << NZ/kskip << endl;	
	outfile << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
	outfile << "SPACING" << d << 1.0*iskip << d << 1.0*jskip << d << 1.0*kskip << endl;	
    outfile << " " << endl;
    outfile << "POINT_DATA " << (NX/iskip)*(NY/jskip)*(NZ/kskip) << endl;
    outfile << "SCALARS " << tagname << " int" << endl;
    outfile << "LOOKUP_TABLE default" << endl;
		
	// -----------------------------------------------
	// Write the integer array data:
	// NOTE: x-data increases fastest,
	//       then y-data
	// -----------------------------------------------
	
	for (int k=0; k<NZ; k+=kskip) {
		for (int j=0; j<NY; j+=jskip) {
			for (int i=0; i<NX; i+=iskip) {
				int ndx = k*NX*NY + j*NX + i;
				outfile << iarray[ndx] << endl;
			}
		}
	}	
	
	// -----------------------------------------------				
	// Write the 'velocity' data:
	// NOTE: x-data increases fastest,
	//       then y-data	
	// -----------------------------------------------
	
	outfile << "   " << endl;
	outfile << "VECTORS Velocity float" << endl;		
	for (int k=0; k<NZ; k+=kskip) {
		for (int j=0; j<NY; j+=jskip) {
			for (int i=0; i<NX; i+=iskip) {
				int ndx = k*NX*NY + j*NX + i;
				outfile << fixed << setprecision(3) << u[ndx] << " "
					                                << v[ndx] << " " 
													<< w[ndx] << endl;
			}
		}
	}
	
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
	
	outfile.close();
	
}



// -----------------------------------------------------------------------------------------
// Write output in a VTK PolyData format:
// -----------------------------------------------------------------------------------------

void write_vtk_polydata(std::string tagname, int tagnum, int nVoxels,
                        int* x, int* y, int* z, float* r,
						float* u, float* v, float* w) 
{	
	
	// -----------------------------------------------
	//	Define the file location and name:
	// -----------------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------------
	//	Write the 'vtk' file header:	
	// -----------------------------------------------
	
	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << std::endl;
	outfile << "VTK file containing particle data" << std::endl;
	outfile << "ASCII" << std::endl;
	outfile << " " << std::endl;
	outfile << "DATASET POLYDATA" << std::endl;
	outfile << " " << std::endl;
	outfile << "POINTS" << d << nVoxels << d << " integer" << std::endl;
	
	// -----------------------------------------------
	// Write the x, y, z positions:
	// -----------------------------------------------
	
	for (int i=0; i<nVoxels; i++) {
	    outfile << fixed << x[i] << d << y[i] << d << z[i] << endl;
	}

	
	// -----------------------------------------------
	// Write the 'rho' data:
	// -----------------------------------------------
		
	outfile << "   " << endl;
	outfile << "POINT_DATA\t" << d << nVoxels << endl;
	outfile << "SCALARS density float\n";
	outfile << "LOOKUP_TABLE default\n";
	for (int i=0; i<nVoxels; i++) {
	    outfile << fixed << setprecision(3) << r[i] << endl;
	}
		
	// -----------------------------------------------				
	// Write the 'velocity' data:	
	// -----------------------------------------------
	
	outfile << "   " << endl;
	outfile << "VECTORS velocity float" << endl;		
	for (int i=0; i<nVoxels; i++) {
		outfile << fixed << setprecision(3) << u[i] << " " << v[i] << " " << w[i] << endl;
	}
	
	// -----------------------------------------------	
	//	Close the file:
	// -----------------------------------------------	
	
	outfile.close();
	
}



// -------------------------------------------------------------------------
// Write IBM mesh to 'vtk' file:
// -------------------------------------------------------------------------

void write_vtk_immersed_boundary_3D(std::string tagname, int tagnum, int nNodes, int nFaces,
                                    node* nodes, int* v1, int* v2, int* v3)
{
	
	// -----------------------------------
	//	Define the file location and name:
	// -----------------------------------

	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);

	// -----------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------

	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing IBM data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET POLYDATA" << endl;			
	
	// -----------------------------------
	//	Write the node positions:
	// -----------------------------------

	outfile << " " << endl;	
	outfile << "POINTS " << nNodes << " float" << endl;
	for (int n=0; n<nNodes; n++) {
		outfile << fixed << setprecision(3) << nodes[n].r.x << "  " << nodes[n].r.y << "  " << nodes[n].r.z << endl;
	}
	
	// -----------------------------------------------
	//	Write the polygon information:
	// -----------------------------------------------
	
	outfile << " " << endl;
	outfile << "POLYGONS " << nFaces << " " << 4*nFaces << endl;
	for (int i=0; i<nFaces; i++) {
		outfile << 3 << " " << v1[i] << " " << v2[i] << " " << v3[i] << endl;
	}
	
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
		
	outfile.close();
	
}



// -------------------------------------------------------------------------
// Write IBM mesh to 'vtk' file:
// -------------------------------------------------------------------------

void write_vtk_immersed_boundary_3D(std::string tagname, int tagnum, int nNodes, int nFaces,
                                    node* nodes, triangle* faces)
{
		
	// -----------------------------------
	//	Define the file location and name:
	// -----------------------------------

	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);

	// -----------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------

	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing IBM data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET POLYDATA" << endl;			
	
	// -----------------------------------
	//	Write the node positions:
	// -----------------------------------

	outfile << " " << endl;	
	outfile << "POINTS " << nNodes << " float" << endl;
	for (int n=0; n<nNodes; n++) {
		outfile << fixed << setprecision(3) << nodes[n].r.x << "  " << nodes[n].r.y << "  " << nodes[n].r.z << endl;
	}
	
	// -----------------------------------------------
	//	Write the polygon information:
	// -----------------------------------------------
	
	outfile << " " << endl;
	outfile << "POLYGONS " << nFaces << " " << 4*nFaces << endl;
	for (int i=0; i<nFaces; i++) {
		outfile << 3 << " " << faces[i].v0 << " " << faces[i].v1 << " " << faces[i].v2 << endl;
	}
	
	// -----------------------------------------------
	//	Write the principal tension for each face:
	// -----------------------------------------------
		
	outfile << " " << endl;
	outfile << "CELL_DATA " << nFaces << endl;
	outfile << "SCALARS " << "tension " << "float" << endl;
	outfile << "LOOKUP_TABLE default" << endl;
	for (int i=0; i<nFaces; i++) {
		float tension = faces[i].T1;
		if (tension != tension) tension = 0.0;  // this is true if NaN
		outfile << tension << endl;
	}
	
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
		
	outfile.close();
		
}



// -------------------------------------------------------------------------
// Write IBM mesh to 'vtk' file:
// -------------------------------------------------------------------------

void write_vtk_immersed_boundary_3D_cellID(std::string tagname, int tagnum, int nNodes, int nFaces,
                                           node* nodes, triangle* faces, cell* cells)
{
		
	// -----------------------------------
	//	Define the file location and name:
	// -----------------------------------

	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);

	// -----------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------

	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing IBM data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET POLYDATA" << endl;			
	
	// -----------------------------------
	//	Write the node positions:
	// -----------------------------------

	outfile << " " << endl;	
	outfile << "POINTS " << nNodes << " float" << endl;
	for (int n=0; n<nNodes; n++) {
		outfile << fixed << setprecision(3) << nodes[n].r.x << "  " << nodes[n].r.y << "  " << nodes[n].r.z << endl;
	}
	
	// -----------------------------------------------
	//	Write the polygon information:
	// -----------------------------------------------
	
	outfile << " " << endl;
	outfile << "POLYGONS " << nFaces << " " << 4*nFaces << endl;
	for (int i=0; i<nFaces; i++) {
		outfile << 3 << " " << faces[i].v0 << " " << faces[i].v1 << " " << faces[i].v2 << endl;
	}
	
	// -----------------------------------------------
	//	Write the principal tension for each face:
	// -----------------------------------------------
		
	outfile << " " << endl;
	outfile << "CELL_DATA " << nFaces << endl;
	outfile << "SCALARS " << "tension " << "float" << endl;
	outfile << "LOOKUP_TABLE default" << endl;
	for (int i=0; i<nFaces; i++) {
		float tension = faces[i].T1;
		if (tension != tension) tension = 0.0;  // this is true if NaN
		outfile << tension << endl;
	}
	
	// -----------------------------------------------
	//	Write the cellID (bool 'intrain') for each face:
	// -----------------------------------------------
		
	outfile << " " << endl;
	//outfile << "CELL_DATA " << nFaces << endl;
	outfile << "SCALARS " << "cellID " << "int" << endl;
	outfile << "LOOKUP_TABLE default" << endl;
	for (int i=0; i<nFaces; i++) {
		int cellID = faces[i].cellID;
		//bool value = cells[cellID].intrain;
		//if (tagnum==0) value = false;   // initial value
		int value = cells[cellID].cellType;
		outfile << value << endl;
	}
	
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
		
	outfile.close();
		
}



// -------------------------------------------------------------------------
// Write IBM mesh to 'vtk' file:
// -------------------------------------------------------------------------

void write_vtk_immersed_boundary_normals_3D(std::string tagname, int tagnum, int nNodes, int nFaces,
                                            int nEdges, node* nodes, triangle* faces, edge* edges)
{
	
	// -----------------------------------
	//	Define the file location and name:
	// -----------------------------------

	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);

	// -----------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------

	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing IBM data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET POLYDATA" << endl;			
	
	// -----------------------------------
	//	Write the node positions:
	// -----------------------------------

	outfile << " " << endl;	
	outfile << "POINTS " << nNodes << " float" << endl;
	for (int n=0; n<nNodes; n++) {
		outfile << fixed << setprecision(3) << nodes[n].r.x << "  " << nodes[n].r.y << "  " << nodes[n].r.z << endl;
	}
	
	// -----------------------------------------------
	//	Write the edge information:
	// -----------------------------------------------
	
	outfile << " " << endl;
	outfile << "LINES " << nEdges << " " << 3*nEdges << endl;
	for (int i=0; i<nEdges; i++) {
		outfile << 2 << " " << edges[i].v0 << " " << edges[i].v1 << endl;
	}
	
	// -----------------------------------------------
	//	Write the edge information:
	// -----------------------------------------------
	
	outfile << " " << endl;
	outfile << "LINES " << nEdges << " " << 3*nEdges << endl;
	for (int i=0; i<nEdges; i++) {
		outfile << 2 << " " << edges[i].v0 << " " << edges[i].v1 << endl;
	}
			
	// -----------------------------------------------
	//	Write the polygon information:
	// -----------------------------------------------
	
	/*
	outfile << " " << endl;
	outfile << "POLYGONS " << nFaces << " " << 4*nFaces << endl;
	for (int i=0; i<nFaces; i++) {
		outfile << 3 << " " << faces[i].v0 << " " << faces[i].v1 << " " << faces[i].v2 << endl;
	}
	*/
			
	// -----------------------------------------------
	//	Write the normal vector information:
	// -----------------------------------------------
	
	/*
	outfile << " " << endl;
	outfile << "CELL_DATA " << nFaces << endl;
	outfile << "NORMALS " << "Face-normals " << "float" << endl;
	for (int i=0; i<nFaces; i++) {
		outfile << faces[i].norm.x << " " << faces[i].norm.y << " " << faces[i].norm.z << endl;
	}
	*/
	
	// -----------------------------------------------
	//	Write the edge angle information:
	// -----------------------------------------------
	
	outfile << " " << endl;
	outfile << "CELL_DATA " << nEdges << endl;
	outfile << "SCALARS " << "edge-angle " << "float " << endl;
	outfile << "LOOKUP_TABLE default" << endl;
	for (int i=0; i<nEdges; i++) {
		outfile << edges[i].theta0 << endl;
	}
	
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
		
	outfile.close();
	
}



// -------------------------------------------------------------------------
// Write IBM mesh to 'vtk' file:
// -------------------------------------------------------------------------

void write_vtk_immersed_boundary_3D_filaments(std::string tagname, int tagnum, int nBeads, int nEdges,
                                              bead* beads, edgefilam* edges)
{
		
	// -----------------------------------
	//	Define the file location and name:
	// -----------------------------------

	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);

	// -----------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------

	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing IBM data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET POLYDATA" << endl;			
	
	// -----------------------------------
	//	Write the bead positions:
	// -----------------------------------

	outfile << " " << endl;	
	outfile << "POINTS " << nBeads << " float" << endl;
	for (int i=0; i<nBeads; i++) {
		outfile << fixed << setprecision(3) << beads[i].r.x << "  " << beads[i].r.y << "  " << beads[i].r.z << endl;
	}
	
	// -----------------------------------------------
	//	Write the line information:
	// -----------------------------------------------
	
	outfile << " " << endl;
	outfile << "LINES " << nEdges << " " << 3*nEdges << endl;
	for (int i=0; i<nEdges; i++) {
		outfile << 2 << " " << edges[i].b0 << " " << edges[i].b1 << endl;
	}
		
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
		
	outfile.close();
		
}



// -------------------------------------------------------------------------
// Write IBM mesh to 'vtk' file:
// -------------------------------------------------------------------------

void write_vtk_immersed_boundary_3D_fibers(std::string tagname, int tagnum, int nBeads, int nEdges,
                                           beadfiber* beads, edgefiber* edges)
{
		
	// -----------------------------------
	//	Define the file location and name:
	// -----------------------------------

	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);

	// -----------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------

	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing IBM data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET POLYDATA" << endl;			
	
	// -----------------------------------
	//	Write the bead positions:
	// -----------------------------------

	outfile << " " << endl;	
	outfile << "POINTS " << nBeads << " float" << endl;
	for (int i=0; i<nBeads; i++) {
		outfile << fixed << setprecision(3) << beads[i].r.x << "  " << beads[i].r.y << "  " << beads[i].r.z << endl;
	}
	
	// -----------------------------------------------
	//	Write the line information:
	// -----------------------------------------------
	
	outfile << " " << endl;
	outfile << "LINES " << nEdges << " " << 3*nEdges << endl;
	for (int i=0; i<nEdges; i++) {
		outfile << 2 << " " << edges[i].b0 << " " << edges[i].b1 << endl;
	}
		
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
		
	outfile.close();
		
}



// -------------------------------------------------------------------------
// Write IBM mesh to 'vtk' file:
// -------------------------------------------------------------------------

void write_vtk_immersed_boundary_3D_rods(std::string tagname, int tagnum, int nBeads, int nBeadsPerRod, beadrod* beads)
{
		
	// -----------------------------------
	//	Define the file location and name:
	// -----------------------------------

	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);

	// -----------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------

	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing IBM data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET POLYDATA" << endl;			
	
	// -----------------------------------
	//	Write the bead positions:
	// -----------------------------------

	outfile << " " << endl;	
	outfile << "POINTS " << nBeads << " float" << endl;
	for (int i=0; i<nBeads; i++) {
		outfile << fixed << setprecision(3) << beads[i].r.x << "  " << beads[i].r.y << "  " << beads[i].r.z << endl;
	}
	
	// -----------------------------------
	//	Write if headBead:
	// -----------------------------------
	
	outfile << " " << endl;
	outfile << "POINT_DATA " << nBeads << endl;
	outfile << "SCALARS " << "Head " << "int" << endl;
	outfile << "LOOKUP_TABLE default" << endl;
	for (int i=0; i<nBeads; i++) {
		int value = 0;
		if (i%nBeadsPerRod==0) value = 1;
		outfile << value << endl;
	}
		
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
		
	outfile.close();
		
}



// -----------------------------------------------------------------------------------------
// Write output in a VTK Unstructured Grid format:
// -----------------------------------------------------------------------------------------

void write_vtk_unstructured_grid(std::string tagname, int tagnum, int nVoxels, int* x,
                                 int* y, int* z)
{
	/*	
	// define the file location and name:
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// write the 'vtk' file header:
	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing unstructured grid data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET UNSTRUCTURED_GRID" << endl;
	
	// write the point data:
	outfile << "POINTS " << nVoxels << " float" << endl;
	for (int i=0; i<nVoxels; i++) {
		
	}
	
	// write the cell data:
	outfile << " " << endl;
	outfile << "CELLS " << nCells << d <<  << " float" << endl;
	
	// write the cell_type data:
	
	// write the velocity data:
	
	// close the file:
	outfile.close();	
	*/
}



// -------------------------------------------------------------------------
// Write VTK file for particles:
// -------------------------------------------------------------------------

void write_vtk_particles(std::string tagname, int tagnum, particle3D_bb* pt, int N)
{

    // -----------------------------------
    //	Define the file location and name:
    // -----------------------------------

    ofstream outfile;
    std::stringstream filenamecombine;
    filenamecombine << "vtkoutput/" << tagname << "_" << tagnum << ".vtk";
    string filename = filenamecombine.str();
    outfile.open(filename.c_str(), ios::out | ios::app);

    // -----------------------------------
    //	Write the 'vtk' file header:
    // -----------------------------------

    string d = "   ";
    outfile << "# vtk DataFile Version 3.1" << endl;
    outfile << "VTK file containing particle data" << endl;
    outfile << "ASCII" << endl;
    outfile << " " << endl;
    outfile << "DATASET POLYDATA" << endl;
    outfile << " " << endl;
    outfile << "POINTS" << d << N << d << " float" << endl;

    // -----------------------------------
    //	Write the position data:
    // -----------------------------------

    for (int i=0; i<N; i++) {
        outfile << fixed << setprecision(3) << pt[i].r.x << d << pt[i].r.y << d << pt[i].r.z << endl;
    }

    // -----------------------------------
    //	write the radius data
    // -----------------------------------

    outfile << "POINT_DATA\t" << d << N << endl;
    outfile << "SCALARS radius float\n";
    outfile << "LOOKUP_TABLE default\n";

    for (int i=0; i<N; i++) {
        outfile << fixed << setprecision(3) << pt[i].rad << endl;
    }

    // -----------------------------------
    //	Close the file:
    // -----------------------------------

    outfile.close();
	
}
		



