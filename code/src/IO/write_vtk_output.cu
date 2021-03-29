
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
                                    float3* r, int* v1, int* v2, int* v3)
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
		outfile << fixed << setprecision(3) << r[n].x << "  " << r[n].y << "  " << r[n].z << endl;
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
                                    float3* r, triangle* faces)
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
		outfile << fixed << setprecision(3) << r[n].x << "  " << r[n].y << "  " << r[n].z << endl;
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
		



