
# include "class_scsp_active_3phi_D2Q9.cuh"
# include "kernels_scsp_active_D2Q9.cuh"
# include "../../IO/GetPot"
# include <math.h>
# include <iostream>
# include <iomanip>
# include <fstream>
# include <sstream>
using namespace std;  



// --------------------------------------------------------
// Constructor:
// --------------------------------------------------------

class_scsp_active_3phi_D2Q9::class_scsp_active_3phi_D2Q9()
{
	Q = 9;
	GetPot inputParams("input.dat");	
	nVoxels = inputParams("Lattice/nVoxels",0);
	numIolets = inputParams("Lattice/numIolets",0);
	nu = inputParams("LBM/nu",0.1666666);
	sf = inputParams("LBM/sf",1.0);
	fricR = inputParams("LBM/fricR",1.0);
	activity = inputParams("LBM/activity",0.0);
	alpha = inputParams("LBM/alpha",1.0);
	beta = inputParams("LBM/beta",0.0);
	kapp = inputParams("LBM/kapp",1.0);
	kapphi = inputParams("LBM/kapphi",1.0);
	mob = inputParams("LBM/mob",1.0);
	a = inputParams("LBM/a",1.0);
	Nx = inputParams("Lattice/Nx",1);
	Ny = inputParams("Lattice/Ny",1);
	Nz = inputParams("Lattice/Nz",1);	
}



// --------------------------------------------------------
// Destructor:
// --------------------------------------------------------

class_scsp_active_3phi_D2Q9::~class_scsp_active_3phi_D2Q9()
{
		
}



// --------------------------------------------------------
// Allocate arrays:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::allocate()
{
	// allocate array memory (host):
    rH = (float*)malloc(nVoxels*sizeof(float));
	phi1H = (float*)malloc(nVoxels*sizeof(float));
	phi2H = (float*)malloc(nVoxels*sizeof(float));
	phi3H = (float*)malloc(nVoxels*sizeof(float));
	uH = (float2*)malloc(nVoxels*sizeof(float2));
	pH = (float2*)malloc(nVoxels*sizeof(float2));
	hH = (float2*)malloc(nVoxels*sizeof(float2));
	nListH = (int*)malloc(nVoxels*Q*sizeof(int));
	voxelTypeH = (int*)malloc(nVoxels*sizeof(int));
	streamIndexH = (int*)malloc(nVoxels*Q*sizeof(int));	
			
	// allocate array memory (device):
	cudaMalloc((void **) &r, nVoxels*sizeof(float));
	cudaMalloc((void **) &u, nVoxels*sizeof(float2));
	cudaMalloc((void **) &F, nVoxels*sizeof(float2));
	cudaMalloc((void **) &p, nVoxels*sizeof(float2));
	cudaMalloc((void **) &h, nVoxels*sizeof(float2));
	cudaMalloc((void **) &f1, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &f2, nVoxels*Q*sizeof(float));	
	cudaMalloc((void **) &phi1, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &phi2, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &phi3, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &chempot1, nVoxels*Q*sizeof(float));	
	cudaMalloc((void **) &chempot2, nVoxels*Q*sizeof(float));
	cudaMalloc((void **) &chempot3, nVoxels*Q*sizeof(float));	
	cudaMalloc((void **) &stress, nVoxels*sizeof(tensor2D));		
	cudaMalloc((void **) &voxelType, nVoxels*sizeof(int));
	cudaMalloc((void **) &streamIndex, nVoxels*Q*sizeof(int));	
	cudaMalloc((void **) &nList, nVoxels*Q*sizeof(int));	
}



// --------------------------------------------------------
// Deallocate arrays:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::deallocate()
{
	// free array memory (host):
	free(uH);
	free(pH);
	free(hH);
	free(rH);
	free(phi1H);
	free(phi2H);
	free(phi3H);
	free(nListH);
	free(voxelTypeH);
	free(streamIndexH);	
		
	// free array memory (device):
	cudaFree(F);
	cudaFree(u);
	cudaFree(p);
	cudaFree(h);
	cudaFree(r);
	cudaFree(f1);
	cudaFree(f2);
	cudaFree(phi1);
	cudaFree(phi2);
	cudaFree(phi3);
	cudaFree(chempot1);
	cudaFree(chempot2);
	cudaFree(chempot3);
	cudaFree(stress);
	cudaFree(voxelType);
	cudaFree(streamIndex);
	cudaFree(nList);
}



// --------------------------------------------------------
// Copy arrays from host to device:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::memcopy_host_to_device()
{
    cudaMemcpy(r, rH, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(u, uH, sizeof(float2)*nVoxels, cudaMemcpyHostToDevice);	
	cudaMemcpy(p, pH, sizeof(float2)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(phi1, phi1H, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(phi2, phi2H, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(phi3, phi3H, sizeof(float)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(nList, nListH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
	cudaMemcpy(voxelType, voxelTypeH, sizeof(int)*nVoxels, cudaMemcpyHostToDevice);
	cudaMemcpy(streamIndex, streamIndexH, sizeof(int)*nVoxels*Q, cudaMemcpyHostToDevice);
}



// --------------------------------------------------------
// Copy arrays from device to host:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::memcopy_device_to_host()
{
    cudaMemcpy(rH, r, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(uH, u, sizeof(float2)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(pH, p, sizeof(float2)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(hH, h, sizeof(float2)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(phi1H, phi1, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(phi2H, phi2, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
	cudaMemcpy(phi3H, phi3, sizeof(float)*nVoxels, cudaMemcpyDeviceToHost);
}



// --------------------------------------------------------
// Initialize lattice as a "box" with periodic BC's:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::create_lattice_box_periodic()
{
	build_box_lattice_D2Q9(nVoxels,Nx,Ny,voxelTypeH,nListH);
}



// --------------------------------------------------------
// Build the streamIndex[] array for PULL streaming:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::stream_index_pull()
{
	stream_index_pull_D2Q9(nVoxels,nListH,streamIndexH);
}



// --------------------------------------------------------
// Setters for host arrays:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::setU(int i, float val)
{
	uH[i].x = val;
}

void class_scsp_active_3phi_D2Q9::setV(int i, float val)
{
	uH[i].y = val;
}

void class_scsp_active_3phi_D2Q9::setPx(int i, float val)
{
	pH[i].x = val;
}

void class_scsp_active_3phi_D2Q9::setPy(int i, float val)
{
	pH[i].y = val;
}

void class_scsp_active_3phi_D2Q9::setR(int i, float val)
{
	rH[i] = val;
}

void class_scsp_active_3phi_D2Q9::setPhi1(int i, float val)
{
	phi1H[i] = val;
}

void class_scsp_active_3phi_D2Q9::setPhi2(int i, float val)
{
	phi2H[i] = val;
}

void class_scsp_active_3phi_D2Q9::setPhi3(int i, float val)
{
	phi3H[i] = val;
}

void class_scsp_active_3phi_D2Q9::setVoxelType(int i, int val)
{
	voxelTypeH[i] = val;
}



// --------------------------------------------------------
// Getters for host arrays:
// --------------------------------------------------------

float class_scsp_active_3phi_D2Q9::getU(int i)
{
	return uH[i].x;
}

float class_scsp_active_3phi_D2Q9::getV(int i)
{
	return uH[i].y;
}

float class_scsp_active_3phi_D2Q9::getR(int i)
{
	return rH[i];
}

float class_scsp_active_3phi_D2Q9::getPhi1(int i)
{
	return phi1H[i];
}

float class_scsp_active_3phi_D2Q9::getPhi2(int i)
{
	return phi2H[i];
}

float class_scsp_active_3phi_D2Q9::getPhi3(int i)
{
	return phi3H[i];
}



// --------------------------------------------------------
// Call to "scsp_initial_equilibrium_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::initial_equilibrium(int nBlocks, int nThreads)
{
	scsp_active_initial_equilibrium_D2Q9 
	<<<nBlocks,nThreads>>> (f1,r,u,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_zero_forces_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::zero_forces(int nBlocks, int nThreads)
{
	scsp_active_zero_forces_D2Q9 
	<<<nBlocks,nThreads>>> (F,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_stream_collide_save_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::stream_collide_save(int nBlocks, int nThreads)
{
	scsp_active_stream_collide_save_D2Q9 
	<<<nBlocks,nThreads>>> (f1,f2,r,u,streamIndex,voxelType,nu,nVoxels);
	float* temp = f1;
	f1 = f2;
	f2 = temp;
}



// --------------------------------------------------------
// Call to "scsp_stream_collide_save_forcing_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::stream_collide_save_forcing(int nBlocks, int nThreads)
{
	scsp_active_stream_collide_save_forcing_D2Q9 
	<<<nBlocks,nThreads>>> (f1,f2,r,u,F,streamIndex,voxelType,nu,nVoxels);
	float* temp = f1;
	f1 = f2;
	f2 = temp;
}



// --------------------------------------------------------
// Call to "scsp_active_update_orientation_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::scsp_active_update_orientation(int nBlocks, int nThreads)
{
	scsp_active_update_orientation_D2Q9 
	<<<nBlocks,nThreads>>> (u,p,h,nList,sf,fricR,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_active_update_orientation_diffusive_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::scsp_active_update_orientation_diffusive(int nBlocks, int nThreads)
{
	scsp_active_update_orientation_diffusive_D2Q9 
	<<<nBlocks,nThreads>>> (p,h,nList,fricR,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_active_fluid_stress_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::scsp_active_fluid_stress(int nBlocks, int nThreads)
{
	scsp_active_fluid_stress_D2Q9 
	<<<nBlocks,nThreads>>> (p,h,stress,nList,sf,kapp,activity,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_active_fluid_forces_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::scsp_active_fluid_forces(int nBlocks, int nThreads)
{
	scsp_active_fluid_forces_D2Q9 
	<<<nBlocks,nThreads>>> (F,stress,nList,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_active_fluid_molecular_field_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::scsp_active_fluid_molecular_field(int nBlocks, int nThreads)
{
	scsp_active_fluid_molecular_field_D2Q9 
	<<<nBlocks,nThreads>>> (h,p,stress,nList,alpha,kapp,nVoxels);
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
	    printf("Error: %s\n", cudaGetErrorString(err));
}



// --------------------------------------------------------
// Call to "scsp_active_fluid_molecular_field_with_phi_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::scsp_active_fluid_molecular_field_with_phi(int nBlocks, int nThreads)
{
	scsp_active_fluid_molecular_field_with_phi_D2Q9 
	<<<nBlocks,nThreads>>> (phi1,h,p,stress,nList,alpha,kapp,beta,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_active_fluid_chemical_potential_3phi_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::scsp_active_fluid_chemical_potential(int nBlocks, int nThreads)
{
	scsp_active_fluid_chemical_potential_3phi_D2Q9 
	<<<nBlocks,nThreads>>> (phi1,phi2,phi3,chempot1,chempot2,chempot3,p,nList,a,alpha,kapphi,beta,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_active_fluid_capillary_force_3phi_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::scsp_active_fluid_capillary_force(int nBlocks, int nThreads)
{
	scsp_active_fluid_capillary_force_3phi_D2Q9 
	<<<nBlocks,nThreads>>> (phi1,phi2,phi3,chempot1,chempot2,chempot3,F,nList,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_active_fluid_update_phi_3phi_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::scsp_active_fluid_update_phi(int nBlocks, int nThreads)
{
	scsp_active_fluid_update_phi_3phi_D2Q9 
	<<<nBlocks,nThreads>>> (phi1,phi2,phi3,chempot1,chempot2,chempot3,u,nList,mob,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_active_fluid_update_phi_diffusive_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::scsp_active_fluid_update_phi_diffusive(int nBlocks, int nThreads)
{
	scsp_active_fluid_update_phi_diffusive_D2Q9 
	<<<nBlocks,nThreads>>> (phi1,chempot1,u,nList,mob,nVoxels);
}



// --------------------------------------------------------
// Call to "scsp_active_fluid_set_velocity_field_D2Q9" kernel:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::scsp_active_fluid_set_velocity_field(int nBlocks, int nThreads)
{
	scsp_active_fluid_set_velocity_field_D2Q9 
	<<<nBlocks,nThreads>>> (u,p,0.00005,nVoxels);
}







// --------------------------------------------------------
// Wrtie output:
// --------------------------------------------------------

void class_scsp_active_3phi_D2Q9::write_output(std::string tagname, int step, int iskip, int jskip)
{
	
	// -----------------------------------------------
	//	Define the file location and name:
	// -----------------------------------------------
	
	ofstream outfile;
	std::stringstream filenamecombine;
	filenamecombine << "vtkoutput/" << tagname << "_" << step << ".vtk";
	string filename = filenamecombine.str();
	outfile.open(filename.c_str(), ios::out | ios::app);
	
	// -----------------------------------------------
	//	find output dimensions considering
	//  iskip, jskip:
	// -----------------------------------------------
	
	int Nxs = Nx/iskip;
	int Nys = Ny/jskip;
	if (Nx%2 && iskip>1) Nxs++;  // if odd, then add 1
	if (Ny%2 && jskip>1) Nys++;
	
	// -----------------------------------------------
	//	Write the 'vtk' file header:
	// -----------------------------------------------
	
	string d = "   ";
	outfile << "# vtk DataFile Version 3.1" << endl;
	outfile << "VTK file containing grid data" << endl;
	outfile << "ASCII" << endl;
	outfile << " " << endl;
	outfile << "DATASET STRUCTURED_POINTS" << endl;
	outfile << "DIMENSIONS" << d << Nxs << d << Nys << d << Nz << endl;
	outfile << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
	outfile << "SPACING" << d << 1.0*iskip << d << 1.0*jskip << d << 1.0 << endl;
	outfile << " " << endl;
	outfile << "POINT_DATA " << Nxs*Nys*Nz << endl;
	outfile << "SCALARS " << tagname << " float" << endl;
	outfile << "LOOKUP_TABLE default" << endl;
	
	// -----------------------------------------------
	// Write the 'rho' data:
	// NOTE: x-data increases fastest,
	//       then y-data
	// -----------------------------------------------
	
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j+=jskip) {
			for (int i=0; i<Nx; i+=iskip) {
				int ndx = k*Nx*Ny + j*Nx + i;
				outfile << fixed << setprecision(5) << phi2H[ndx] << endl;
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
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j+=jskip) {
			for (int i=0; i<Nx; i+=iskip) {
				int ndx = k*Nx*Ny + j*Nx + i;
				outfile << fixed << setprecision(5) << uH[ndx].x << " "
					                                << uH[ndx].y << " " 
													<< 0.0 << endl;
			}
		}
	}
	
	// -----------------------------------------------				
	// Write the 'orientation' data:
	// NOTE: x-data increases fastest,
	//       then y-data	
	// -----------------------------------------------
	
	outfile << "   " << endl;
	outfile << "VECTORS Orientation float" << endl;		
	for (int k=0; k<Nz; k++) {
		for (int j=0; j<Ny; j+=jskip) {
			for (int i=0; i<Nx; i+=iskip) {
				int ndx = k*Nx*Ny + j*Nx + i;
				outfile << fixed << setprecision(5) << pH[ndx].x << " "
					                                << pH[ndx].y << " " 
													<< 0.0 << endl;
			}
		}
	}
		
	// -----------------------------------------------
	//	Close the file:
	// -----------------------------------------------
	
	outfile.close();
	
}








