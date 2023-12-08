# ifndef WRITE_VTK_OUTPUT_H
# define WRITE_VTK_OUTPUT_H

# include "../IBM/3D/data_structs/membrane_data.h"
# include "../IBM/3D/data_structs/filament_data.h"
# include "../IBM/3D/data_structs/rod_data.h"
# include "../D3Q19/mcmp_SC_bb/kernels_mcmp_SC_bb_D3Q19.cuh"
# include <string>


void write_vtk_structured_grid_2D(std::string,int,int,int,int,float*,
                                  float*,float*);
							   
void write_vtk_structured_grid_2D(std::string,int,int,int,int,float*,
                                  float*,float*,float*,float*);

void write_vtk_structured_grid_2D(std::string,int,int,int,int,float*,
                                  float*,float*,float*);
								  
void write_vtk_immersed_boundary_2D(std::string,int,int,float*,float*);

void write_vtk_immersed_boundary_3D(std::string,int,int,int,node*,int*,int*,int*);

void write_vtk_immersed_boundary_3D(std::string,int,int,int,node*,triangle*);

void write_vtk_immersed_boundary_3D_cellID(std::string,int,int,int,node*,triangle*,cell*);

void write_vtk_immersed_boundary_normals_3D(std::string,int,int,int,int,node*,triangle*,edge*);

void write_vtk_structured_grid(std::string,int,int,int,int,float*,
                               int,int,int,int);

void write_vtk_structured_grid(std::string,int,int,int,int,float*,
                               float*,float*,float*);
							   														  							   
void write_vtk_structured_grid(std::string,int,int,int,int,float*,
                               float*,float*,float*,int,int,int,int);
							   
void write_vtk_structured_grid_slit_scaled(std::string,int,int,int,int,float*,
                                           float*,float*,float*,int,int,int,int,
										   float,float,float);
							   
void write_vtk_structured_grid(std::string,int,int,int,int,float*,
                               float*,float*,float*,float*,int,int,int);
							   
void write_vtk_structured_grid(std::string,int,int,int,int,int*,
							   float*,float*,float*,int,int,int);
															  
void write_vtk_polydata(std::string,int,int,int*,int*,int*,float*,
                        float*,float*,float*);
						
void write_vtk_immersed_boundary_3D_filaments(std::string,int,int,int,
                                              bead*,edgefilam*);

void write_vtk_immersed_boundary_3D_rods(std::string,int,int,beadrod*);

void write_vtk_unstructured_grid(std::string,int,int,int*,int*,int*);

void write_vtk_particles(std::string,int,particle3D_bb*,int);


# endif  // WRITE_VTK_OUTPUT_H