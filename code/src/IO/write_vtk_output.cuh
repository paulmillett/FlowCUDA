# ifndef WRITE_VTK_OUTPUT_H
# define WRITE_VTK_OUTPUT_H

# include "../IBM/3D/membrane_data.h"
# include "../D3Q19/mcmp_SC_bb/kernels_mcmp_SC_bb_D3Q19.cuh"
# include <string>


void write_vtk_structured_grid_2D(std::string,int,int,int,int,float*,
                                  float*,float*);
							   
void write_vtk_structured_grid_2D(std::string,int,int,int,int,float*,
                                  float*,float*,float*,float*);

void write_vtk_structured_grid_2D(std::string,int,int,int,int,float*,
                                  float*,float*,float*);
								  
void write_vtk_immersed_boundary_2D(std::string,int,int,float*,float*);

void write_vtk_immersed_boundary_3D(std::string,int,int,int,float3*,int*,int*,int*);

void write_vtk_immersed_boundary_3D(std::string,int,int,int,float3*,triangle*);

void write_vtk_immersed_boundary_3D_cellID(std::string,int,int,int,float3*,triangle*,cell*);

void write_vtk_immersed_boundary_normals_3D(std::string,int,int,int,int,float3*,triangle*,edge*);

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

void write_vtk_unstructured_grid(std::string,int,int,int*,int*,int*);

void write_vtk_particles(std::string,int,particle3D_bb*,int);


# endif  // WRITE_VTK_OUTPUT_H