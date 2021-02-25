# ifndef WRITE_VTK_OUTPUT_H
# define WRITE_VTK_OUTPUT_H

# include "../IBM/3D/membrane_data.h"
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

void write_vtk_structured_grid(std::string,int,int,int,int,float*,
                               float*,float*,float*);
							   														  							   
void write_vtk_structured_grid(std::string,int,int,int,int,float*,
                               float*,float*,float*,int,int,int);
							   
void write_vtk_structured_grid(std::string,int,int,int,int,float*,
                               float*,float*,float*,float*,int,int,int);
							   
void write_vtk_structured_grid(std::string,int,int,int,int,int*,
							   float*,float*,float*,int,int,int);
															  
void write_vtk_polydata(std::string,int,int,int*,int*,int*,float*,
                        float*,float*,float*);

void write_vtk_unstructured_grid(std::string,int,int,int*,int*,int*);
							   
# endif  // WRITE_VTK_OUTPUT_H