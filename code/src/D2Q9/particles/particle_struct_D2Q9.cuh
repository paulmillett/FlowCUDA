
# ifndef PARTICLE_STRUCT_D2Q9_H
# define PARTICLE_STRUCT_D2Q9_H

// --------------------------------------------------------
// Struct containing particle data (2D):
// --------------------------------------------------------

struct particle2D {
	float rx,ry;
	float vx,vy;
	float fx,fy;
	float rInner;
	float rOuter;
};

# endif  // PARTICLE_STRUCT_D2Q9_H