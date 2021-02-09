
# ifndef PARTICLE2D_DIP_H
# define PARTICLE2D_DIP_H

// --------------------------------------------------------
// Struct containing 2D particle data for a diffuse-
// interface particle (dip):
// --------------------------------------------------------

struct particle2D_dip {
	float2 r,v,f;
	float rInner;
	float rOuter;
	float mass;
};

# endif  // PARTICLE2D_DIP_H