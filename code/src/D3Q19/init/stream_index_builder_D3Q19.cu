
# include "stream_index_builder_D3Q19.cuh"



// --------------------------------------------------------
// Build the streamIndex[] array, which stores the indices
// of the streaming populations of neighboring voxels:
// Note: this is for "PUSH" streaming!
// --------------------------------------------------------

void stream_index_push_D3Q19(int nVoxels, int* nListDir, int* streamIndex)
{		
	int Q = 19;
	for (int i=0; i<nVoxels; i++) {
		int offst = i*Q;
		streamIndex[offst+0] = offst;                      // dir-0 (no streaming)
		streamIndex[offst+1] = Q*nListDir[offst + 1] + 1;  // dir-1 streamed to nbr 1
		streamIndex[offst+2] = Q*nListDir[offst + 2] + 2;  // dir-2 streamed to nbr 2
		streamIndex[offst+3] = Q*nListDir[offst + 3] + 3;  // dir-3 streamed to nbr 3
		streamIndex[offst+4] = Q*nListDir[offst + 4] + 4;  // dir-4 streamed to nbr 4
		streamIndex[offst+5] = Q*nListDir[offst + 5] + 5;  // dir-5 streamed to nbr 5
		streamIndex[offst+6] = Q*nListDir[offst + 6] + 6;  // dir-6 streamed to nbr 6
		streamIndex[offst+7] = Q*nListDir[offst + 7] + 7;  // dir-7 streamed to nbr 7
		streamIndex[offst+8] = Q*nListDir[offst + 8] + 8;  // dir-8 streamed to nbr 8	
		streamIndex[offst+9] = Q*nListDir[offst + 9] + 9;  // dir-9 streamed to nbr 9	
		streamIndex[offst+10] = Q*nListDir[offst + 10] + 10;  // dir-10 streamed to nbr 10	
		streamIndex[offst+11] = Q*nListDir[offst + 11] + 11;  // dir-11 streamed to nbr 11	
		streamIndex[offst+12] = Q*nListDir[offst + 12] + 12;  // dir-12 streamed to nbr 12	
		streamIndex[offst+13] = Q*nListDir[offst + 13] + 13;  // dir-13 streamed to nbr 13	
		streamIndex[offst+14] = Q*nListDir[offst + 14] + 14;  // dir-14 streamed to nbr 14	
		streamIndex[offst+15] = Q*nListDir[offst + 15] + 15;  // dir-15 streamed to nbr 15	
		streamIndex[offst+16] = Q*nListDir[offst + 16] + 16;  // dir-16 streamed to nbr 16	
		streamIndex[offst+17] = Q*nListDir[offst + 17] + 17;  // dir-17 streamed to nbr 17	
		streamIndex[offst+18] = Q*nListDir[offst + 18] + 18;  // dir-18 streamed to nbr 18			
	}				
}



// --------------------------------------------------------
// Build the streamIndex[] array, which stores the indices
// of the streaming populations of neighboring voxels:
// Note: this is for "PULL" streaming!
// --------------------------------------------------------

void stream_index_pull_D3Q19(int nVoxels, int* nListDir, int* streamIndex)
{	
	int Q = 19;
	for (int i=0; i<nVoxels; i++) {
		int offst = i*Q;
		// dir. 0 (no streaming)
		int nbrindex = nListDir[offst + 0];
		streamIndex[offst+0] = Q*nbrindex + 0;   // dir-0 (no streaming)
		// dir. 1 (streamed from nbr 2)
		nbrindex = nListDir[offst + 2];
		if (nbrindex >= 0) streamIndex[offst+1] = Q*nbrindex + 1;
		if (nbrindex <  0) streamIndex[offst+1] = offst + 2;  // bounceback
		// dir. 2 (streamed from nbr 1)
		nbrindex = nListDir[offst + 1];
		if (nbrindex >= 0) streamIndex[offst+2] = Q*nbrindex + 2;
		if (nbrindex <  0) streamIndex[offst+2] = offst + 1;  // bounceback
		// dir. 3 (streamed from nbr 4)
		nbrindex = nListDir[offst + 4];
		if (nbrindex >= 0) streamIndex[offst+3] = Q*nbrindex + 3;
		if (nbrindex <  0) streamIndex[offst+3] = offst + 4;  // bounceback
		// dir. 4 (streamed from nbr 3)
		nbrindex = nListDir[offst + 3];
		if (nbrindex >= 0) streamIndex[offst+4] = Q*nbrindex + 4;
		if (nbrindex <  0) streamIndex[offst+4] = offst + 3;  // bounceback
		// dir. 5 (streamed from nbr 6)
		nbrindex = nListDir[offst + 6];
		if (nbrindex >= 0) streamIndex[offst+5] = Q*nbrindex + 5;
		if (nbrindex <  0) streamIndex[offst+5] = offst + 6;  // bounceback
		// dir. 6 (streamed from nbr 5)
		nbrindex = nListDir[offst + 5];
		if (nbrindex >= 0) streamIndex[offst+6] = Q*nbrindex + 6;
		if (nbrindex <  0) streamIndex[offst+6] = offst + 5;  // bounceback 		
		// dir. 7 (streamed from nbr 8)
		nbrindex = nListDir[offst + 8];
		if (nbrindex >= 0) streamIndex[offst+7] = Q*nbrindex + 7;
		if (nbrindex <  0) streamIndex[offst+7] = offst + 8;  // bounceback 		
		// dir. 8 (streamed from nbr 7)
		nbrindex = nListDir[offst + 7];
		if (nbrindex >= 0) streamIndex[offst+8] = Q*nbrindex + 8;
		if (nbrindex <  0) streamIndex[offst+8] = offst + 7;  // bounceback 
		// dir. 9 (streamed from nbr 10)
		nbrindex = nListDir[offst + 10];
		if (nbrindex >= 0) streamIndex[offst+9] = Q*nbrindex + 9;
		if (nbrindex <  0) streamIndex[offst+9] = offst + 10;  // bounceback 
		// dir. 10 (streamed from nbr 9)
		nbrindex = nListDir[offst + 9];
		if (nbrindex >= 0) streamIndex[offst+10] = Q*nbrindex + 10;
		if (nbrindex <  0) streamIndex[offst+10] = offst + 9;  // bounceback 
		// dir. 11 (streamed from nbr 12)
		nbrindex = nListDir[offst + 12];
		if (nbrindex >= 0) streamIndex[offst+11] = Q*nbrindex + 11;
		if (nbrindex <  0) streamIndex[offst+11] = offst + 12;  // bounceback 
		// dir. 12 (streamed from nbr 11)
		nbrindex = nListDir[offst + 11];
		if (nbrindex >= 0) streamIndex[offst+12] = Q*nbrindex + 12;
		if (nbrindex <  0) streamIndex[offst+12] = offst + 11;  // bounceback 
		// dir. 13 (streamed from nbr 14)
		nbrindex = nListDir[offst + 14];
		if (nbrindex >= 0) streamIndex[offst+13] = Q*nbrindex + 13;
		if (nbrindex <  0) streamIndex[offst+13] = offst + 14;  // bounceback 
		// dir. 14 (streamed from nbr 13)
		nbrindex = nListDir[offst + 13];
		if (nbrindex >= 0) streamIndex[offst+14] = Q*nbrindex + 14;
		if (nbrindex <  0) streamIndex[offst+14] = offst + 13;  // bounceback 
		// dir. 15 (streamed from nbr 16)
		nbrindex = nListDir[offst + 16];
		if (nbrindex >= 0) streamIndex[offst+15] = Q*nbrindex + 15;
		if (nbrindex <  0) streamIndex[offst+15] = offst + 16;  // bounceback 
		// dir. 16 (streamed from nbr 15)
		nbrindex = nListDir[offst + 15];
		if (nbrindex >= 0) streamIndex[offst+16] = Q*nbrindex + 16;
		if (nbrindex <  0) streamIndex[offst+16] = offst + 15;  // bounceback 
		// dir. 17 (streamed from nbr 18)
		nbrindex = nListDir[offst + 18];
		if (nbrindex >= 0) streamIndex[offst+17] = Q*nbrindex + 17;
		if (nbrindex <  0) streamIndex[offst+17] = offst + 18;  // bounceback 
		// dir. 18 (streamed from nbr 17)
		nbrindex = nListDir[offst + 17];
		if (nbrindex >= 0) streamIndex[offst+18] = Q*nbrindex + 18;
		if (nbrindex <  0) streamIndex[offst+18] = offst + 17;  // bounceback 
	}
}