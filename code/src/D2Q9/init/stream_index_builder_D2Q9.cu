
# include "stream_index_builder_D2Q9.cuh"



// --------------------------------------------------------
// Build the streamIndex[] array, which stores the indices
// of the streaming populations of neighboring voxels:
// Note: this is for "PUSH" streaming!
// --------------------------------------------------------

void stream_index_push_D2Q9(int nVoxels, int* nListDir, int* streamIndex)
{		
	int Q = 9;
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
	}				
}



// --------------------------------------------------------
// Build the streamIndex[] array, which stores the indices
// of the streaming populations of neighboring voxels:
// Note: this is for "PULL" streaming!
// --------------------------------------------------------

void stream_index_pull_D2Q9(int nVoxels, int* nListDir, int* streamIndex)
{		
	int Q = 9;
	for (int i=0; i<nVoxels; i++) {
		int offst = i*Q;
		// dir. 0 (no streaming)
		int nbrindex = nListDir[offst + 0];
		streamIndex[offst+0] = Q*nbrindex + 0;
		// dir. 1 (streamed from nbr 3)
		nbrindex = nListDir[offst + 3];
		if (nbrindex >= 0) streamIndex[offst+1] = Q*nbrindex + 1;
		if (nbrindex  < 0) streamIndex[offst+1] = offst + 3;		
		// dir. 2 (streamed from nbr 4)
		nbrindex = nListDir[offst + 4];
		if (nbrindex >= 0) streamIndex[offst+2] = Q*nbrindex + 2;
		if (nbrindex  < 0) streamIndex[offst+2] = offst + 4;				
		// dir. 3 (streamed from nbr 1)
		nbrindex = nListDir[offst + 1];
		if (nbrindex >= 0) streamIndex[offst+3] = Q*nbrindex + 3;
		if (nbrindex  < 0) streamIndex[offst+3] = offst + 1;		
		// dir. 4 (streamed from nbr 2)
		nbrindex = nListDir[offst + 2];
		if (nbrindex >= 0) streamIndex[offst+4] = Q*nbrindex + 4;
		if (nbrindex  < 0) streamIndex[offst+4] = offst + 2;	
		// dir. 5 (streamed from nbr 7)
		nbrindex = nListDir[offst + 7];
		if (nbrindex >= 0) streamIndex[offst+5] = Q*nbrindex + 5;
		if (nbrindex  < 0) streamIndex[offst+5] = offst + 7;	
		// dir. 6 (streamed from nbr 8)
		nbrindex = nListDir[offst + 8];
		if (nbrindex >= 0) streamIndex[offst+6] = Q*nbrindex + 6;
		if (nbrindex  < 0) streamIndex[offst+6] = offst + 8;			
		// dir. 7 (streamed from nbr 5)
		nbrindex = nListDir[offst + 5];
		if (nbrindex >= 0) streamIndex[offst+7] = Q*nbrindex + 7;
		if (nbrindex  < 0) streamIndex[offst+7] = offst + 5;
		// dir. 8 (streamed from nbr 6)
		nbrindex = nListDir[offst + 6];
		if (nbrindex >= 0) streamIndex[offst+8] = Q*nbrindex + 8;
		if (nbrindex  < 0) streamIndex[offst+8] = offst + 6;		
	}				
}