
# include "FlowBase.cuh"
# include "../IO/GetPot"
# include <cstdlib>
# include <string>



int main(int argc, char *argv[])
{
	
	// ----------------------------------------------------
	// 'GetPot' object containing input parameters:
	// ----------------------------------------------------
	
	GetPot inputParams("input.dat"); 
		
	// ----------------------------------------------------
	// create output directory:
	// ----------------------------------------------------
	
	std::system("mkdir -p vtkoutput");
 	std::system("exec rm -rf vtkoutput/*.vtk");   
	
	// ----------------------------------------------------
	// create the FlowCUDA object:
	// ----------------------------------------------------
	
	std::string LBApp = inputParams("LBM/App","mcmp2D");
	FlowBase* fc = FlowBase::FlowObjectFactory(LBApp);
	
	// ----------------------------------------------------
	// initialize the system:
	// ----------------------------------------------------
	
	fc->initSystem();
	
	// ----------------------------------------------------
	// run the simulation:
	// ----------------------------------------------------
	
	int nSteps = inputParams("Time/nSteps",0);
	int nOutputs = inputParams("Output/nOutputs",0);
	int nCycles = min(nSteps,nOutputs);
	int stepsPerCycle = nSteps/nCycles;
	
	if (nSteps < nOutputs) {
		stepsPerCycle = 1;
	}
	
	if (nSteps == 0) {
		stepsPerCycle = 0;
	}
		
	for (int cycle=0; cycle < nCycles; cycle++) {
			fc->cycleForward(stepsPerCycle,cycle);
	}	
	
	// ----------------------------------------------------
	// delete object:
	// ----------------------------------------------------
	
	delete fc;
	
}