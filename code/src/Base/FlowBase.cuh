
# ifndef FlOWBASE_H
# define FlOWBASE_H

# include <string>
using namespace std;

// ---------------------------------------------------------------------
// This is the base class for the FlowCUDA project.  All application 
// classes in the FlowCUDA project inherent from this class.
// ---------------------------------------------------------------------

class FlowBase {

public:

   // -------------------------------------------------------------------
   // Define factory method that creates objects of sub-classes:
   // -------------------------------------------------------------------

   static FlowBase* FlowObjectFactory(string specifier);

   // -------------------------------------------------------------------
   // All sub-classes must define the below pure virtual functions:
   // -------------------------------------------------------------------

   virtual void initSystem() = 0;
   virtual void cycleForward(int,int) = 0;

   // -------------------------------------------------------------------
   // Virtual destructor:
   // -------------------------------------------------------------------

   virtual ~FlowBase()
   {
   }
   
};

# endif  // FlOWBASE_H
