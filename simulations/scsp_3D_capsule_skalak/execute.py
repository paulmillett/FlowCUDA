import os
import subprocess as sp

# create subdirectory:
if not os.path.exists('./test'):
    os.makedirs('test')

# create file in subdirectory:
os.chdir('./test')
retval = os.getcwd()
print ("Directory changed successfully %s" % retval)
f = open("input.dat", "w")
os.chdir('../')   # leave subdirectory







#sp.call(["../../code/bin/FlowCUDA"])

