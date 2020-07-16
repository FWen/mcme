function printIntro()


printSeparator('*');
disp('                      Demo program for paper #702');
disp(['       AN EXACT PENALTY METHOD FOR LOCALLY CONVERGENT MAXIMUM CONSENSUS ']);
printSeparator('*');

disp('                                 WARNING!!!!! ');
disp(' - This demo program  uses SeDuMi as the default solver (which is provided together');
disp('with the package). If Gurobi was installed on your system with a proper license,');
disp('the solver variable can be changed to "gurobi". Please note that you also need  ');
disp('to specify the correct path to gurobi by setting the variable gurobiPath in the');
disp('file prepareSolver.m');

disp('- SeDuMi may run SLOWER than Gurobi in general, while Gurobi was used for all'); 
disp('experiments reported in the main paper. Note also that different solvers');
disp('may return slightly different qualitative results.');

disp('- This code was tested on a 64 bit Ubuntu Machine with MATLAB R20016b ');

printSeparator('*');





end