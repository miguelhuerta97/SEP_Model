"""
Paths and other machine related configurations
"""
import os, sys
def normpath(*args):
  return os.path.normpath(os.sep.join(args))

AMPL_FOLDER = normpath('','home', 'miguel', 'ampl_linux')
sys.path.append(AMPL_FOLDER)

KNITRO_PATH = normpath(AMPL_FOLDER, 'knitro')
GUROBI_PATH = normpath(AMPL_FOLDER, 'gurobi')
CPLEX_PATH  = normpath(AMPL_FOLDER, 'cplex')
IPOPT_PATH  = normpath(AMPL_FOLDER, 'ipopt')