import data_structure as ds
from front_end_utilities import load_df_result
import re
import gc
import pandas as pd
from sklearn.cluster import KMeans
import os
import sys
from config import *
from amplpy import AMPL, Environment, DataFrame
import numpy as np
from numpy.linalg import inv
from global_definitions import *
from datetime import timedelta
from abc import ABC, abstractmethod
from errors import InputError, ExperimentError, ProceduralError, NotSupportedYet

# TODO:
#   - Refactor Mpc.run_ncognizant() to utilize sta_opf. -> It is complicated because of pdata_bucle
#   - Mpc methods don't consider b2bs
#   - Refactor time behaviour-> time_map shoud utilize tiniout - tend by default (real_predictor)
#   - Standarize run_...() output: tuple or single dataframe?
#   - Standarize solution columns order. Cluster problems by the solution datastructure requirements
#   - How to pass problem modstrings in general config file -> Design standarized (key,value) pairs.
#   - Incorporate solver params in general config file

# OPTI:
#   - Aliasing inside classes must reference pdata elements, not calculations over pdata elements
#     * Modify data_structure.Adn to contain convenient redundances
#   - Implement StaticPflow and refactor all pdata_bucle hacks
# Paths


THIS_DIR          = os.path.dirname(__file__)
SIMULATE_NCOGNIZANT_MODFILE = os.path.join(THIS_DIR, 'ampl_models/simulate_ncognizant.mod')
NCOGNIZANT_MODFILE      = os.path.join(THIS_DIR, 'ampl_models/ncognizant_det.mod')
OPF_MODFILE         = os.path.join(THIS_DIR, 'ampl_models/opf.mod')
OPF_REG_DISTFLOW_MODFILE  = os.path.join(THIS_DIR, 'ampl_models/opf_distflow_regularized.mod')
PROJ_MODFILE        = os.path.join(THIS_DIR, 'ampl_models/proj_mindistance.mod')
PROPOSED_MODFILE      = os.path.join(THIS_DIR, 'ampl_models/proposed.mod')
NCMCCORMICK_MODFILE     = os.path.join(THIS_DIR, 'ampl_models/nc_mccormick.mod')
ROB_MODFILE         = os.path.join(THIS_DIR, 'ampl_models/nc_mc_robust.mod')
PFLOW_BIM_MODFILE       = os.path.join(THIS_DIR, 'ampl_models/pflow_bim.mod')
OPF_STO_MODFILE       = os.path.join(THIS_DIR, 'ampl_models/opf_lindistflow.mod')
OPF_AC_MODFILE        = os.path.join(THIS_DIR, 'ampl_models/opf_ac.mod')
MODFILE_TOY_BILINEAR    = os.path.join(THIS_DIR, 'ampl_models/toy_stochastic_robust.mod')
MODFILE_TOY_MCCORMICK     = os.path.join(THIS_DIR, 'ampl_models/toy_mccormick.mod')

## No funciona la normalización de la ruta
# SIMULATE_NCOGNIZANT_MODFILE = normpath(THIS_DIR, "ampl_models", "simulate_ncognizant.mod")
# NCOGNIZANT_MODFILE      = normpath(THIS_DIR, "ampl_models", "ncognizant_det.mod")
# OPF_MODFILE         = normpath(THIS_DIR, "ampl_models", "opf.mod")
# OPF_REG_DISTFLOW_MODFILE  = normpath(THIS_DIR, "ampl_models", "opf_distflow_regularized.mod")
# PROJ_MODFILE        = normpath(THIS_DIR, "ampl_models", "proj_mindistance.mod")
# PROPOSED_MODFILE      = normpath(THIS_DIR, "ampl_models", "proposed.mod")
# NCMCCORMICK_MODFILE     = normpath(THIS_DIR, "ampl_models", "nc_mccormick.mod")
# ROB_MODFILE         = normpath(THIS_DIR, "ampl_models", "nc_mc_robust.mod")
# PFLOW_BIM_MODFILE       = normpath(THIS_DIR, "ampl_models", "pflow_bim.mod")
# OPF_STO_MODFILE       = normpath(THIS_DIR, "ampl_models", "opf_lindistflow.mod")
# OPF_AC_MODFILE        = normpath(THIS_DIR, "ampl_models", "opf_ac.mod")
# MODFILE_TOY_BILINEAR    = normpath(THIS_DIR, "ampl_models", "toy_stochastic_robust.mod")
# MODFILE_TOY_MCCORMICK     = normpath(THIS_DIR, "ampl_models", "toy_mccormick.mod")


LOCAL_BILINEAR_MODSTRING = (
  'DROOP_B2BS: FOBJ_LASSO_dgpq, FOBJ_UB,V0LIMS,VUB,VLB,PBALANCE,QBALANCE,'
  'FLUX_RELAXED,FLUX_EQUALITY,FLUX_UB1,FLUX_UB2,PV_SNOM,PV_OID,'
  'DROOP_LOCAL_LASSO_dgq,DROOP_LOCAL_LASSO_dgp,B2B_SNOM_F,B2B_SNOM_T,B2B_PLINK,'
  'DROOP_LOCAL_b2bfp,DROOP_LOCAL_b2bfq,DROOP_LOCAL_b2btq,ep,em,v,l,P,Q,nse,dgp,dgq,'
  'b2btp,b2btq,b2bfp,b2bfq,K1v_p_b2bfp,K1v_m_b2bfp,K1v_p_b2bfq,K1v_m_b2bfq,'
  'K1v_p_b2btq,K1v_m_b2btq,K1Pmax_p_dgq,K1v_p_dgq,K1Pmax_m_dgq,K1v_m_dgq,'
  'K1Pmax_p_dgp,K1v_p_dgp,K1Pmax_m_dgp,K1v_m_dgp'
)
LOCAL_BILINEAR_MODSTRING_V2 = (
  'DROOP_B2BS: FOBJ_LASSO_dgpq, V0LIMS,VUB,VLB,PBALANCE,QBALANCE,'
  'FLUX_RELAXED,FLUX_EQUALITY,FLUX_UB1,FLUX_UB2,PV_SNOM,PV_OID,'
  'DROOP_LOCAL_LASSO_dgq,DROOP_LOCAL_LASSO_dgp,B2B_SNOM_F,B2B_SNOM_T,B2B_PLINK,'
  'DROOP_LOCAL_b2bfp,DROOP_LOCAL_b2bfq,DROOP_LOCAL_b2btq,ep,em,v,l,P,Q,nse,dgp,dgq,'
  'b2btp,b2btq,b2bfp,b2bfq,K1v_p_b2bfp,K1v_m_b2bfp,K1v_p_b2bfq,K1v_m_b2bfq,'
  'K1v_p_b2btq,K1v_m_b2btq,K1v_p_dgq,K1v_m_dgq,'
  'K1v_p_dgp,K1v_m_dgp,K1Pmax_p_dgq,K1Pmax_m_dgq,K1Pmax_p_dgp,K1Pmax_m_dgp'
)
LOCAL_BILINEAR_MODSTRING_V2_1 = (
  'DROOP_B2BS: FOBJ_LASSO_dgpq, V0LIMS,VUB,VLB,PBALANCE,QBALANCE,'
  'FLUX_RELAXED,FLUX_EQUALITY,FLUX_UB1,FLUX_UB2,PV_SNOM,PV_OID,'
  'DROOP_LOCAL_LASSO_dgq,DROOP_LOCAL_LASSO_dgp,B2B_SNOM_F,B2B_SNOM_T,B2B_PLINK,'
  'DROOP_LOCAL_b2bfp,DROOP_LOCAL_b2bfq,DROOP_LOCAL_b2btq,ep,em,v,l,P,Q,nse,dgp,dgq,'
  'b2btp,b2btq,b2bfp,b2bfq,K1v_p_b2bfp,K1v_m_b2bfp,K1v_p_b2bfq,K1v_m_b2bfq,'
  'K1v_p_b2btq,K1v_m_b2btq,K1v_p_dgq,K1v_m_dgq,'
  'K1v_p_dgp,K1v_m_dgp,K1Pmax_p_dgq,K1Pmax_m_dgq,K1Pmax_p_dgp,K1Pmax_m_dgp,K0v_p_dgq,'
  'K0v_m_dgq,K0v_p_dgp,K0v_m_dgp;'
)

OPF_SOCP_MODSTRING = (
  'FOBJ,V0LIMS,VUB,VLB,PBALANCE,QBALANCE,FLUX_RELAXED,FLUX_EQUALITY,'
  'FLUX_UB1,FLUX_UB2,PV_SNOM,PV_PMAX,B2B_SNOM_F,B2B_SNOM_T,B2B_PLINK,v,l,P,Q,'
  'nse,dgp,dgq,b2btp,b2btq,b2bfp,b2bfq,ep,em'
)

OPF_LINDISTFLOW_MODSTRING = (
  'FOBJ_LINDISTFLOW,V0LIMS,VUB,VLB,PBALANCE,QBALANCE,FLUX_LINDISTFLOW,FLUX_UB_LINDISTFLOW,'
  'PV_SNOM,PV_PMAX,B2B_SNOM_F,B2B_SNOM_T,B2B_PLINK,ep,em,v,P,Q,nse,dgp,dgq,b2btp,b2btq,b2bfp,'
  'b2bfq'
)

OPF_DISTFLOW = (
  'FOBJ,V0LIMS,VUB,VLB,PBALANCE,QBALANCE,FLUX_NON_RELAXED,FLUX_EQUALITY,'
  'FLUX_UB1,FLUX_UB2,PV_SNOM,PV_PMAX,B2B_SNOM_F,B2B_SNOM_T,B2B_PLINK,v,l,P,Q,'
  'nse,dgp,dgq,b2btp,b2btq,b2bfp,b2bfq,ep,em'
)

opf_modstrings = {
  'lindistflow': OPF_LINDISTFLOW_MODSTRING,
  'socp': OPF_SOCP_MODSTRING,
  'distflow': OPF_DISTFLOW
}
OPF_HOTSTART_MODEL = 'socp'

opf_mod_fobj_names = {}
for kk, vv in opf_modstrings.items():
  assert vv.split(',')[0].startswith('FOBJ')
  opf_mod_fobj_names[kk] = vv.split(',')[0]

ROB_NEUMAN_MODSTRING = (
  'FOBJ,INFINITE_NORM_UB,INFINITE_NORM_LB,'
  'STABILITY,NEUMAN_PFLOW,epi_aux,eps,dv_pos,dv_neg,gp,gq')  # VOLTAGE_LB,VOLTAGE_UB,GQ_LB,GP_LB,
ROB_MCORMICK_MODSTRING = (
  'FOBJ,INFINITE_NORM_UB,INFINITE_NORM_LB,VOLTAGE_LB,VOLTAGE_UB,GP_LB,GQ_LB,STABILITY,'
  'MCCORMICK_PFLOW,PROPOSED_CONTROL_P,PROPOSED_CONTROL_Q,MCCORMICK_P_POS0,MCCORMICK_P_POS1,'
  'MCCORMICK_P_POS2,MCCORMICK_P_POS3,MCCORMICK_P_NEG0,MCCORMICK_P_NEG1,MCCORMICK_P_NEG2,'
  'MCCORMICK_P_NEG3,MCCORMICK_Q_POS0,MCCORMICK_Q_POS1,MCCORMICK_Q_POS2,MCCORMICK_Q_POS3,'
  'MCCORMICK_Q_NEG0,MCCORMICK_Q_NEG1,MCCORMICK_Q_NEG2,MCCORMICK_Q_NEG3,epi_aux,eps,dv_pos,'
  'dv_neg,k_p_pos,k_p_neg,k_q_pos,k_q_neg,gp,gq,dup,duq'
)
ROB_MCORMICK_MODSTRING_TESTING = (
  'FOBJ,INFINITE_NORM_UB,INFINITE_NORM_LB,VOLTAGE_LB,VOLTAGE_UB'
  'epi_aux,eps,dv_pos,dv_neg,dup,duq,MCCORMICK_PFLOW'
)
ROB_PROPOSED_MODSTRING = (
  'PROPOSED_FOBJ,PROPOSED_EPIGRAPH,VOLTAGE_LB2,VOLTAGE_UB2,GP_LB,GQ_LB,STABILITY,'
  'MCCORMICK_PFLOW,PROPOSED_CONTROL_P,PROPOSED_CONTROL_Q,MCCORMICK_P_POS0,MCCORMICK_P_POS1,'
  'MCCORMICK_P_POS2,MCCORMICK_P_POS3,MCCORMICK_P_NEG0,MCCORMICK_P_NEG1,MCCORMICK_P_NEG2,'
  'MCCORMICK_P_NEG3,MCCORMICK_Q_POS0,MCCORMICK_Q_POS1,MCCORMICK_Q_POS2,MCCORMICK_Q_POS3,'
  'MCCORMICK_Q_NEG0,MCCORMICK_Q_NEG1,MCCORMICK_Q_NEG2,MCCORMICK_Q_NEG3,epi_aux,eps,dv_pos,'
  'dv_neg,k_p_pos,k_p_neg,k_q_pos,k_q_neg,gp,gq,DL_pp_pos,DL_pq_pos,DL_qp_pos,DL_qq_pos,'
  'DL_pp_neg,DL_pq_neg,DL_qp_neg,DL_qq_neg,DP_p_pos,DP_q_pos,DP_p_neg,DP_q_neg,dup,duq,e_pos,'
  'e_neg'
)
ROB_BILINEAR_MODSTRING = (
  'FOBJ,INFINITE_NORM_UB,INFINITE_NORM_LB,VOLTAGE_LB,VOLTAGE_UB,GP_LB,GQ_LB,STABILITY,'
  'BILINEAR_PFLOW,epi_aux,eps,dv_pos,dv_neg,k_p_pos,k_p_neg,k_q_pos,k_q_neg,gp,gq')

MODSTRING_TOY_STO_DROOPPOLY_BILINEAR = (
  'FOBJ_STOCHASTIC,VOLTAGE_LB2,VOLTAGE_UB2,LINEAR_PFLOW,STABILITY,'
  'PROPOSED_CONTROL_P_BILINEAR,PROPOSED_CONTROL_Q_BILINEAR,e_pos,'
  'e_neg,eps,dv,dup,duq,gp,gq,DL_pp_pos,DL_pq_pos,DL_qp_pos,DL_qq_pos,'
  'DL_pp_neg,DL_pq_neg,DL_qp_neg,DL_qq_neg,DP_p_pos,DP_q_pos,DP_p_neg,'
  'DP_q_neg,PROPOSED_PMAX,PROPOSED_SMAX,PROPOSED_PMIN'
)
MODSTRING_TOY_ROB_DROOPPOLY_MCCORMICK = (
  'FOBJ_ROBUST,EPIGRAPH_UB,VOLTAGE_LB2,VOLTAGE_UB2,LINEAR_PFLOW,STABILITY,'
  'PROPOSED_CONTROL_P_MCCORMICK,PROPOSED_CONTROL_Q_MCCORMICK,PROPOSED_SMAX,PROPOSED_PMAX,' 
  'PROPOSED_PMIN,MCCORMICK_P_POS0,MCCORMICK_P_POS1,MCCORMICK_P_POS2,MCCORMICK_P_POS3,'
  'MCCORMICK_P_NEG0,MCCORMICK_P_NEG1,MCCORMICK_P_NEG2,MCCORMICK_P_NEG3,MCCORMICK_Q_POS0,'
  'MCCORMICK_Q_POS1,MCCORMICK_Q_POS2,MCCORMICK_Q_POS3,MCCORMICK_Q_NEG0,MCCORMICK_Q_NEG1,'
  'MCCORMICK_Q_NEG2,MCCORMICK_Q_NEG3,epi_ub,e_pos,e_neg,eps,dv_pos,dv_neg,k_p_pos,k_p_neg,'
  'k_q_pos,k_q_neg,dup,duq,gp,gq,DL_pp_pos,DL_pq_pos,DL_qp_pos,DL_qq_pos,DL_pp_neg,DL_pq_neg,'
  'DL_qp_neg,DL_qq_neg,DP_p_pos,DP_q_pos,DP_p_neg,DP_q_neg,VOLTAGE_LB,VOLTAGE_UB'
)
MODSTRING_TOY_STO_DROOPPOLY_MCCORMICK = (
  'FOBJ_STOCHASTIC,EPIGRAPH_UB,VOLTAGE_LB2,VOLTAGE_UB2,LINEAR_PFLOW,STABILITY,'
  'PROPOSED_CONTROL_P_MCCORMICK,PROPOSED_CONTROL_Q_MCCORMICK,PROPOSED_SMAX,PROPOSED_PMAX,'
  'PROPOSED_PMIN,MCCORMICK_P_POS0,MCCORMICK_P_POS1,MCCORMICK_P_POS2,MCCORMICK_P_POS3,'
  'MCCORMICK_P_NEG0,MCCORMICK_P_NEG1,MCCORMICK_P_NEG2,MCCORMICK_P_NEG3,MCCORMICK_Q_POS0,'
  'MCCORMICK_Q_POS1,MCCORMICK_Q_POS2,MCCORMICK_Q_POS3,MCCORMICK_Q_NEG0,MCCORMICK_Q_NEG1,'
  'MCCORMICK_Q_NEG2,MCCORMICK_Q_NEG3,epi_ub,e_pos,e_neg,eps,dv_pos,dv_neg,k_p_pos,k_p_neg,'
  'k_q_pos,k_q_neg,dup,duq,gp,gq,DL_pp_pos,DL_pq_pos,DL_qp_pos,DL_qq_pos,DL_pp_neg,DL_pq_neg,'
  'DL_qp_neg,DL_qq_neg,DP_p_pos,DP_q_pos,DP_p_neg,DP_q_neg,VOLTAGE_LB,VOLTAGE_UB'
)
MODSTRING_TOY_ROB_DROOP_MCCORMICK = (
  'FOBJ_ROBUST,EPIGRAPH_UB,VOLTAGE_LB2,VOLTAGE_UB2,LINEAR_PFLOW,STABILITY,'
  'PROPOSED_CONTROL_P_MCCORMICK,PROPOSED_CONTROL_Q_MCCORMICK,PROPOSED_SMAX,PROPOSED_PMAX,'
  'PROPOSED_PMIN,MCCORMICK_P_POS0,MCCORMICK_P_POS1,MCCORMICK_P_POS2,MCCORMICK_P_POS3,'
  'MCCORMICK_P_NEG0,MCCORMICK_P_NEG1,MCCORMICK_P_NEG2,MCCORMICK_P_NEG3,MCCORMICK_Q_POS0,'
  'MCCORMICK_Q_POS1,MCCORMICK_Q_POS2,MCCORMICK_Q_POS3,MCCORMICK_Q_NEG0,MCCORMICK_Q_NEG1,'
  'MCCORMICK_Q_NEG2,MCCORMICK_Q_NEG3,epi_ub,e_pos,e_neg,eps,dv_pos,dv_neg,k_p_pos,k_p_neg,'
  'k_q_pos,k_q_neg,dup,duq,gp,gq,VOLTAGE_LB,VOLTAGE_UB'
)
MODSTRING_TOY_STO_DROOP_MCCORMICK = (
  'FOBJ_STOCHASTIC,EPIGRAPH_UB,VOLTAGE_LB2,VOLTAGE_UB2,LINEAR_PFLOW,STABILITY,'
  'PROPOSED_CONTROL_P_MCCORMICK,PROPOSED_CONTROL_Q_MCCORMICK,PROPOSED_SMAX,PROPOSED_PMAX,'
  'PROPOSED_PMIN,MCCORMICK_P_POS0,MCCORMICK_P_POS1,MCCORMICK_P_POS2,MCCORMICK_P_POS3,'
  'MCCORMICK_P_NEG0,MCCORMICK_P_NEG1,MCCORMICK_P_NEG2,MCCORMICK_P_NEG3,MCCORMICK_Q_POS0,'
  'MCCORMICK_Q_POS1,MCCORMICK_Q_POS2,MCCORMICK_Q_POS3,MCCORMICK_Q_NEG0,MCCORMICK_Q_NEG1,'
  'MCCORMICK_Q_NEG2,MCCORMICK_Q_NEG3,epi_ub,e_pos,e_neg,eps,dv_pos,dv_neg,k_p_pos,k_p_neg,'
  'k_q_pos,k_q_neg,dup,duq,gp,gq,VOLTAGE_LB,VOLTAGE_UB'
)

MODSTRING_TOY_ROB_DROOP_NEUMAN = ('FOBJ_ROBUST,EPIGRAPH_UB_NEUMAN,NEUMAN_PFLOW,STABILITY,epi_ub,eps,'
                  'dv_pos,dv_neg,gp,gq')

MODSTRING_TOY_ROB_DROOP_BILINEAR_NEUMAN = (
  'FOBJ_ROBUST,EPIGRAPH_UB_NEUMAN_1,EPIGRAPH_UB_NEUMAN_2,LINEAR_PFLOW,STABILITY,epi_ub,eps,dv,'
  'dup,duq,gp,gq')


print(AMPL_FOLDER)
print(KNITRO_PATH)
print(GUROBI_PATH)
print(CPLEX_PATH)
print(IPOPT_PATH)


EPS_FLOAT_ADALASSO = 1e-6
GUROBI_OPTIONS_DEFAULT = 'outlev=1' # numericfocus=3 aggregate=0 barhomogeneous=1 barcorrectors=1000'
SOLVER_OPTIONS_KNITRO  = 'outlev=1 ms_enable=1 par_numthreads=4 outmode=0 ms_terminate=1 feastol=3.2e-5  convex=0'
SOLVER_OPTIONS_KNITRO_SIMPLE = 'outlev=1 ms_enable=0 outmode=0 feastol=2e-5 convex=0'

def hack_command(function, *args):
  try:
    ret = function(*args)
  except Exception as e:
    raise e
  return ret

def init_ampl_and_solver(ampl_folder, solver_path, solver_options):
  # ampl hacked init
  # ampl = hack_command(AMPL, Environment(ampl_folder))
  ampl = AMPL(Environment(ampl_folder))
  # Set solver
  solver_path = solver_path  # all system 
  ampl.setOption('solver', solver_path)
  solver_name = solver_path.split(os.sep)[-1]
  if solver_options is not None:
    ampl.setOption(solver_name + '_options', solver_options)
  return ampl


def retrieve_2idx_var(ampl, var_name, idx1_name, idx2_name, l_cols_idx2):
  df_ret = ampl.getVariable(var_name).getValues().toPandas()
  df_ret.index = pd.MultiIndex.from_tuples(
    [(int(i), int(j)) for i, j in df_ret.index.to_list()],
    names=(idx1_name, idx2_name)
  )
  df_ret = df_ret.swaplevel(0, 1)
  df_ret = df_ret.unstack(idx1_name)
  df_ret = df_ret.droplevel(0, axis=1)
  df_ret.columns = l_cols_idx2
  return df_ret

# FIXME:
#   - Handle case when there is no branches (unique bus)
#   - Handle no dgs and no b2bs
def opf(pdata, solver_path, solver_options, tmap_mode=1):
  # -------------------------------------------------------------------------------------------- #
  # Initialize ampl environment
  # ---------------------------
  ampl = init_ampl_and_solver(AMPL_FOLDER, solver_path, solver_options)

  # Read mod file
  ampl.read(OPF_MODFILE)
  # -------------------------------------------------------------------------------------------- #
  # Load model data
  # ---------------
  time_map = None
  if type(pdata.df_data.index) is pd.DatetimeIndex:
    time_map = pdata.time_map(tmap_mode)
    n_t = len(time_map)
    l_t = range(n_t)
  else:
    n_t = pdata.df_data.shape[0]
    l_t = range(n_t)

  l_buses0 = pdata.buses.index.to_list()
  l_buses = l_buses0.copy()
  l_buses.remove(pdata.slack_bus)
  l_loads = pdata.l_loads
  l_dgs = pdata.dgs.index.to_list()
  l_b2bs = pdata.b2bs.index.to_list()
  l_branches = pdata.branches.index.to_list()

  # Sets
  ampl.getSet('BUSES').setValues(l_buses)
  ampl.getSet('PERIODS').setValues(list(l_t))

  # General params
  ampl.getParameter('slack_bus').set(pdata.slack_bus)
  ampl.getParameter('cost_nse').set(pdata.cost_nse)
  ampl.getParameter('cost_putility').set(pdata.cost_putility)
  ampl.getParameter('cost_vlim').set(pdata.cost_vlim)
  ampl.getParameter('cost_losses').set(pdata.cost_losses)

  # Indexed params
  #   Buses
  map_buses_rename = {'vmin': 'busVmin', 'vmax': 'busVmax'}
  df_buses_aux = pdata.buses[map_buses_rename.keys()].rename(columns=map_buses_rename)
  df_buses_aux.index.name = 'BUSES0'
  df_buses = DataFrame.fromPandas(df_buses_aux, ['BUSES0'])

  ampl.setData(df_buses, 'BUSES0')

  #   DGs
  if not pdata.dgs.empty:
    map_dgs_rename = {'bus': 'dgBus', 'snom': 'dgSnom'}
    df_dgs_aux = pdata.dgs[map_dgs_rename.keys()].rename(columns=map_dgs_rename)
    df_dgs_aux.index.name = 'DGS'
    df_dgs = DataFrame.fromPandas(df_dgs_aux, ['DGS'])
    ampl.setData(df_dgs, 'DGS')
  else:
    ampl.getSet('DGS').setValues(l_dgs)

  #   B2Bs
  if not pdata.b2bs.empty:
    map_b2bs_rename = {'busf': 'b2bFrom', 'bust': 'b2bSnom', 'snom': 'b2bSnom'}
    df_b2bs_aux = pdata.b2bs[map_b2bs_rename.keys()].rename(columns=map_b2bs_rename)
    df_b2bs_aux.index.name = 'B2BS'
    df_b2bs = DataFrame.fromPandas(df_b2bs_aux, ['B2BS'])
    ampl.setData(df_b2bs, 'B2BS')

  else:
    ampl.getSet('B2BS').setValues(l_b2bs)

  #   Branches
  map_br_rename = {'busf': 'lFrom', 'bust': 'lTo', 'imax': 'lImax', 'x': 'lX', 'r': 'lR',
           'b': 'lB'}
  df_br_aux = pdata.branches[map_br_rename.keys()].rename(columns=map_br_rename)
  df_br_aux.index.name = 'LINES'
  df_br = DataFrame.fromPandas(df_br_aux, ['LINES'])
  ampl.setData(df_br, 'LINES')

  #   Loads
  df_demands = DataFrame('DEMANDS', ['dBus'])
  for i in l_loads:
    df_demands.addRow(i, i)
  ampl.setData(df_demands, 'DEMANDS')

  # Time series params
  if type(pdata.df_data.index) is pd.DatetimeIndex:
    df_proc_data = pdata.df_data.loc[time_map, :]
  else:
    df_proc_data = pdata.df_data.loc[l_t, :]

  l_cols_loadp = [i for i in df_proc_data.columns if i.startswith(SNAM_LOADP)]
  map_loadp_nam2int = {i: int(i.split(SNAM_LOADP)[-1]) for i in l_cols_loadp}
  l_cols_loadq = [i for i in df_proc_data.columns if i.startswith(SNAM_LOADQ)]
  map_loadq_nam2int = {i: int(i.split(SNAM_LOADQ)[-1]) for i in l_cols_loadq}
  l_cols_dgpmax = [i for i in df_proc_data.columns if i.startswith(SNAM_DGPMAX)]
  map_dgpmax_nam2int = {i: int(i.split(SNAM_DGPMAX)[-1]) for i in l_cols_dgpmax}

  df_aux_p = df_proc_data[l_cols_loadp].reset_index(drop=True).rename(
    columns=map_loadp_nam2int)
  df_aux_q = df_proc_data[l_cols_loadq].reset_index(drop=True).rename(
    columns=map_loadq_nam2int)
  df_aux_dgpmax = df_proc_data[l_cols_dgpmax].reset_index(drop=True).rename(
    columns=map_dgpmax_nam2int)

  df_aux_p.columns.name = 'DEMANDS'
  df_aux_p.index.name = 'PERIODS'
  df_aux_p = df_aux_p.stack()
  df_aux_p = df_aux_p.to_frame('tsDemandP')

  df_aux_q.columns.name = 'DEMANDS'
  df_aux_q.index.name = 'PERIODS'
  df_aux_q = df_aux_q.stack()
  df_aux_q = df_aux_q.to_frame('tsDemandQ')

  df_aux_dgpmax.columns.name = 'DGS'
  df_aux_dgpmax.index.name = 'PERIODS'
  df_aux_dgpmax = df_aux_dgpmax.stack()
  df_aux_dgpmax = df_aux_dgpmax.to_frame('tsPmax')

  df_loads_ampl_input = pd.concat([df_aux_p, df_aux_q], axis=1)
  df_tsdemands = DataFrame.fromPandas(df_loads_ampl_input, ['PERIODS', 'DEMANDS'])
  ampl.setData(df_tsdemands)

  df_dgpmax = DataFrame.fromPandas(df_aux_dgpmax, ['PERIODS', 'DGS'])
  ampl.setData(df_dgpmax)

  # Solve
  ampl.eval('problem OPF: {};'.format(OPF_SOCP_MODSTRING))

  


  ampl.solve()# hack_command(ampl.solve)
  str_status = ampl.getObjective("FOBJ").result()
  # str_status = re.search(r'(?<== )\w+', ampl.getOutput('display FOBJ.result;')).group(0)
  assert str_status == 'solved'

  # Return solution
  #   Initialize df_sol
  l_cols_sol = (
    [SNAM_NSE + str(i) for i in l_loads] +
    [SNAM_DGP + str(i) for i in l_dgs] +
    [SNAM_DGQ + str(i) for i in l_dgs] +
    [SNAM_B2BFP + str(i) for i in l_b2bs] +
    [SNAM_B2BFQ + str(i) for i in l_b2bs] +
    [SNAM_B2BTP + str(i) for i in l_b2bs] +
    [SNAM_B2BTQ + str(i) for i in l_b2bs] +
    [SNAM_P + str(i) for i in l_branches] +
    [SNAM_Q + str(i) for i in l_branches] +
    [SNAM_I + str(i) for i in l_branches]
  )
  df_opf_sol = pd.DataFrame(index=l_t, columns=l_cols_sol)

  for t in l_t:
    for i in l_buses0:
      df_opf_sol.loc[t, SNAM_V + str(i)] = np.sqrt(ampl.getVariable('v')[i, t].value())

    for i in l_branches:
      df_opf_sol.loc[t, SNAM_P + str(i)] = ampl.getVariable('P')[i, t].value()
      df_opf_sol.loc[t, SNAM_Q + str(i)] = ampl.getVariable('Q')[i, t].value()
      df_opf_sol.loc[t, SNAM_I + str(i)] = np.sqrt(ampl.getVariable('l')[i, t].value())

    for i in l_loads:
      df_opf_sol.loc[t, SNAM_NSE + str(i)] = ampl.getVariable('nse')[i, t].value()
    for i in l_dgs:
      df_opf_sol.loc[t, SNAM_DGP + str(i)] = ampl.getVariable('dgp')[i, t].value()
      df_opf_sol.loc[t, SNAM_DGQ + str(i)] = ampl.getVariable('dgq')[i, t].value()
    for i in l_b2bs:
      df_opf_sol.loc[t, SNAM_B2BTP + str(i)] = ampl.getVariable('b2btp')[i, t].value()
      df_opf_sol.loc[t, SNAM_B2BTQ + str(i)] = ampl.getVariable('b2btq')[i, t].value()
      df_opf_sol.loc[t, SNAM_B2BFP + str(i)] = ampl.getVariable('b2bfp')[i, t].value()
      df_opf_sol.loc[t, SNAM_B2BFQ + str(i)] = ampl.getVariable('b2bfq')[i, t].value()

  if type(pdata.df_data.index) is pd.DatetimeIndex:
    df_opf_sol.index = time_map
  else:
    df_opf_sol.index = l_t
  df_opf_sol = df_opf_sol.astype(float)
  return df_opf_sol

class StaticModel(ABC):
  def __init__(self, pdata):
    self.pdata = pdata

  def run(self, df_data):
    ...

class OPF(StaticModel):
  def __init__(self, pdata):
    super().__init__(pdata)

    if pdata.opf_model == 'distflow':
      self._require_hotstart = True
      self.solver_path = KNITRO_PATH  # IPOPT_PATH
      # FIXME: This if alternative is broken and unused
    else:
      self._require_hotstart = False
      self.solver_path = GUROBI_PATH

    self.mod_file = OPF_MODFILE
    self.mod_string = opf_modstrings[pdata.opf_model]
    self.solver_options = 'outlev=1'  # iisfind=1 iismethod=1
    self.ampl = init_ampl_and_solver(AMPL_FOLDER, self.solver_path, self.solver_options)
    self.ampl.read(OPF_MODFILE)

    # Load grid data
    self.pdata = pdata
    self.l_buses0 = pdata.buses.index.to_list()
    self.l_buses = self.l_buses0.copy()
    self.l_buses.remove(pdata.slack_bus)
    self.l_loads = pdata.l_loads
    self.l_dgs = pdata.dgs.index.to_list()
    self.l_b2bs = pdata.b2bs.index.to_list()
    self.l_branches = pdata.branches.index.to_list()
    self.l_cols_loadp = [i for i in pdata.df_data.columns if i.startswith(SNAM_LOADP)]
    self.map_loadp_nam2int = {i: int(i.split(SNAM_LOADP)[-1]) for i in self.l_cols_loadp}
    self.l_cols_loadq = [i for i in pdata.df_data.columns if i.startswith(SNAM_LOADQ)]
    self.map_loadq_nam2int = {i: int(i.split(SNAM_LOADQ)[-1]) for i in self.l_cols_loadq}
    self.l_cols_dgpmax = [i for i in pdata.df_data.columns if i.startswith(SNAM_DGPMAX)]
    self.map_dgpmax_nam2int = {i: int(i.split(SNAM_DGPMAX)[-1]) for i in self.l_cols_dgpmax}

    # Sets
    self.ampl.getSet('BUSES').setValues(self.l_buses)

    # General params
    self.ampl.getParameter('slack_bus').set(pdata.slack_bus)
    self.ampl.getParameter('cost_nse').set(pdata.cost_nse)
    self.ampl.getParameter('cost_putility').set(pdata.cost_putility)
    self.ampl.getParameter('cost_vlim').set(pdata.cost_vlim)
    self.ampl.getParameter('cost_losses').set(pdata.cost_losses)

    # Indexed params
    #   Buses
    map_buses_rename = {'vmin': 'busVmin', 'vmax': 'busVmax'}
    df_buses_aux = pdata.buses[map_buses_rename.keys()].rename(columns=map_buses_rename)
    df_buses_aux.index.name = 'BUSES0'
    df_buses = DataFrame.fromPandas(df_buses_aux, ['BUSES0'])

    self.ampl.setData(df_buses, 'BUSES0')

    #   DGs
    if not pdata.dgs.empty:
      map_dgs_rename = {'bus': 'dgBus', 'snom': 'dgSnom'}
      df_dgs_aux = pdata.dgs[map_dgs_rename.keys()].rename(columns=map_dgs_rename)
      df_dgs_aux.index.name = 'DGS'
      df_dgs = DataFrame.fromPandas(df_dgs_aux, ['DGS'])
      self.ampl.setData(df_dgs, 'DGS')
    else:
      self.ampl.getSet('DGS').setValues(self.l_dgs)

    #   B2Bs
    if not pdata.b2bs.empty:
      map_b2bs_rename = {'busf': 'b2bFrom', 'bust': 'b2bSnom', 'snom': 'b2bSnom'}
      df_b2bs_aux = pdata.b2bs[map_b2bs_rename.keys()].rename(columns=map_b2bs_rename)
      df_b2bs_aux.index.name = 'B2BS'
      df_b2bs = DataFrame.fromPandas(df_b2bs_aux, ['B2BS'])
      self.ampl.setData(df_b2bs, 'B2BS')

    else:
      self.ampl.getSet('B2BS').setValues(self.l_b2bs)

    #   Branches
    map_br_rename = {'busf': 'lFrom', 'bust': 'lTo', 'imax': 'lImax', 'x': 'lX', 'r': 'lR',
             'b': 'lB'}
    df_br_aux = pdata.branches[map_br_rename.keys()].rename(columns=map_br_rename)
    df_br_aux.index.name = 'LINES'
    df_br = DataFrame.fromPandas(df_br_aux, ['LINES'])
    self.ampl.setData(df_br, 'LINES')

    #   Loads
    df_demands = DataFrame('DEMANDS', ['dBus'])
    for i in self.l_loads:
      df_demands.addRow(i, i)
    self.ampl.setData(df_demands, 'DEMANDS')

    #   Initialize df_sol
    self.l_cols_sol = (
      [SNAM_NSE + str(i) for i in self.l_loads] +
      [SNAM_DGP + str(i) for i in self.l_dgs] +
      [SNAM_DGQ + str(i) for i in self.l_dgs] +
      [SNAM_B2BFP + str(i) for i in self.l_b2bs] +
      [SNAM_B2BFQ + str(i) for i in self.l_b2bs] +
      [SNAM_B2BTP + str(i) for i in self.l_b2bs] +
      [SNAM_B2BTQ + str(i) for i in self.l_b2bs] +
      [SNAM_P + str(i) for i in self.l_branches] +
      [SNAM_Q + str(i) for i in self.l_branches] +
      [SNAM_I + str(i) for i in self.l_branches]
    )

    self.ampl.eval('problem opf: {};'.format(self.mod_string))

    self.df_sol = None
    self.n_t = None
    self.l_t = None

  def set_hot_start(self):
    hot_start_modstring = opf_modstrings[OPF_HOTSTART_MODEL]
    self.ampl.eval('problem hot_start: {};'.format(hot_start_modstring))
    hack_command(self.ampl.eval, 'solve hot_start;')

  def run(self, df_data=None, trange=None):
    assert df_data is None or trange is None

    if trange is None and df_data is None:
      trange = self.pdata.time_map(0)
      df_data = self.pdata.df_data.loc[trange, :]

    elif trange is None:
      if isinstance(df_data.index, pd.DatetimeIndex):
        trange = df_data.index
      else:
        trange = None
    else:
      df_data = self.pdata.df_data.loc[trange, :]

    self.n_t = df_data.shape[0]
    self.l_t = range(self.n_t)
    self.ampl.getSet('PERIODS').setValues(list(self.l_t))

    if self.df_sol is None:
      self.df_sol = pd.DataFrame(index=self.l_t, columns=self.l_cols_sol)

    # Time series params
    df_aux_p = df_data[self.l_cols_loadp].reset_index(drop=True).rename(
      columns=self.map_loadp_nam2int)
    df_aux_q = df_data[self.l_cols_loadq].reset_index(drop=True).rename(
      columns=self.map_loadq_nam2int)
    df_aux_dgpmax = df_data[self.l_cols_dgpmax].reset_index(drop=True).rename(
      columns=self.map_dgpmax_nam2int)

    df_aux_p.columns.name = 'DEMANDS'
    df_aux_p.index.name = 'PERIODS'
    df_aux_p = df_aux_p.stack()
    df_aux_p = df_aux_p.to_frame('tsDemandP')

    df_aux_q.columns.name = 'DEMANDS'
    df_aux_q.index.name = 'PERIODS'
    df_aux_q = df_aux_q.stack()
    df_aux_q = df_aux_q.to_frame('tsDemandQ')

    df_aux_dgpmax.columns.name = 'DGS'
    df_aux_dgpmax.index.name = 'PERIODS'
    df_aux_dgpmax = df_aux_dgpmax.stack()
    df_aux_dgpmax = df_aux_dgpmax.to_frame('tsPmax')

    df_loads_ampl_input = pd.concat([df_aux_p, df_aux_q], axis=1)
    df_tsdemands = DataFrame.fromPandas(df_loads_ampl_input, ['PERIODS', 'DEMANDS'])
    self.ampl.setData(df_tsdemands)

    df_dgpmax = DataFrame.fromPandas(df_aux_dgpmax, ['PERIODS', 'DGS'])
    self.ampl.setData(df_dgpmax)

    # Run model initialization procedure if necessary
    if self._require_hotstart:
      self.set_hot_start()
      self._require_hotstart = False  # Run initialization only the first time

    hack_command(self.ampl.eval, 'solve opf;')

    # TODO: avoid patching
    # objective function name flexibility patch
    fobj_name = opf_mod_fobj_names[self.pdata.opf_model]
    str_status = re.search(r'(?<== )\w+', self.ampl.getOutput('display {}.result;'.format(
      fobj_name))).group(0)

    if str_status != 'solved':
      raise ExperimentError('Static model OPF is unfeasible!')

    # OPTI: Avoid for loops to extract data from AMPL. Use dataframes instead !!!!!!!!!!!
    for t in self.l_t:
      for i in self.l_buses0:
        self.df_sol.loc[t, SNAM_V + str(i)] = np.sqrt(self.ampl.getVariable('v')[i, t].value())

      for i in self.l_branches:
        self.df_sol.loc[t, SNAM_P + str(i)] = self.ampl.getVariable('P')[i, t].value()
        self.df_sol.loc[t, SNAM_Q + str(i)] = self.ampl.getVariable('Q')[i, t].value()
        self.df_sol.loc[t, SNAM_I + str(i)] = np.sqrt(self.ampl.getVariable('l')[i, t].value())

      for i in self.l_loads:
        self.df_sol.loc[t, SNAM_NSE + str(i)] = self.ampl.getVariable('nse')[i, t].value()
      for i in self.l_dgs:
        self.df_sol.loc[t, SNAM_DGP + str(i)] = self.ampl.getVariable('dgp')[i, t].value()
        self.df_sol.loc[t, SNAM_DGQ + str(i)] = self.ampl.getVariable('dgq')[i, t].value()
      for i in self.l_b2bs:
        self.df_sol.loc[t, SNAM_B2BTP + str(i)] = self.ampl.getVariable('b2btp')[i, t].value()
        self.df_sol.loc[t, SNAM_B2BTQ + str(i)] = self.ampl.getVariable('b2btq')[i, t].value()
        self.df_sol.loc[t, SNAM_B2BFP + str(i)] = self.ampl.getVariable('b2bfp')[i, t].value()
        self.df_sol.loc[t, SNAM_B2BFQ + str(i)] = self.ampl.getVariable('b2bfq')[i, t].value()

    return self.df_sol

StaticModel.register(OPF)

class OPFsto(StaticModel):
  def __init__(self, pdata):
    super().__init__(pdata)

    # Initial config
    self.fobj_name = 'FOBJ'
    self.mod_file = OPF_STO_MODFILE
    self.solver_options = 'outlev=1'
    self.solver_path = GUROBI_PATH
    self.ampl = init_ampl_and_solver(AMPL_FOLDER, self.solver_path, self.solver_options)
    self.ampl.read(self.mod_file)

    # Load grid data
    self.pdata = pdata
    self.l_buses0 = pdata.buses.index.to_list()
    self.l_buses = self.l_buses0.copy()
    self.l_buses.remove(pdata.slack_bus)
    self.l_loads = pdata.l_loads
    self.l_dgs = pdata.dgs.index.to_list()
    self.l_b2bs = pdata.b2bs.index.to_list()
    self.l_branches = pdata.branches.index.to_list()
    self.l_cols_loadp = [i for i in pdata.df_data.columns if i.startswith(SNAM_LOADP)]
    self.map_loadp_nam2int = {i: int(i.split(SNAM_LOADP)[-1]) for i in self.l_cols_loadp}
    self.l_cols_loadq = [i for i in pdata.df_data.columns if i.startswith(SNAM_LOADQ)]
    self.map_loadq_nam2int = {i: int(i.split(SNAM_LOADQ)[-1]) for i in self.l_cols_loadq}
    self.l_cols_dgpmax = [i for i in pdata.df_data.columns if i.startswith(SNAM_DGPMAX)]
    self.map_dgpmax_nam2int = {i: int(i.split(SNAM_DGPMAX)[-1]) for i in self.l_cols_dgpmax}

    # Sets
    self.ampl.getSet('BUSES').setValues(self.l_buses)

    # General params
    self.ampl.getParameter('slack_bus').set(pdata.slack_bus)
    self.ampl.getParameter('cost_putility').set(pdata.cost_putility)
    self.ampl.getParameter('cost_vlim').set(pdata.cost_vlim)

    # Indexed params
    #   Buses
    map_buses_rename = {'vmin': 'busVmin', 'vmax': 'busVmax'}
    df_buses_aux = pdata.buses[map_buses_rename.keys()].rename(columns=map_buses_rename)
    df_buses_aux.index.name = 'BUSES0'
    df_buses = DataFrame.fromPandas(df_buses_aux, ['BUSES0'])

    self.ampl.setData(df_buses, 'BUSES0')

    #   DGs
    if not pdata.dgs.empty:
      map_dgs_rename = {'bus': 'dgBus', 'snom': 'dgSnom'}
      df_dgs_aux = pdata.dgs[map_dgs_rename.keys()].rename(columns=map_dgs_rename)
      df_dgs_aux.index.name = 'DGS'
      df_dgs = DataFrame.fromPandas(df_dgs_aux, ['DGS'])
      self.ampl.setData(df_dgs, 'DGS')
    else:
      self.ampl.getSet('DGS').setValues(self.l_dgs)

    #   B2Bs
    if not pdata.b2bs.empty:
      map_b2bs_rename = {'busf': 'b2bFrom', 'bust': 'b2bSnom', 'snom': 'b2bSnom'}
      df_b2bs_aux = pdata.b2bs[map_b2bs_rename.keys()].rename(columns=map_b2bs_rename)
      df_b2bs_aux.index.name = 'B2BS'
      df_b2bs = DataFrame.fromPandas(df_b2bs_aux, ['B2BS'])
      self.ampl.setData(df_b2bs, 'B2BS')

    else:
      self.ampl.getSet('B2BS').setValues(self.l_b2bs)

    #   Branches
    map_br_rename = {'busf': 'lFrom', 'bust': 'lTo', 'imax': 'lImax', 'x': 'lX', 'r': 'lR',
             'b': 'lB'}
    df_br_aux = pdata.branches[map_br_rename.keys()].rename(columns=map_br_rename)
    df_br_aux.index.name = 'LINES'
    df_br = DataFrame.fromPandas(df_br_aux, ['LINES'])
    self.ampl.setData(df_br, 'LINES')

    #   Loads
    df_demands = DataFrame('DEMANDS', ['dBus'])
    for i in self.l_loads:
      df_demands.addRow(i, i)
    self.ampl.setData(df_demands, 'DEMANDS')

    self.l_cols_dgp = [SNAM_DGP + str(i) for i in self.l_dgs]
    self.l_cols_dgq = [SNAM_DGQ + str(i) for i in self.l_dgs]
    self.l_cols_b2bfp = [SNAM_B2BFP + str(i) for i in self.l_b2bs]
    self.l_cols_b2bfq = [SNAM_B2BFQ + str(i) for i in self.l_b2bs]
    self.l_cols_b2btp = [SNAM_B2BTP + str(i) for i in self.l_b2bs]
    self.l_cols_b2btq = [SNAM_B2BTQ + str(i) for i in self.l_b2bs]

    self.l_cols_sol = (
      self.l_cols_dgp + self.l_cols_dgq + self.l_cols_b2bfp + self.l_cols_b2bfq +
      self.l_cols_b2btp + self.l_cols_b2btq
    )

    #   Initialize df_sol
    self.df_sol = pd.DataFrame(index=[0], columns=self.l_cols_sol, dtype='float64')
    self.n_t = None
    self.l_t = None

  def run(self, df_data=None, trange=None, df_sto_weights=None):
    # Validate input
    if not (df_data is None or trange is None):
      raise InputError('Only one of the arguments df_data and trange must be passed!')

    if trange is None and df_data is None:
      trange = self.pdata.time_map(0)
      df_data = self.pdata.df_data.loc[trange, :]

    elif trange is None:
      if isinstance(df_data.index, pd.DatetimeIndex):
        trange = df_data.index
      else:
        trange = None
    else:
      df_data = self.pdata.df_data.loc[trange, :]

    # Set Params
    if not (df_sto_weights is None):
      self.ampl.setData(DataFrame.fromPandas(df_sto_weights, ['PERIODS']))

    self.n_t = df_data.shape[0]
    self.l_t = range(self.n_t)
    self.ampl.getSet('PERIODS').setValues(list(self.l_t))

    # Time series params
    df_aux_p = df_data[self.l_cols_loadp].reset_index(drop=True).rename(
      columns=self.map_loadp_nam2int)
    df_aux_q = df_data[self.l_cols_loadq].reset_index(drop=True).rename(
      columns=self.map_loadq_nam2int)
    df_aux_dgpmax = df_data[self.l_cols_dgpmax].reset_index(drop=True).rename(
      columns=self.map_dgpmax_nam2int)

    df_aux_p.columns.name = 'DEMANDS'
    df_aux_p.index.name = 'PERIODS'
    df_aux_p = df_aux_p.stack()
    df_aux_p = df_aux_p.to_frame('tsDemandP')

    df_aux_q.columns.name = 'DEMANDS'
    df_aux_q.index.name = 'PERIODS'
    df_aux_q = df_aux_q.stack()
    df_aux_q = df_aux_q.to_frame('tsDemandQ')

    df_aux_dgpmax.columns.name = 'DGS'
    df_aux_dgpmax.index.name = 'PERIODS'
    df_aux_dgpmax = df_aux_dgpmax.stack()
    df_aux_dgpmax = df_aux_dgpmax.to_frame('tsPmax')

    df_loads_ampl_input = pd.concat([df_aux_p, df_aux_q], axis=1)
    df_tsdemands = DataFrame.fromPandas(df_loads_ampl_input, ['PERIODS', 'DEMANDS'])
    self.ampl.setData(df_tsdemands)

    df_dgpmax = DataFrame.fromPandas(df_aux_dgpmax, ['PERIODS', 'DGS'])
    self.ampl.setData(df_dgpmax)

    

    # TODO: avoid patching
    # objective function name flexibility patch
    # hack_command(self.ampl.solve)
    self.ampl.solve()
    
    str_status = self.ampl.getObjective(self.fobj_name).result()
    if str_status != 'solved':
      raise ExperimentError('Static model OPF is unfeasible!')

    # ---------------------------------------------------------------------------------------- #
    # Obtain solution                                      #
    # ---------------------------------------------------------------------------------------- #
    # Main solution
    ar_dgp = self.ampl.getData('dgp').toPandas()['dgp'].values
    ar_dgq = self.ampl.getData('dgq').toPandas()['dgq'].values
    self.df_sol.loc[0, self.l_cols_dgp] = ar_dgp
    self.df_sol.loc[0, self.l_cols_dgq] = ar_dgq

    if self.l_b2bs:
      ar_b2btp = self.ampl.GetData('b2btp').toPandas()['b2btp'].values
      ar_b2btq = self.ampl.GetData('b2btq').toPandas()['b2btq'].values
      ar_b2bfp = self.ampl.GetData('b2bfp').toPandas()['b2bfp'].values
      ar_b2bfq = self.ampl.GetData('b2bfq').toPandas()['b2bfq'].values
      self.df_sol.loc[0, self.l_cols_b2btp] = ar_b2btp
      self.df_sol.loc[0, self.l_cols_b2btq] = ar_b2btq
      self.df_sol.loc[0, self.l_cols_b2bfp] = ar_b2bfp
      self.df_sol.loc[0, self.l_cols_b2bfq] = ar_b2bfq

    return self.df_sol

StaticModel.register(OPFsto)

class Pflow:
  def __init__(self, pdata: ds.Adn):

    if pdata.l_b2bs:
      raise(NotSupportedYet('{} does not support b2bs yet!'.format(self.__class__.__name__)))

    # Aliasing
    self.l_buses0 = pdata.l_buses0
    self.l_buses = pdata.l_buses
    self.l_dgs = pdata.l_dgs
    self.l_loads = pdata.l_loads

    self.n_buses = len(self.l_buses)
    self.n_loads = len(self.l_loads)
    self.n_dgs = len(self.l_dgs)

    self.dgs = pdata.dgs
    self.slack_bus = pdata.slack_bus
    self.sbase_mva = pdata.sbase_mva
    self.branches = pdata.branches

    self.l_cols_loadp = [SNAM_LOADP + str(i) for i in self.l_loads]
    self.l_cols_loadq = [SNAM_LOADQ + str(i) for i in self.l_loads]
    self.l_cols_dgpmax = [SNAM_DGPMAX + str(i) for i in self.l_dgs]
    self.l_cols_dgp = [SNAM_DGP + str(i) for i in self.l_dgs]
    self.l_cols_dgq = [SNAM_DGQ + str(i) for i in self.l_dgs]
    self.l_cols_v = [SNAM_V + str(i) for i in self.l_buses]
    self.l_cols_vang = [SNAM_VANG + str(i) for i in self.l_buses]

    # Make pdata matrices
    pdata.make_connectivity_mats()
    self.A_load_t = pdata.A_load.values[:, 1:].transpose()
    assert self.A_load_t.shape == (self.n_buses, self.n_loads)
    self.A_dg_t = pdata.A_dg.values.transpose()

    # Solver config
    self.solver_path_sim = KNITRO_PATH  # IPOPT_PATH
    self.mod_file_sim = PFLOW_BIM_MODFILE
    self.solver_options_sim = 'outlev=0'

    # Initialize ampl environment
    self.ampl = init_ampl_and_solver(AMPL_FOLDER, self.solver_path_sim, self.solver_options_sim)
    self.ampl.read(self.mod_file_sim)

    # Initialize sets & params
    #   Sets
    self.ampl.getSet('BUSES0').setValues(self.l_buses0)
    self.ampl.getSet('BUSES').setValues(self.l_buses)

    #   Params
    self.ampl.getParameter('SLACK_BUS').set(self.slack_bus)

    map_dgs_rename = {'bus': 'dgBus'}
    df_dgs_aux = self.dgs.loc[:, map_dgs_rename.keys()].rename(columns=map_dgs_rename)

    if not self.dgs.empty:
      df_dgs_aux.index.name = 'DGS'
      df_dgs = DataFrame.fromPandas(df_dgs_aux, ['DGS'])
      self.ampl.setData(df_dgs, 'DGS')
    else:
      self.ampl.getSet('DGS').setValues(self.l_dgs)

    map_branches_rename = {'bust': 'lTo'}
    df_branches_aux = self.branches[map_branches_rename.keys()].rename(
      columns=map_branches_rename)
    df_branches_aux['ly'] = self.branches.eval('1/sqrt(x ** 2 + r ** 2)')
    df_branches_aux['ltheta'] = - np.arctan2(self.branches['x'], self.branches['r'])
    df_branches_aux.index.name = 'LINES'
    df_branches = DataFrame.fromPandas(df_branches_aux, ['BUSES'])
    self.ampl.setData(df_branches)

    df_demands = DataFrame('DEMANDS', ['dBus'])
    for i in self.l_loads:
      df_demands.addRow(i, i)
    self.ampl.setData(df_demands, 'DEMANDS')

  def run(self, df_data: pd.DataFrame, df_opf_sol: pd.DataFrame):
    # Validate input
    if df_opf_sol.shape[0] != 1:
      raise InputError('df_opf_sol must have only one row indexed 0 !')

    # Initialize solution container
    trange = df_data.index
    df_sim_sol = pd.DataFrame(index=trange, columns=self.l_cols_sol, dtype='float64')

    # Set opf solution
    l_dgp = df_opf_sol.loc[0, self.l_cols_dgp].to_list()
    l_dgq = df_opf_sol.loc[0, self.l_cols_dgq].to_list()

    for t in trange:
      self.ampl.getParameter('loadp').setValues(
        df_data.loc[t, self.l_cols_loadp].to_list())
      self.ampl.getParameter('loadq').setValues(
        df_data.loc[t, self.l_cols_loadq].to_list())

      self.ampl.getParameter('dgp').setValues(l_dgp)
      self.ampl.getParameter('dgq').setValues(l_dgq)

      self.ampl.solve()

      str_status = re.search(
        r'(?<== )\w+', self.ampl.getOutput('display FOBJ.result;')).group(0)

      if str_status != 'solved':
        raise ExperimentError(
          'Infeasible pflow instance at index {} (str_status={})'.format(t, str_status)
        )

      df_aux_sol = self.ampl.getVariable('V').getValues().toPandas()
      df_sim_sol.loc[t, self.l_cols_v] = df_aux_sol['V.val'].values[1:]
      df_aux_sol = self.ampl.getVariable('theta').getValues().toPandas()
      df_sim_sol.loc[t, self.l_cols_vang] = df_aux_sol['theta.val'].values[1:]
      df_sim_sol.loc[t, self.l_cols_dgp] = l_dgp
      df_sim_sol.loc[t, self.l_cols_dgq] = l_dgq

    return df_sim_sol

  @property
  def l_cols_sol(self):
    map_names2sets = {
      SNAM_V: self.l_buses,
      SNAM_VANG: self.l_buses,
      SNAM_DGP: self.l_dgs,
      SNAM_DGQ: self.l_dgs
    }
    col_names = []

    for k, v in map_names2sets.items():
      col_names += [k + str(i) for i in v]

    return col_names

class OPFac:
  def __init__(self, pdata: ds.Adn):
    # Aliasing
    self.l_buses0 = pdata.l_buses0
    self.l_buses = pdata.l_buses
    self.l_dgs = pdata.l_dgs
    self.l_loads = pdata.l_loads

    self.n_buses = len(self.l_buses)
    self.n_loads = len(self.l_loads)
    self.n_dgs = len(self.l_dgs)

    self.dgs = pdata.dgs
    self.slack_bus = pdata.slack_bus
    self.sbase_mva = pdata.sbase_mva
    self.branches = pdata.branches
    self.cost_putility = pdata.cost_putility
    self.cost_vlim = pdata.cost_vlim

    self.l_cols_loadp = [SNAM_LOADP + str(i) for i in self.l_loads]
    self.l_cols_loadq = [SNAM_LOADQ + str(i) for i in self.l_loads]
    self.l_cols_dgpmax = [SNAM_DGPMAX + str(i) for i in self.l_dgs]
    self.l_cols_dgp = [SNAM_DGP + str(i) for i in self.l_dgs]
    self.l_cols_dgq = [SNAM_DGQ + str(i) for i in self.l_dgs]
    self.l_cols_v = [SNAM_V + str(i) for i in self.l_buses]
    self.l_cols_vang = [SNAM_VANG + str(i) for i in self.l_buses]

    # Solver config
    self.solver_path_sim = KNITRO_PATH  # IPOPT_PATH
    self.mod_file_sim = OPF_AC_MODFILE
    self.solver_options_sim = 'outlev=0'

    # Initialize ampl environment
    self.ampl = init_ampl_and_solver(AMPL_FOLDER, self.solver_path_sim, self.solver_options_sim)
    self.ampl.read(self.mod_file_sim)

    # Initialize sets & params
    #   Sets
    # self.ampl.getSet('BUSES0').setValues(self.l_buses0)
    self.ampl.getSet('BUSES').setValues(self.l_buses)

    #   Params
    self.ampl.getParameter('SLACK_BUS').set(self.slack_bus)
    self.ampl.getParameter('cost_putility').set(self.cost_putility)
    self.ampl.getParameter('cost_vlim').set(self.cost_vlim)

    #   Buses
    map_buses_rename = {'vmin': 'busVmin', 'vmax': 'busVmax'}
    df_buses_aux = pdata.buses[map_buses_rename.keys()].rename(columns=map_buses_rename)
    df_buses_aux.index.name = 'BUSES0'
    df_buses = DataFrame.fromPandas(df_buses_aux, ['BUSES0'])

    self.ampl.setData(df_buses, 'BUSES0')

    #   Dgs
    map_dgs_rename = {'bus': 'dgBus', 'snom': 'dgSnom'}
    df_dgs_aux = self.dgs.rename(columns=map_dgs_rename)

    if not self.dgs.empty:
      df_dgs_aux.index.name = 'DGS'
      df_dgs = DataFrame.fromPandas(df_dgs_aux, ['DGS'])
      self.ampl.setData(df_dgs, 'DGS')
    else:
      self.ampl.getSet('DGS').setValues(self.l_dgs)

    map_branches_rename = {'bust': 'lTo'}
    df_branches_aux = self.branches[map_branches_rename.keys()].rename(
      columns=map_branches_rename)
    df_branches_aux['ly'] = self.branches.eval('1/sqrt(x ** 2 + r ** 2)')
    df_branches_aux['ltheta'] = - np.arctan2(self.branches['x'], self.branches['r'])
    df_branches_aux.index.name = 'LINES'
    df_branches = DataFrame.fromPandas(df_branches_aux, ['BUSES'])
    self.ampl.setData(df_branches)

    df_demands = DataFrame('DEMANDS', ['dBus'])
    for i in self.l_loads:
      df_demands.addRow(i, i)
    self.ampl.setData(df_demands, 'DEMANDS')

  def run(self, df_data: pd.DataFrame, df_opf_hotstart: pd.DataFrame = None):
    # Modify
    # - FOBJ column was added in dataframe, for each operating point.
    # Validate input
    if df_opf_hotstart.shape[0] != 1:
      raise InputError('df_opf_sol must have only one row indexed 0 !')

    # Initialize solution container
    # df_sim_sol = pd.DataFrame(index=trange, columns=self.l_cols_sol, dtype='float64')
    trange = df_data.index
    df_sim_sol = pd.DataFrame(index=trange, columns=self.l_cols_sol+['FOBJ'], dtype='float64')

    # Set hotstart if given
    if df_opf_hotstart is not None:
      for i in self.l_dgs:
        self.ampl.getVariable('dgp')[i].setValue(df_opf_hotstart.loc[0, SNAM_DGP + str(i)])
        self.ampl.getVariable('dgq')[i].setValue(df_opf_hotstart.loc[0, SNAM_DGQ + str(i)])

    for t in trange:
      # Set time series parameters
      self.ampl.getParameter('loadp').setValues(
        df_data.loc[t, self.l_cols_loadp].to_list())
      self.ampl.getParameter('loadq').setValues(
        df_data.loc[t, self.l_cols_loadq].to_list())
      self.ampl.getParameter('dgpmax').setValues(
        df_data.loc[t, self.l_cols_dgpmax].to_list()
      )

      # Solve model
      self.ampl.solve()

      str_status = re.search(
        r'(?<== )\w+', self.ampl.getOutput('display FOBJ.result;')).group(0)

      if str_status != 'solved':
        raise ExperimentError(
          'Infeasible pflow instance at index {} (str_status={})'.format(t, str_status)
        )

        
      
      df_aux_sol = self.ampl.getVariable('V').getValues().toPandas()
      df_sim_sol.loc[t, self.l_cols_v] = df_aux_sol['V.val'].values[1:]
      df_aux_sol = self.ampl.getVariable('theta').getValues().toPandas()
      df_sim_sol.loc[t, self.l_cols_vang] = df_aux_sol['theta.val'].values[1:]
      df_aux_sol = self.ampl.getVariable('dgp').getValues().toPandas()
      df_sim_sol.loc[t, self.l_cols_dgp] = df_aux_sol['dgp.val'].values
      df_aux_sol = self.ampl.getVariable('dgq').getValues().toPandas()
      df_sim_sol.loc[t, self.l_cols_dgq] = df_aux_sol['dgq.val'].values

      # Modifications
      df_aux_sol = self.ampl.getObjective('FOBJ').getValues().toPandas()
      df_sim_sol.loc[t, 'FOBJ'] = df_aux_sol['FOBJ'].values
      

    return df_sim_sol

  @property
  def l_cols_sol(self):
    map_names2sets = {
      SNAM_V: self.l_buses,
      SNAM_VANG: self.l_buses,
      SNAM_DGP: self.l_dgs,
      SNAM_DGQ: self.l_dgs
    }
    col_names = []

    for k, v in map_names2sets.items():
      col_names += [k + str(i) for i in v]

    return col_names

class Controller(ABC):
  def __init__(self, u_snom=None):
    solver_path = KNITRO_PATH  # IPOPT_PATH
    solver_options = 'outlev=0'
    self.ampl = init_ampl_and_solver(AMPL_FOLDER, solver_path, solver_options)
    self.ampl.read(PROJ_MODFILE)
    self.FLOAT_EPS = 1e-5
    if u_snom is not None:
      self.u_snom = u_snom
      self.n = u_snom.shape[0]
    else:
      self.u_snom = None
    self.project = True

  def do_projection(self, up, uq, up_max):
    # Evaluate conditions to be projected
    up = np.maximum(up, 0.)
    bool_proj = np.logical_or(
      abs(up) >= self.FLOAT_EPS,
      abs(uq) >= self.FLOAT_EPS
    )
    bool_proj_aux = np.logical_or(
      up > up_max, up ** 2 + uq ** 2 > self.u_snom ** 2)
    bool_proj_aux = np.logical_or(bool_proj_aux, up < 0.)

    bool_proj = np.logical_and(bool_proj, bool_proj_aux).flatten()

    # Projection
    for i in range(len(bool_proj)):
      if bool_proj[i]:
        self.ampl.getParameter('pmax').set(up_max[i, 0])
        self.ampl.getParameter('snom').set(self.u_snom[i, 0])
        self.ampl.getParameter('dgp_setpoint').set(up[i, 0])
        self.ampl.getParameter('dgq_setpoint').set(uq[i, 0])

        self.ampl.solve()

        str_status = re.search(r'(?<== )\w+',
                     self.ampl.getOutput('display FOBJ.result;')).group(0)
        assert str_status == 'solved'

        up[i, 0] = self.ampl.getVariable('dgp').value()
        uq[i, 0] = self.ampl.getVariable('dgq').value()
    return up, uq

  @abstractmethod
  def ret(self, v_abs, up_max, bus_loadp, bus_loadq):
    pass

class OPFController(Controller):
  def __init__(self, u_snom):
    super().__init__()
    self.n = self.n = u_snom.shape[0]
    self.u_snom = u_snom.reshape(self.n, 1)
    self.up_nom = None
    self.uq_nom = None

  def update(self, up_nom, uq_nom):
    self.up_nom = up_nom
    self.uq_nom = uq_nom

  def is_valid(self):
    if self.u_snom is None:
      return False

    return True

  def ret(self, v_abs, up_max, bus_loadp, bus_loadq):
    up = self.up_nom
    uq = self.uq_nom
    if self.project:
      up, uq = self.do_projection(up, uq, up_max)
    return up, uq

class CtrlNCognizant(Controller):
  def __init__(self):
    super().__init__()
    self.Gv_p = None
    self.Gv_q = None
    self.v_abs_nom = None
    self.up_nom = None
    self.uq_nom = None
    self.active = True

  def update(self, Gv_p, Gv_q, v_abs_nom, up_nom, uq_nom):
    # TODO: delete after debug
    log_file_name = 'log_ncognizant_update.csv'
    ar_g = np.concatenate([Gv_p.diagonal(), Gv_q.diagonal()])
    with open(log_file_name, 'a+') as hfile:
      hfile.write(','.join([str(i) for i in ar_g]) + '\n')

    self.Gv_p = Gv_p
    self.Gv_q = Gv_q
    self.v_abs_nom = v_abs_nom
    self.up_nom = up_nom
    self.uq_nom = uq_nom

  def isinit(self):
    return (
      self.Gv_p is not None and
      self.Gv_q is not None and
      self.v_abs_nom is not None and
      self.up_nom is not None and
      self.uq_nom is not None and
      self.snom is not None
    )

  def ret(self, v_abs, up_max, bus_loadp, bus_loadq):
    # Calculate power injection
    if self.active:
      n_buses = self.up_nom.shape[0]
      v_abs_del = v_abs - self.v_abs_nom.reshape(n_buses, 1)
      up = self.Gv_p @ v_abs_del + self.up_nom
      uq = self.Gv_q @ v_abs_del + self.uq_nom

      if self.project:
        up, uq = self.do_projection(up, uq, up_max)

    else:
      up = self.up_nom
      uq = self.uq_nom

    return up, uq

class CtrlIEEE(Controller):
  def __init__(self, u_snom, dt_seconds, filter_tau=300.):
    super().__init__()
    self.n = u_snom.shape[0]
    self.u_snom = u_snom.reshape(self.n, 1)

    # Solution arrays initialization
    self.up = np.zeros(shape=self.u_snom.shape, dtype='float64')
    self.uq = np.zeros(shape=self.u_snom.shape, dtype='float64')

    # IEEE Q points
    self.ar_q1 = self.u_snom * 0.44
    self.ar_q4 = - self.ar_q1
    self.v1_q = 0.9
    self.v4_q = 1.1
    self.v2_q = 0.98
    self.v3_q = 1.02
    self.v1_p = 1.06
    self.v2_p = 1.1

    # IEEE P points
    self.ar_p1 = self.u_snom
    ar_zeros = np.zeros(shape=(self.n, 1), dtype='float64')
    self.ar_p2 = ar_zeros

    # Piece wise linear functions (vmin, intercepts, slopes)
    abs_slope_q = self.ar_q1 / (self.v2_q - self.v1_q)

    self.pwl_fs_q = [
      lambda x, idxer: self.ar_q1[idxer],
      lambda x, idxer: self.v2_q * abs_slope_q[idxer] - abs_slope_q[idxer] * x[idxer],
      lambda x, idxer: ar_zeros[idxer],
      lambda x, idxer: abs_slope_q[idxer] * self.v3_q - abs_slope_q[idxer] * x[idxer],
      lambda x, idxer: self.ar_q4[idxer]
    ]
    abs_slope_p = (self.ar_p1 - self.ar_p2) / (self.v2_p - self.v1_p)
    self.pwl_fs_p = [
      lambda x, idxer: self.ar_p1[idxer],
      lambda x, idxer: (self.v2_p * abs_slope_p[idxer] + self.ar_p2[idxer]
                - abs_slope_p[idxer] * x[idxer]),
      lambda x, idxer: self.ar_p2[idxer]
    ]

    # Previous condition
    self.v_abs_before = np.ones(shape=(self.n, 1), dtype='float64')

    # Linear filter weights
    self.w1 = dt_seconds / (dt_seconds + filter_tau)
    self.w2 = filter_tau / (dt_seconds + filter_tau)

  def filter(self, v_abs):
    return self.w1 * v_abs + self.w2 * self.v_abs_before

  def ret(self, v_abs, up_max, bus_loadp, bus_loadq):
    v_abs_filtered = self.filter(v_abs)
    # OPTI: Avoid creating a list every ret
    l_ar_bool_q = [
      v_abs_filtered < self.v1_q,
      np.logical_and(v_abs_filtered < self.v2_q, v_abs_filtered >= self.v1_q),
      np.logical_and(v_abs_filtered < self.v3_q, v_abs_filtered >= self.v2_q),
      np.logical_and(v_abs_filtered < self.v4_q, v_abs_filtered >= self.v3_q),
      v_abs_filtered >= self.v4_q
    ]

    i = 0
    for idxer in l_ar_bool_q:
      self.uq[idxer.flatten(), 0] = self.pwl_fs_q[i](v_abs_filtered, idxer)
      i += 1

    l_ar_bool_p = [
      v_abs_filtered < self.v1_p,
      np.logical_and(v_abs_filtered < self.v2_p, v_abs_filtered >= self.v1_p),
      v_abs_filtered >= self.v2_p
    ]

    i = 0
    for idxer in l_ar_bool_p:
      self.up[idxer.flatten(), 0] = self.pwl_fs_p[i](v_abs_filtered, idxer)
      i += 1

    # Update
    self.v_abs_before[:] = v_abs_filtered

    # Project
    if self.project:
      self.up, self.uq = self.do_projection(self.up, self.uq, up_max)

    return self.up, self.uq

class CtrlProposed(Controller):
  def __init__(
    self,
    u_snom: np.ndarray,
    polypol_deg: int
  ):

    super().__init__(u_snom)
    self.n = u_snom.shape[0]
    self.up = np.zeros(shape=(self.n, 1), dtype='float64')
    self.uq = np.zeros(shape=(self.n, 1), dtype='float64')

    # General params
    self.l_degs = range(polypol_deg + 1)

    # Feedback and feedforward coefficients
    self.Gv_p = None
    self.Gv_q = None
    self.DL_pp = None
    self.DL_pq = None
    self.DL_qp = None
    self.DL_qq = None
    self.DP_p = None
    self.DP_q = None

    # Nominal values
    self.v_abs_nom = np.ones(shape=(self.n, 1), dtype='float64')
    self.up_nom = None
    self.uq_nom = None
    self.p_nc_mean = None
    self.q_nc_mean = None
    self.p_nc_std = None
    self.q_nc_std = None

    self.up_max_mean = None
    self.up_max_std = None

    self.p_nc_condition = None
    self.q_nc_condition = None
    self.up_max_condition = None

  def update_psi_params(
    self,
    p_nc_mean: np.ndarray,
    p_nc_std: np.ndarray,
    q_nc_mean: np.ndarray,
    q_nc_std: np.ndarray,
    up_max_mean: np.ndarray,
    up_max_std: np.ndarray
  ):
    self.p_nc_mean = p_nc_mean.reshape((self.n, 1))
    self.p_nc_std = p_nc_std.reshape((self.n, 1))
    self.q_nc_mean = q_nc_mean.reshape((self.n, 1))
    self.q_nc_std = q_nc_std.reshape((self.n, 1))
    self.up_max_mean = up_max_mean.reshape((self.n, 1))
    self.up_max_std = up_max_std.reshape((self.n, 1))

    self.p_nc_condition = self.p_nc_std != 0.
    self.q_nc_condition = self.q_nc_std != 0.
    self.up_max_condition = self.up_max_std != 0.

  def scale_p_nc(self, p_nc):
    return np.divide(p_nc - self.p_nc_mean, self.p_nc_std,
             where=self.p_nc_condition, out=np.zeros_like(p_nc))

  def scale_q_nc(self, q_nc):
    return np.divide(q_nc - self.q_nc_mean,  self.q_nc_std,
             where=self.q_nc_condition, out=np.zeros_like(q_nc))

  def scale_up_max(self, up_max):
    return np.divide(up_max - self.up_max_mean, self.up_max_std,
             where=self.up_max_condition, out=np.zeros_like(up_max))

  def update(
    self,
    Gv_p: np.ndarray,
    Gv_q: np.ndarray,
    DL_pp: np.ndarray,
    DL_pq: np.ndarray,
    DL_qp: np.ndarray,
    DL_qq: np.ndarray,
    DP_p: np.ndarray,
    DP_q: np.ndarray,
    up_nom: np.ndarray,
    uq_nom: np.ndarray,
    v_abs_nom: np.ndarray = None
  ):
    # Feedback and feedforward coefficients
    self.Gv_p = Gv_p
    self.Gv_q = Gv_q
    self.DL_pp = DL_pp
    self.DL_pq = DL_pq
    self.DL_qp = DL_qp
    self.DL_qq = DL_qq
    self.DP_p = DP_p
    self.DP_q = DP_q

    # Nominal values
    if v_abs_nom is not None:
      self.v_abs_nom = v_abs_nom

    self.up_nom = up_nom
    self.uq_nom = uq_nom

  def ret(self, v_abs, up_max, p_nc, q_nc):
    # Set controlled bus injections
    ar_p_nc_scaled = self.scale_p_nc(p_nc)
    ar_q_nc_scaled = self.scale_q_nc(q_nc)
    ar_up_max_scaled = self.scale_up_max(up_max)
    self.up[:, [0]] = (
      # Feedback
      self.Gv_p @ (v_abs - self.v_abs_nom)
      # Feedforward
      + sum(
        [
          self.DL_pp[:, :, k] @ ar_p_nc_scaled ** k
          + self.DL_pq[:, :, k] @ ar_q_nc_scaled ** k
          + self.DP_p[:, :, k] @ ar_up_max_scaled ** k
          for k in self.l_degs
        ]
      )
    ) + self.up_nom

    self.uq[:, [0]] = (
      # Feedback
      self.Gv_q @ (v_abs - self.v_abs_nom)
      # Feedforward
      + sum(
        [
          self.DL_qp[:, :, k] @ ar_p_nc_scaled ** k
          + self.DL_qq[:, :, k] @ ar_q_nc_scaled ** k
          + self.DP_q[:, :, k] @ ar_up_max_scaled ** k
          for k in self.l_degs
        ]
      )
    ) + self.uq_nom
    # Project
    if self.project:
      self.up, self.uq = self.do_projection(self.up, self.uq, up_max)

    return self.up, self.uq

Controller.register(CtrlNCognizant)
Controller.register(OPFController)
Controller.register(CtrlIEEE)
Controller.register(CtrlProposed)

class SceGenerator(ABC):
  def __init__(self, df_data: pd.DataFrame, n_sce: int, n_win_ahead: int = None,
         n_days: int = None, n_win_before: int = None, n_days_delay: int = None):
    self.df_data = df_data
    self.l_cols_dgpmax = [i for i in self.df_data.columns if i.startswith(SNAM_DGPMAX)]
    self.l_cols_data_fund = (
      [i for i in self.df_data.columns if i.startswith(SNAM_LOADP)] +
      [i for i in self.df_data.columns if i.startswith(SNAM_LOADQ)] +
      self.l_cols_dgpmax
    )
    self.n_sce = n_sce
    self.df_centroid = pd.DataFrame(index=range(1), columns=self.l_cols_data_fund, dtype='float64')
    self.df_scenarios = pd.DataFrame(index=range(n_sce), columns=self.l_cols_data_fund,
                     dtype='float64')
    self.n_win_ahead = n_win_ahead
    self.n_days = n_days
    self.n_win_before = n_win_before
    self.n_days_delay = n_days_delay

  @abstractmethod
  def run(self, trange_insample, trange_target):
    pass

  def centroid(self):
    self.df_centroid.loc[0, :] = self.df_scenarios.mean()
    return self.df_centroid

# ISSUE: Note that centroid() must be called after calling run(...), try to avoid this
class SceGenShift(SceGenerator):
  """
  Simple shift scenario generator
  """
  def run(self, trange_insample: pd.DatetimeIndex, trange_target: pd.DatetimeIndex = None):
    if trange_insample.shape[0] != self.n_sce:
      raise ValueError('trange_insample is required to have self.n_sce = {} elements'.format(
        self.n_sce))
    self.df_scenarios.loc[:, :] = self.df_data.loc[trange_insample, :].values
    return self.df_scenarios

class SceGenDDUSnaive(SceGenerator):
  DAYS = 3

  def run(self, trange_insample: pd.DatetimeIndex, trange_target: pd.DatetimeIndex):
    assert trange_target.shape[0] * SceGenDDUSnaive.DAYS == self.n_sce
    tini = trange_target[0] - timedelta(days=SceGenDDUSnaive.DAYS)
    if not (tini in self.df_data.index):
      raise ValueError(
        '(trange_target[0] - {} days)={} is not contained in self.df_data'.format(
          self.DAYS, tini))

    dt = trange_target.freq
    if dt is None:
      raise ValueError('Frequency of trange_target is None!')

    idx_insample = trange_target - timedelta(days=1)
    for i in range(1, SceGenDDUSnaive.DAYS):
      idx_insample = idx_insample.union(trange_target - timedelta(days=i + 1))

    self.df_scenarios.loc[:, :] = self.df_data.loc[idx_insample, :].values

    return self.df_scenarios

class SceGenDDUSKmeans(SceGenerator):
  def __init__(self, df_data: pd.DataFrame, n_sce: int, n_win_ahead: int = None,
         n_days: int = None, n_win_before: int = None, n_days_delay: int = None,
         ddus_type: str = None):
    super().__init__(df_data, n_sce, n_win_ahead, n_days, n_win_before, n_days_delay)
    self.ar_prob = None
    self.ddus_type = ddus_type


  def run(self, trange_insample: pd.DatetimeIndex, trange_target: pd.DatetimeIndex):
    # ---------------------------------------------------------------------------------------- #
    # Validate input
    # ---------------------------------------------------------------------------------------- #
    if self.n_days_delay is None:
      raise InputError('Must pass n_days_delay argument!')
    if self.n_days is None:
      raise InputError('Must pass n_days_before argument!')
    if self.n_win_ahead is None:
      raise InputError('Must pass n_win_ahead argument!')
    if self.n_win_before is None:
      raise InputError('Must pass n_win_before argument!')

    tini = trange_target[0] - timedelta(days=self.n_days)
    if not (tini in self.df_data.index):
      raise ValueError(
        '(trange_target[0] - {} days)={} is not contained in self.df_data'.format(
          self.n_days, tini))
    dt = trange_target.freq
    if dt is None:
      raise ValueError('Frequency of trange_target is None!')

    # ---------------------------------------------------------------------------------------- #
    # Construct scenarios
    # ---------------------------------------------------------------------------------------- #
    dt_win = pd.to_timedelta(trange_target.freq * trange_target.shape[0])
    if dt_win >= timedelta(days=1):
      idx_insample = trange_target - dt_win - timedelta(days=self.n_days_delay)
      dt_delay = dt_win
      for j in range(1, self.n_days):
        idx_insample = idx_insample.union(trange_target - dt_delay)
        dt_delay += dt_win

    else:
      idx_insample = trange_target - timedelta(days=self.n_days_delay)
      trange_target_daily = trange_target - timedelta(days=self.n_days_delay)
      for j in range(1, self.n_win_ahead + 1):
        trange_win_ahead = trange_target_daily.shift(len(trange_target_daily) * j)
        idx_insample = idx_insample.union(trange_win_ahead)

      for j in range(1, self.n_win_before + 1):
        trange_win_before = trange_target_daily.shift(-len(trange_target_daily) * j)
        idx_insample = idx_insample.union(trange_win_before)

      trange_target_daily = idx_insample.copy()

      for j in range(1, self.n_days):
        idx_insample = idx_insample.union(trange_target_daily - timedelta(days=j))

    # Check if the insample date ranges are contained in the data
    ar_aux_idx_isin = idx_insample.isin(self.df_data.index)
    if not ar_aux_idx_isin.all():
      raise InputError(
        'Scenario config leads to an insample date range that is not contained in pdata'
      )

    if self.ddus_type is None:
      if idx_insample.shape[0] >= 200000:
        X = self.df_data.loc[idx_insample, self.l_cols_data_fund].resample('T').mean().values
      else:
        X = self.df_data.loc[idx_insample, self.l_cols_data_fund].values
      kmeans_sol = KMeans(n_clusters=self.n_sce, random_state=0).fit(X)
      Y = kmeans_sol.cluster_centers_
      labels, counts = np.unique(kmeans_sol.labels_, return_counts=True)
      total_counts = counts.sum()
      self.ar_prob = counts / total_counts
      self.df_scenarios.loc[:, :] = Y
    elif self.ddus_type == 'minute_based':
      dt_data = (self.df_data.index[1] - self.df_data.index[0])
      n_dt_per_minute = timedelta(minutes=1) // dt_data
      n_total_minutes = idx_insample.shape[0] // n_dt_per_minute
      if n_total_minutes != self.n_sce:
        raise InputError('n_sce is inconsistent with the minute based clustering!')
      ar_minute_group = np.repeat(np.asarray(range(n_total_minutes)), n_dt_per_minute)
      self.df_scenarios.loc[:, :] = self.df_data.loc[idx_insample, self.l_cols_data_fund].groupby(ar_minute_group).mean().values
      self.ar_prob = np.ones((n_total_minutes,), dtype='float64') / n_total_minutes
    elif self.ddus_type == 'all':
      n_sce_all = idx_insample.shape[0]
      if n_sce_all != self.n_sce:
        raise InputError('n_sce is inconsistent with the all based scenario generation!')
      self.df_scenarios.loc[:, :] = self.df_data.loc[idx_insample, self.l_cols_data_fund].values
      self.ar_prob = np.ones((n_sce_all,), dtype='float64') / n_sce_all
    # Clip dgpmax values to be positive
    #   If there is a value of dgpmax slightly lower than zero then it will make the tunning
    #   problem infeasible
    self.df_scenarios.loc[:, self.l_cols_dgpmax] = (
      self.df_scenarios.loc[:, self.l_cols_dgpmax].clip(lower=0.)
    )
    return self.df_scenarios

SceGenerator.register(SceGenShift)
SceGenerator.register(SceGenDDUSnaive)
SceGenerator.register(SceGenDDUSKmeans)

class CCGenerator(ABC):

  def __init__(self):
    pass

class CCGenNaive(CCGenerator):
  def __init__(self):
    super().__init__()
    pass

CCGenerator.register(CCGenNaive)


# FIXME: This assumes that the slack bus is in the first possition
def makeFOT(Ybus, v_abs, v_ang, bus_p, bus_q, v0=1):
  """
  Assuming there are only slack and PQ buses. Let n be the number of PQ buses.
  :param Ybus: {numpy.ndarray} Admitance matrix of the network (n+1 x n+1)
  :param v_abs: {numpy.ndarray} Solution vector of voltage magnitudes for each
  :param v_ang: {numpy.ndarray} Solution vector of voltage angles in radians.
  :param bus_p: {numpy.ndarray} Solution vector of active power injections in p.u.
  :param bus_q: {numpy.ndarray} Solution vector of reactive power injections in p.u.
  :param v0: {float} Slack bus voltage
  :return: {tuple} (K, b) of numpy.ndarrays containing the linear model v = K x + b
  """
  # pflow solution dataframe to vectors
  n = v_abs.shape[0]
  YLL = Ybus[1:, 1:].astype(complex)
  YL0 = Ybus[1:, 0].reshape((Ybus.shape[0] - 1, 1)).astype(complex)
  v = v_abs * np.exp(1j * v_ang)

  x = np.concatenate([bus_p, bus_q])
  # obtaining FOT linear model
  A1 = np.diag(v) @ np.conj(YLL)
  A2 = np.diag((np.conj(YL0 * v0 + YLL @ v.reshape(n, 1))).reshape((n,)))
  A = np.concatenate([
    np.concatenate([np.real(A1) + np.real(A2), np.imag(A1) - np.imag(A2)], axis=1),
    np.concatenate([np.imag(A1) + np.imag(A2), -np.real(A1) + np.real(A2)], axis=1)
  ], axis=0
  )
  U = np.kron(np.eye(2, dtype=int), np.eye(n))
  M = np.linalg.solve(A, U)
  M = M[:n, :] + 1j * M[n:, :]

  K = inv(np.diag(np.abs(v))) @ np.real(np.diag(np.conj(v)) @ M)
  b = v_abs.reshape((n, 1)) - K @ x.reshape((2 * n, 1))

  return K, b


def makeLindistFlow(A_r: np.ndarray, ar_r: np.ndarray, ar_x: np.ndarray):
  assert len(ar_r.shape) == 1
  assert len(ar_x.shape) == 1

  F = inv(A_r)
  R = F @ np.diag(ar_r) @ F.transpose()
  X = F @ np.diag(ar_x) @ F.transpose()

  return R, X


class Mpc:
  def __init__(self, pdata: ds.Adn):
    # Solver config
    self.solver_path_in = None
    self.solver_path_sim = None
    self.mod_file_in = None
    self.mod_in_string = None
    self.mod_file_sim = None
    self.mod_sim_string = None
    self.solver_options_in = None
    self.solver_options_sim = None

    # Pdata aliasing
    self.df_data = pdata.df_data
    self.pdata = pdata
    self.buses = pdata.buses
    self.dgs = pdata.dgs
    self.branches = pdata.branches
    self.l_branches = pdata.branches.index.to_list()
    self.l_buses0 = pdata.l_buses0
    self.l_buses = pdata.l_buses
    self.l_dgs = pdata.l_dgs
    self.l_loads = pdata.l_loads
    self.l_data_cols = pdata.df_data.columns
    self.n_buses = len(self.l_buses)
    self.n_dgs = pdata.dgs.shape[0]
    self.n_loads = len(self.l_loads)
    self.l_cols_v = [SNAM_V + str(i) for i in self.l_buses]
    self.l_cols_vang = [SNAM_VANG + str(i) for i in self.l_buses]
    self.l_cols_dgp = [SNAM_DGP + str(i) for i in self.l_dgs]
    self.l_cols_dgq = [SNAM_DGQ + str(i) for i in self.l_dgs]
    self.l_cols_loadp = [SNAM_LOADP + str(i) for i in self.l_loads]
    self.l_cols_loadq = [SNAM_LOADQ + str(i) for i in self.l_loads]
    self.l_cols_dgpmax = [SNAM_DGPMAX + str(i) for i in self.l_dgs]
    self.l_cols_v_abs_opf = [SNAM_V_OPF + str(i) for i in self.l_buses]

    # Time config
    self.time_map = pdata.time_map(2)
    self.tini = self.time_map[0]
    self.tend = self.time_map[-1]
    self.dt = pdata.time_config['dt']

    # Rolling horizon
    self.n_t = self.time_map.shape[0]
    self.n_rh = pdata.time_config['n_rh']
    assert self.n_t % self.n_rh == 0
    self.n_win = self.n_t // self.n_rh
    self.l_t_sim = range(self.n_rh)
    self.l_t_ins = range(self.n_rh)
    self.n_sce = None
    assert timedelta(days=1) % self.dt == timedelta(0)

    try:
      self.robust = pdata.ctrl_robust
    except KeyError:
      self.robust = False

    if self.robust:
      self.n_sce = pdata.ctrl_robust_n_sce
      self.l_sce = range(self.n_sce)

  def summary(self, dic_args: dict):
    str_report = ''
    str_report += '#' + 79 * '=' + '\nGeneral params\n' + 79 * '=' + '#' + '\n'
    for k in self.pdata.grid_tables['general'].index:
      v = self.pdata.grid_tables['general'].loc[k, 'value']
      if v is not np.nan:
        str_report += '{:15s} {}\n'.format(k, v)

    str_report += '#' + 79 * '=' + '\nTime config\n' + '#' + 79 * '=' + '\n'
    for k, v in self.pdata.time_config.items():
      if v is not None:
        str_report += '{:15s} {}\n'.format(k, v)
        
    str_report += '#' + 79 * '=' + '\n'
    str_report += '# Ampl model and solver config \n'
    str_report += '#' + 79 * '=' + '\n'

    solver_name_ins = self.solver_path_in
    if solver_name_ins is not None:
      solver_name_ins = solver_name_ins.split('/')[-1]

    solver_name_sim = self.solver_path_sim
    if solver_name_sim is not None:
      solver_name_sim = solver_name_sim.split('/')[-1]
    
    model_name_in = self.mod_file_in
    if model_name_in is not None:
      model_name_in = model_name_in.split('/')[-1]

    model_name_sim = self.mod_file_sim
    if model_name_sim is not None:
      model_name_sim = model_name_sim.split('/')[-1]
    
    str_report += '{:20s} {}\n'.format('model_ins', model_name_in)
    str_report += '{:20s} {}\n'.format('solver_ins', solver_name_ins)
    str_report += '{:20s} {}\n'.format('solver_options_ins', self.solver_options_in)
    str_report += '{:20s} {}\n'.format('model_sim', model_name_sim)
    str_report += '{:20s} {}\n'.format('solver_sim', solver_name_sim)
    str_report += '{:20s} {}\n'.format('solver_options_sim', self.solver_options_sim)

    str_report += '#' + 79 * '=' + '\n'
    for k, v in dic_args.items():
      str_report += '{:20s} {}\n'.format(k, v)

    return str_report

  @property
  def ctrl_project(self):
    return self.pdata.ctrl_project

  @property
  def ctrl_active(self):
    return self.pdata.ctrl_active

  @property
  def l_cols_sol(self):
    map_names2sets = {
      SNAM_V: self.l_buses,
      SNAM_VANG: self.l_buses,
      SNAM_DGP: self.l_dgs,
      SNAM_DGQ: self.l_dgs
    }
    col_names = []

    for k, v in map_names2sets.items():
      col_names += [k + str(i) for i in v]

    return col_names

  @staticmethod
  def perfect_predictor(df_win_pred, df_sol, trange):
    df_win_pred.values[:, :] = df_sol.loc[trange, df_win_pred.columns]

  # TODO: Implement sce_generator
  def sce_generator(self, df_win_insample: pd.DataFrame, trange: pd.DatetimeIndex,
            n_dt_delay: int = 0, n_dt_ahead: int = None):
    if n_dt_ahead == None:
      n_dt_ahead = trange.shape[0]
    n_sce = df_win_insample.shape[0]
    trange_win_ahead = trange.shift(n_dt_ahead)
    trange_win_past = trange.shift(-n_dt_delay)
    df_data_avail = self.df_data.loc[trange_win_past, :]

  @staticmethod
  def mean_perfect_filter(df_win_insample, df_data, trange, n_sce=1):
    n_rh = trange.shape[0]
    assert n_rh == n_sce
    # if n_sce >= trange.shape[0]:
    #  raise ValueError('The number of centroids is larger than window')

    df_win_insample.loc[:, :] = df_data.loc[trange, :].values

  def df_data_mean(self, df_win, trange: pd.DatetimeIndex, n_dt_delay: int = 0):
    df_win.loc[:, :] = self.df_data.loc[trange.shift(n_dt_delay), :]

  def realistic_delay_predictor_mean(self, df_win_insample, trange, n_win_ahead=2, past_delay=1):
    n_rh = trange.shape[0]
    trange_win_ahead = trange.union(trange.shift(n_rh))
    n_delay = round(timedelta(days=past_delay) / self.dt)
    trange_win_past = trange_win_ahead.shift(-n_delay)
    df_win_insample.loc[:, :] = self.df_data.loc[trange_win_past, :].mean().values

  def run_acopf(self):
    df_data = self.df_data.loc[self.time_map, :]

    # Calculate hotstart
    sta_opf_hotstart = OPFsto(self.pdata)
    idx_0 = df_data.index[0]
    df_sol_hotstart = sta_opf_hotstart.run(df_data.loc[[idx_0], :])

    # Run acopf
    sta_opf = OPFac(self.pdata)
    df_sim = sta_opf.run(df_data, df_sol_hotstart)

    return df_sim

  def run_perfect_opf(self):
    l_cols_nse = [SNAM_NSE + str(i) for i in self.pdata.l_loads]
    l_cols_sol_opf = self.l_cols_dgp + self.l_cols_dgq + l_cols_nse
    l_cols_sol_v_abs_insample = [SNAM_V_OPF + str(i) for i in self.l_buses]
    l_cols_v_abs = [SNAM_V + str(i) for i in self.l_buses]

    df_sol_opf = pd.DataFrame(
      index=self.time_map,
      columns=l_cols_sol_v_abs_insample + l_cols_sol_opf, dtype='float64')
    sta_opf = OPF(self.pdata)
    sta_pflow = Pflow(self.pdata)
    l_cols_sol_pflow = sta_pflow.l_cols_sol
    df_sol_pflow = pd.DataFrame(index=self.time_map, columns=l_cols_sol_pflow)

    # Rolling horizon config
    tini = self.time_map[0]
    tend = self.time_map[self.n_rh - 1]
    trange = pd.date_range(start=tini, end=tend, freq=self.dt)

    for _ in range(self.n_win):
      df_sol_opf.loc[trange, :] = sta_opf.run(trange=trange).loc[
                    :, l_cols_v_abs + l_cols_sol_opf].values
      trange = trange.shift(self.n_rh)

      #df_sol_pflow.loc[] = sta_pflow.run(self.pdata.df_data.loc[trange, :], df_sol_opf)

    #df_sol = pd.concat([df_sol_pflow, df_sol_opf], axis=1)

    return df_sol_pflow

  def run_noctrl_new(self):
    # Validate input
    if self.dgs.empty:
      raise InputError('No dgs!')

    # ---------------------------------------------------------------------------------------- #
    # Config and initialization
    # ---------------------------------------------------------------------------------------- #
    # Solver config for report
    self.solver_path_in = GUROBI_PATH
    self.mod_file_in = OPF_STO_MODFILE
    self.mod_file_sim = PFLOW_BIM_MODFILE
    self.mod_in_string = None
    self.mod_sim_string = None

    # Scenario generator
    if self.pdata.scegen_type == 'ddus_kmeans':
      sce_gen = SceGenDDUSKmeans(self.df_data, self.n_sce,
                     self.pdata.scegen_n_win_ahead,
                     self.pdata.scegen_n_days,
                     self.pdata.scegen_n_win_before,
                     self.pdata.scegen_n_days_delay)

    elif self.pdata.scegen_type == 'minute_based':
      sce_gen = SceGenDDUSKmeans(self.df_data, self.n_sce,
                     self.pdata.scegen_n_win_ahead,
                     self.pdata.scegen_n_days,
                     self.pdata.scegen_n_win_before,
                     self.pdata.scegen_n_days_delay,
                     ddus_type='minute_based'
                     )
    elif self.pdata.scegen_type == 'all':
      sce_gen = SceGenDDUSKmeans(self.df_data, self.n_sce,
                     self.pdata.scegen_n_win_ahead,
                     self.pdata.scegen_n_days,
                     self.pdata.scegen_n_win_before,
                     self.pdata.scegen_n_days_delay,
                     ddus_type='all'
                     )
    else:
      raise NotImplementedError("""
      scegen_type = {}. Only SceGenDDUSKmeans is supported!""".format(self.pdata.scegen_type))

    # Initialize OPF model
    sta_opf = OPFsto(self.pdata)

    # Construct system matrices
    self.pdata.make_connectivity_mats()
    ar_ybus = self.pdata.ybus.values
    ar_Ar_load = self.pdata.A_load.values[:, 1:]
    ar_Ar_dg_t = self.pdata.A_dg.values[:, 1:].transpose()

    # Initialize controller
    u_snom = ar_Ar_dg_t @ self.dgs.loc[:, ['snom']].values
    ctrl = OPFController(u_snom=u_snom)
    # Initialize containers
    df_win_sim = pd.DataFrame(index=self.l_t_sim, columns=self.l_data_cols, dtype='float64')
    df_sim_sol = pd.DataFrame(index=self.time_map, columns=self.l_cols_sol, dtype='float64')

    dt_win = self.n_rh * self.dt
    idx_insample_sol = pd.date_range(start=self.time_map[0], freq=dt_win, periods=self.n_win)
    l_cols_insample_sol = (self.l_cols_dgp + self.l_cols_dgq + self.l_cols_v + self.l_cols_vang)
    df_insample_sol = pd.DataFrame(index=idx_insample_sol, columns=l_cols_insample_sol)

    df_aux_scenarios = pd.DataFrame(index=self.l_sce, columns=[
      'sto_weights'], dtype='float64')

    # Set time config
    tini = self.time_map[0]
    tend = self.time_map[self.n_rh - 1]
    trange_win = pd.date_range(start=tini, end=tend, freq=self.dt)

    # Set log folder
    log_folder = None
    if 'log_folder' in self.pdata.grid_tables['general'].index:
      log_folder = self.pdata.log_folder
      if log_folder is not None:
        if not os.path.exists(log_folder):
          os.mkdir(log_folder)

    # Set system initial condition
    # FIXME: obtain a point of the manifold given the initial xi
    v_abs_sim = np.ones(shape=(self.n_buses, 1), dtype=float)

    for k in range(self.n_win):
      tini_win = trange_win[0]
      # Create scenarios
      df_win_scenarios = sce_gen.run(trange_win, trange_win)
      df_aux_scenarios.loc[:, 'sto_weights'] = sce_gen.ar_prob

      # Calculate OPF solution
      df_sol_in_opf = sta_opf.run(df_data=df_win_scenarios,
                    df_sto_weights=df_aux_scenarios.loc[:, 'sto_weights'])
      if log_folder:
        df_2stage_vars = sta_opf.ampl.getData('v').toPandas()
        df_2stage_vars.loc[:, 'v'] = np.sqrt(df_2stage_vars.loc[:, 'v'].values)
        df_2stage_vars.rename(columns={'v': SNAM_V},
                    inplace=True)
        df_idx_2stage_vars = pd.DataFrame(zip(*df_2stage_vars.index)).transpose().astype(
          'int64')
        idx_mult_2stage_vars = pd.MultiIndex.from_frame(df_idx_2stage_vars,
                                names=['bus', 'sce'])
        df_2stage_vars.index = idx_mult_2stage_vars
        df_2stage_vars = df_2stage_vars.unstack(level='bus')
        df_2stage_vars.columns = df_2stage_vars.columns.map(
          lambda x: ''.join([str(i) for i in x]))

        df_2stage_vars.to_csv(os.path.join(log_folder, 'df_log_opf_{}.csv'.format(k)),
                    index_label='index')

      ar_dgp_nom = df_sol_in_opf.loc[0, self.l_cols_dgp].values.reshape((self.n_dgs, 1))
      ar_dgq_nom = df_sol_in_opf.loc[0, self.l_cols_dgq].values.reshape((self.n_dgs, 1))
      ar_up_nom = ar_Ar_dg_t @ ar_dgp_nom
      ar_uq_nom = ar_Ar_dg_t @ ar_dgq_nom

      # Save insample solution
      df_insample_sol.loc[tini_win, self.l_cols_dgp] = ar_dgp_nom.flatten()
      df_insample_sol.loc[tini_win, self.l_cols_dgq] = ar_dgq_nom.flatten()

      # Update controller
      ctrl.update(ar_up_nom, ar_uq_nom)

      # Simulate and obtain solution
      df_win_sim.loc[:, :] = self.df_data.loc[trange_win, :].values
      try:
        df_sim_sol.loc[trange_win, :] = self.simulate_controller_new(
          self.pdata, df_win_sim, ctrl, v_abs_sim).values
      except ExperimentError:
        with open('experiment_log.txt', 'w') as hfile:
          hfile.write('Experiment error at simulation level at tini_win {}'.format(
            tini_win))
        exit(1)

      # Shift to next time window
      trange_win = trange_win.shift(self.n_rh)

      # Clean memory
      gc.collect()

    return df_sim_sol, df_insample_sol


  def run_noctrl(self):
    assert not self.dgs.empty

    self.solver_path_in = GUROBI_PATH
    self.mod_file_in = OPF_MODFILE
    self.mod_in_string = None
    self.solver_options_in = 'outlev=1'

    # Config and initialization
    #   Initialize scenario generator
    # Initialize SceGenerator
    try:
      scegen_type = self.pdata.scegen_type
      if scegen_type == 'ddus_kmeans':
        sce_gen = SceGenDDUSKmeans(self.df_data, self.n_sce)
      else:
        sce_gen = SceGenDDUSnaive(self.df_data, self.n_sce)
    except KeyError:
      sce_gen = SceGenDDUSnaive(self.df_data, self.n_sce)
    #   Initialize controller

    ctrl = OPFController()
    ctrl.project = self.ctrl_project

    #   Initialize data containers
    df_win_insample_nom = pd.DataFrame(index=range(1), columns=self.l_data_cols,
                       dtype='float64')
    df_win_sim = pd.DataFrame(index=self.l_t_sim, columns=self.l_data_cols,
                  dtype='float64')

    #   Initialize solution containers
    df_sim_sol = pd.DataFrame(index=self.time_map,
                  columns=self.l_cols_sol,
                  dtype='float64')

    #   System initial condition
    v_abs_sim = np.ones(shape=(self.n_buses, 1), dtype=float)

    pdata_bucle = self.pdata.grid_pointer()
    pdata_bucle.df_data = self.df_data
    pdata_bucle.make_connectivity_mats()
    A_dg_transpose = pdata_bucle.A_dg.transpose().values[1:]
    ctrl.u_snom = (A_dg_transpose @ pdata_bucle.dgs.loc[:, 'snom'].values).reshape(
      self.n_buses, 1)


    tini = self.time_map[0]
    tend = self.time_map[self.n_rh - 1]
    trange = pd.date_range(start=tini, end=tend, freq=self.dt)

    assert ctrl.is_valid()
    for k in range(self.n_win):
      # Predict uncertain parameters
      if isinstance(sce_gen, SceGenDDUSKmeans):
        df_win_insample = sce_gen.run(trange + timedelta(days=1), trange + timedelta(days=1))
      else:
        df_win_insample = sce_gen.run(trange, trange)

      df_win_insample_nom = sce_gen.centroid()
      pdata_bucle.df_data = df_win_insample_nom

      # Calculate operational setpoint (solve OPF)
      df_sol_in_opf = opf(pdata_bucle, GUROBI_PATH, solver_options='outlev=1', tmap_mode=1)
      dgp_nom = df_sol_in_opf.loc[0, self.l_cols_dgp].values.reshape(self.n_dgs, 1)
      dgq_nom = df_sol_in_opf.loc[0, self.l_cols_dgq].values.reshape(self.n_dgs, 1)
      up_nom = (A_dg_transpose @ dgp_nom)
      uq_nom = (A_dg_transpose @ dgq_nom)

      ctrl.update(up_nom, uq_nom)

      df_win_sim.loc[:, :] = self.df_data.loc[trange, :].values
      pdata_bucle.df_data = df_win_sim
      df_sim_sol.loc[trange, :] = self.simulate_controller(
        pdata_bucle, df_win_sim, ctrl, v_abs_sim).values

      trange = trange.shift(self.n_rh)

    return df_sim_sol, None

  @staticmethod
  def __l_cols_D_to_multidx(l_cols_D):
    n = len(l_cols_D)
    l_ret = [None] * n
    i = 0
    for nam in l_cols_D:
      l_nam = nam.split('_')
      l_ret[i] = (l_nam[0], int(l_nam[1]), int(l_nam[2]))
      i += 1
    return l_ret

  def __set_hotstart_to_zero(self, ampl, set_buses_dgs, map_lbuses2idxar, ar_p_nc_condition,
                 ar_q_nc_condition, ar_up_max_condition):
    # ------------------------------------------------------------------------------------ #
    # Set Hotstart to nominal setpoint (deltas -> 0.)
    # ------------------------------------------------------------------------------------ #
    ampl.getVariable('eps').setValue(0.)
    if self.pdata.risk_measure == 'worst_case':
      ampl.getVariable('epi_ub').setValue(0.)
    if self.pdata.bilinear_approx == 'bilinear':
      for i in self.l_buses:
        for t in self.l_sce:
          ampl.getVariable('dv')[i, t].setValue(0.)
    elif (self.pdata.bilinear_approx == 'mccormick' or
        self.pdata.bilinear_approx == 'neuman'):
      for i in self.l_buses:
        for t in self.l_sce:
          ampl.getVariable('dv_pos')[i, t].setValue(0.)
          ampl.getVariable('dv_neg')[i, t].setValue(0.)
    else:
      raise NotSupportedYet('bilinear_approx: {} is not supported!'.format(
        self.pdata.bilinear_approx))

    for i in set_buses_dgs:
      ampl.getVariable('gp')[i].setValue(0.)
      ampl.getVariable('gq')[i].setValue(0.)
      for t in self.l_sce:
        ampl.getVariable('dup')[i, t].setValue(0.)
        ampl.getVariable('duq')[i, t].setValue(0.)

    for i in self.l_buses:
      for t in self.l_sce:
        ampl.getVariable('e_pos')[i, t].setValue(0.)
        ampl.getVariable('e_neg')[i, t].setValue(0.)

    if self.pdata.ctrl_type == 'droop_polypol':
      for i in set_buses_dgs:
        j = map_lbuses2idxar[i]
        for kk in range(self.pdata.polypol_deg + 1):
          if ar_p_nc_condition[j, 0]:
            ampl.getVariable('DL_pp_pos')[i, kk].setValue(0.)
            ampl.getVariable('DL_pp_neg')[i, kk].setValue(0.)
            ampl.getVariable('DL_qp_pos')[i, kk].setValue(0.)
            ampl.getVariable('DL_qp_neg')[i, kk].setValue(0.)
          if ar_q_nc_condition[j, 0]:
            ampl.getVariable('DL_pq_pos')[i, kk].setValue(0.)
            ampl.getVariable('DL_pq_neg')[i, kk].setValue(0.)
            ampl.getVariable('DL_qq_pos')[i, kk].setValue(0.)
            ampl.getVariable('DL_qq_neg')[i, kk].setValue(0.)
          if ar_up_max_condition[j, 0]:
            ampl.getVariable('DP_p_pos')[i, kk].setValue(0.)
            ampl.getVariable('DP_p_neg')[i, kk].setValue(0.)
            ampl.getVariable('DP_q_pos')[i, kk].setValue(0.)
            ampl.getVariable('DP_q_neg')[i, kk].setValue(0.)

  def run(self):
    # Config and initialization
    self.solver_path_in = None
    self.solver_path_sim = KNITRO_PATH  # IPOPT_PATH
    self.mod_in_string = None
    self.mod_sim_string = None
    self.solver_options_in = 'outlev=1'  # SOLVER_OPTIONS_KNITRO
    self.solver_options_sim = 'outlev=0'
    self.mod_file_in = None

    

    """ ------------------------------------------------------------------------------------ """
    """ ########################## Initialize controller setter ############################ """
    """ ------------------------------------------------------------------------------------ """
    # Configure model declaration
    try:
      if self.pdata.bilinear_approx == 'bilinear':
        self.mod_file_in = MODFILE_TOY_BILINEAR
        self.solver_path_in = KNITRO_PATH  # IPOPT_PATH
        self.solver_options_in = 'outlev=3' # SOLVER_OPTIONS_KNITRO_SIMPLE

        if self.pdata.risk_measure == 'expected_value':
          if self.pdata.ctrl_type == 'droop_polypol':
            self.mod_in_string = MODSTRING_TOY_STO_DROOPPOLY_BILINEAR
          elif self.pdata.ctrl_type == 'droop':
            self.mod_in_string = (
              """
              FOBJ_STOCHASTIC,VOLTAGE_LB2,VOLTAGE_UB2,LINEAR_PFLOW,STABILITY,
              PROPOSED_CONTROL_P_BILINEAR,PROPOSED_CONTROL_Q_BILINEAR,e_pos,
              e_neg,eps,dv,dup,duq,gp,gq,PROPOSED_PMAX,PROPOSED_SMAX,PROPOSED_PMIN
              """
            )
          else:
            raise InputError("Invalid general configuration parameter: ctrl_type")
        elif self.pdata.risk_measure == 'worst_case':
          if self.pdata.ctrl_type == 'droop_polypol':
            self.mod_in_string = (
              """
              FOBJ_ROBUST,EPIGRAPH_UB,LINEAR_PFLOW,VOLTAGE_UB2,VOLTAGE_LB2,
              STABILITY,PROPOSED_CONTROL_P_BILINEAR,PROPOSED_CONTROL_Q_BILINEAR,
              PROPOSED_SMAX,PROPOSED_PMAX,PROPOSED_PMIN,epi_ub,e_pos,e_neg,eps,dv,dup,duq,gp,gq,
              DL_pp_pos,DL_pq_pos,DL_qp_pos,DL_qq_pos,DL_pp_neg,DL_pq_neg,DL_qp_neg,
              DL_qq_neg,DP_p_pos,DP_q_pos,DP_p_neg,DP_q_neg
              """
            )
          elif self.pdata.ctrl_type == 'droop':
            self.mod_in_string = (
              """
              FOBJ_ROBUST,EPIGRAPH_UB,VOLTAGE_LB2,VOLTAGE_UB2,LINEAR_PFLOW,
              STABILITY,PROPOSED_CONTROL_P_BILINEAR,PROPOSED_CONTROL_Q_BILINEAR,
              PROPOSED_SMAX,PROPOSED_PMAX,PROPOSED_PMIN,epi_ub,e_pos,e_neg,eps,dv,dup,duq,gp,gq
              """
            )
          else:
            raise InputError("Invalid general configuration parameter: ctrl_type")
      elif self.pdata.bilinear_approx == 'mccormick':
        self.mod_file_in = MODFILE_TOY_MCCORMICK
        self.solver_path_in = GUROBI_PATH
        if 'opt_gurobi' in self.pdata.grid_tables['general'].index:
          gurobi_options = self.pdata.opt_gurobi
        else:
          gurobi_options = GUROBI_OPTIONS_DEFAULT
        self.solver_options_in = gurobi_options

        if self.pdata.risk_measure == 'worst_case':
          if self.pdata.ctrl_type == 'droop_polypol':
            self.mod_in_string = MODSTRING_TOY_ROB_DROOPPOLY_MCCORMICK
          elif self.pdata.ctrl_type == 'droop':
            self.mod_in_string = MODSTRING_TOY_ROB_DROOP_MCCORMICK
        elif self.pdata.risk_measure == 'expected_value':
          if self.pdata.ctrl_type == 'droop_polypol':
            self.mod_in_string = MODSTRING_TOY_STO_DROOPPOLY_MCCORMICK
          elif self.pdata.ctrl_type == 'droop':
            self.mod_in_string = MODSTRING_TOY_STO_DROOP_MCCORMICK

      elif self.pdata.bilinear_approx == 'neuman':
        self.mod_file_in = MODFILE_TOY_MCCORMICK
        self.solver_path_in = GUROBI_PATH
        if 'opt_gurobi' in self.pdata.grid_tables['general'].index:
          gurobi_options = self.pdata.opt_gurobi
        else:
          gurobi_options = GUROBI_OPTIONS_DEFAULT
        self.solver_options_in = gurobi_options

        if self.pdata.ctrl_type == 'droop':
          if self.pdata.risk_measure == 'worst_case':
            self.mod_in_string = MODSTRING_TOY_ROB_DROOP_NEUMAN
          elif self.pdata.risk_measure == 'expected_value':
            raise NotSupportedYet('expected_value risk measure is not implemented for '
                        'bilinear_approx=neuman yet!')
        else:
          raise InputError('Neuman bilinear approx is only compatible with '
                   'ctrl_type = droop!')

      elif self.pdata.bilinear_approx == 'bilinear_neuman':
        self.mod_file_in = MODFILE_TOY_BILINEAR
        self.solver_path_in = KNITRO_PATH
        self.solver_options_in = 'outlev=1'
        if self.pdata.ctrl_type == 'droop':
          if self.pdata.risk_measure == 'worst_case':
            self.mod_in_string = MODSTRING_TOY_ROB_DROOP_BILINEAR_NEUMAN
          else:
            raise NotSupportedYet('Stochastic bilinear neuman is not supported!')

        else:
          raise InputError(
            'incompatible control type with bilinear_approx=bilinear_neuman!')

      else:
        raise InputError('Invalid bilinear_approx value!')
    except KeyError:
      raise InputError('risk_measure is not defined in the general parameters')

    # Setting DG mode
    if 'ctrl_tunning_dg_mode_v2' in self.pdata.grid_tables['general'].index:
      str_dgmode = self.pdata.ctrl_tunning_dg_mode_v2
      if str_dgmode == 'free':
        self.mod_in_string = self.mod_in_string.replace(',PROPOSED_SMAX', '')
        self.mod_in_string = self.mod_in_string.replace(',PROPOSED_PMAX', '')

    # Initialize objective function name
    if self.pdata.risk_measure == 'expected_value':
      fobj_name = 'FOBJ_STOCHASTIC'
    elif self.pdata.risk_measure == 'worst_case':
      fobj_name = 'FOBJ_ROBUST'
    else:
      raise InputError('Invalid general param risk_measure!')

    
    
    ampl = init_ampl_and_solver(AMPL_FOLDER, self.solver_path_in, self.solver_options_in)
    ampl.read(self.mod_file_in)
    ampl.eval('problem tunning: {};'.format(self.mod_in_string))

    

    # Initialize sets & params
    # Initialize auxiliary containers
    df_aux_scenarios = pd.DataFrame(index=self.l_sce, columns=[
      'sto_weights'], dtype='float64')
    df_aux_buses = pd.DataFrame(index=self.l_buses, columns=[
      'dv_lb', 'dv_ub', 'gp_lb', 'gq_lb', 'up_mean', 'uq_mean',
      'p_mean', 'q_mean', 'smax'], dtype='float64')

    idx_mul_buses_periods = pd.MultiIndex.from_product(
      [self.l_buses, list(self.l_sce)], names=['BUSES', 'PERIODS'])
    df_aux_buses_periods = pd.DataFrame(
      0.,
      index=idx_mul_buses_periods,
      columns=['dp_nc', 'dq_nc', 'up_max', 'dp_nc_scaled', 'dq_nc_scaled', 'up_max_scaled'],
      dtype='float64'
    )
    
    
    idx_mul_links = pd.MultiIndex.from_product([self.l_buses, self.l_buses],
                           names=['BUSES', 'BUSES'])

    # OPTI: Utilize sparse dataframes for link related params
    df_aux_links = pd.DataFrame(index=idx_mul_links, columns=['R', 'X'])
    df_aux_links.index.name = 'LINKS'
    # Sets
    ampl.getSet('BUSES').setValues(self.l_buses)
    ampl.getSet('PERIODS').setValues(list(self.l_sce))
    ampl.getSet('LINKS').setValues(idx_mul_links.to_list())

    # General params
    ampl.getParameter('cost_stability').set(self.pdata.cost_stability)
    ampl.getParameter('cost_lasso_v').set(self.pdata.cost_lasso_v)
    ampl.getParameter('cost_lasso_x').set(self.pdata.cost_lasso_x)
    ampl.getParameter('cost_putility').set(self.pdata.cost_putility)
    ampl.getParameter('cost_losses').set(self.pdata.cost_losses)
    ampl.getParameter('cost_vlim').set(self.pdata.cost_vlim)

    n_degs = self.pdata.polypol_deg
    ampl.getParameter('n_deg').set(n_degs)
    
    # Buses indexed params (fill df_aux_buses)
    df_aux_buses.loc[:, 'v_lb'] = self.pdata.buses.loc[:, 'vmin']
    df_aux_buses.loc[:, 'v_ub'] = self.pdata.buses.loc[:, 'vmax']


    map_lbuses2idxar = dict(zip(self.l_buses, range(self.n_buses)))
    set_buses_dgs = set(self.dgs['bus'].to_list())
    set_buses_no_dgs = set(self.l_buses) - set_buses_dgs

    if self.pdata.bilinear_approx == 'mccormick':
      df_aux_buses.loc[:, 'dv_lb'] = self.pdata.buses.loc[:, 'mc_dv_lb']
      df_aux_buses.loc[:, 'dv_ub'] = self.pdata.buses.loc[:, 'mc_dv_ub']
      df_aux_buses.loc[:, 'gp_lb'] = self.pdata.buses.loc[:, 'mc_gp_lb']
      df_aux_buses.loc[:, 'gq_lb'] = self.pdata.buses.loc[:, 'mc_gq_lb']

      for i in set_buses_no_dgs:
        for t in self.l_sce:
          ampl.getVariable('k_p_pos')[i, t].fix(0.)
          ampl.getVariable('k_p_neg')[i, t].fix(0.)
          ampl.getVariable('k_q_pos')[i, t].fix(0.)
          ampl.getVariable('k_q_neg')[i, t].fix(0.)

    
    # Fix variables for buses without controlled DGS
    for i in set_buses_no_dgs:
      ampl.getVariable('gp')[i].fix(0.)
      ampl.getVariable('gq')[i].fix(0.)
      for t in self.l_sce:
        ampl.getVariable('dup')[i, t].fix(0.)
        ampl.getVariable('duq')[i, t].fix(0.)

      if self.pdata.ctrl_type == 'droop_polypol':
        for k in range(self.pdata.polypol_deg + 1):
          ampl.getVariable('DL_pp_pos')[i, k].fix(0.)
          ampl.getVariable('DL_pq_pos')[i, k].fix(0.)
          ampl.getVariable('DL_qp_pos')[i, k].fix(0.)
          ampl.getVariable('DL_qq_pos')[i, k].fix(0.)
          ampl.getVariable('DL_pp_neg')[i, k].fix(0.)
          ampl.getVariable('DL_pq_neg')[i, k].fix(0.)
          ampl.getVariable('DL_qp_neg')[i, k].fix(0.)
          ampl.getVariable('DL_qq_neg')[i, k].fix(0.)

          ampl.getVariable('DP_p_pos')[i, k].fix(0.)
          ampl.getVariable('DP_q_pos')[i, k].fix(0.)
          ampl.getVariable('DP_p_neg')[i, k].fix(0.)
          ampl.getVariable('DP_q_neg')[i, k].fix(0.)
    
    """ ------------------------------------------------------------------------------------ """
    """ ################################# Rolling horizon ################################## """
    """ ------------------------------------------------------------------------------------ """
    # Set system initial condition
    # FIXME: obtain a point of the manifold given the initial xi
    v_abs_sim = np.ones(shape=(self.n_buses, 1), dtype=float)

    # Construct system matrices
    self.pdata.make_connectivity_mats()
    ar_ybus = self.pdata.ybus.values
    ar_Ar_load = self.pdata.A_load.values[:, 1:]
    ar_Ar_dg = self.pdata.A_dg.values[:, 1:]

    # Set time config
    tini = self.time_map[0]
    tend = self.time_map[self.n_rh - 1]
    trange_win = pd.date_range(start=tini, end=tend, freq=self.dt)

    # Initialize nominal setpoint calculator
    sta_opf = OPFsto(self.pdata)
    sta_pflow = Pflow(self.pdata)

    df_win_sim = pd.DataFrame(index=self.l_t_sim, columns=self.l_data_cols, dtype='float64')
    df_sim_sol = pd.DataFrame(index=self.time_map, columns=self.l_cols_sol, dtype='float64')

    dt_win = self.n_rh * self.dt
    idx_insample_sol = pd.date_range(start=self.time_map[0], freq=dt_win, periods=self.n_win)

    l_cols_gpv = [SNAM_GPV + str(i) for i in self.l_buses]
    l_cols_gqv = [SNAM_GQV + str(i) for i in self.l_buses]
    """ SNAM_DLPP = 'DLpp'
      SNAM_DLPQ = 'DLpq'
      SNAM_DLQP = 'DLqp'
      SNAM_DLQQ = 'DLqq'
      SNAM_DP = 'DPp'
      SNAM_DP = 'DPq'
    """
    l_cols_DL_pp = [SNAM_DLPP + '_' + '_'.join([str(ii), str(jj)])
            for ii in self.l_buses for jj in range(n_degs + 1)]
    l_cols_DL_pq = [SNAM_DLPQ + '_' + '_'.join([str(ii), str(jj)])
            for ii in self.l_buses for jj in range(n_degs + 1)]
    l_cols_DL_qp = [SNAM_DLQP + '_' + '_'.join([str(ii), str(jj)])
            for ii in self.l_buses for jj in range(n_degs + 1)]
    l_cols_DL_qq = [SNAM_DLQQ + '_' + '_'.join([str(ii), str(jj)])
            for ii in self.l_buses for jj in range(n_degs + 1)]
    l_cols_DP_p = [SNAM_DPP + '_' + '_'.join([str(ii), str(jj)])
            for ii in self.l_buses for jj in range(n_degs + 1)]
    l_cols_DP_q = [SNAM_DPQ + '_' + '_'.join([str(ii), str(jj)])
            for ii in self.l_buses for jj in range(n_degs + 1)]

    l_cols_D = (l_cols_DL_pp + l_cols_DL_pq + l_cols_DL_qp + l_cols_DL_qq + l_cols_DP_p +
          l_cols_DP_q)

    l_cols_data_basic = self.l_cols_loadp + self.l_cols_loadq + self.l_cols_dgpmax


    
    l_cols_fobj2stage = [SNAM_FOBJ_EXP, SNAM_FOBJ_CVAR, SNAM_FOBJ_MAX]
    l_cols_insample_sol = (self.l_cols_v + self.l_cols_vang + self.l_cols_dgp + self.l_cols_dgq
                 + l_cols_gpv + l_cols_gqv + l_cols_D + l_cols_fobj2stage)
    df_insample_sol = pd.DataFrame(index=idx_insample_sol, columns=l_cols_insample_sol)

    # Initialize controller:
    u_snom = ar_Ar_dg.transpose() @ self.dgs.loc[:, ['snom']].values
    df_aux_buses.loc[:, 'smax'] = u_snom

    # Initialize CtrlProposed
    ctrl = CtrlProposed(u_snom, self.pdata.polypol_deg)

    # Initialize SceGenerator
    # sce_gen = SceGenShift(self.df_data, self.n_sce)

    if self.pdata.scegen_type == 'ddus_kmeans':
      sce_gen = SceGenDDUSKmeans(self.df_data, self.n_sce,
                     self.pdata.scegen_n_win_ahead,
                     self.pdata.scegen_n_days,
                     self.pdata.scegen_n_win_before,
                     self.pdata.scegen_n_days_delay)

    elif self.pdata.scegen_type == 'minute_based':
      sce_gen = SceGenDDUSKmeans(self.df_data, self.n_sce,
                     self.pdata.scegen_n_win_ahead,
                     self.pdata.scegen_n_days,
                     self.pdata.scegen_n_win_before,
                     self.pdata.scegen_n_days_delay,
                     ddus_type='minute_based'
                     )
    elif self.pdata.scegen_type == 'all':
      sce_gen = SceGenDDUSKmeans(self.df_data, self.n_sce,
                     self.pdata.scegen_n_win_ahead,
                     self.pdata.scegen_n_days,
                     self.pdata.scegen_n_win_before,
                     self.pdata.scegen_n_days_delay,
                     ddus_type='all'
                     )
    else:
      raise NotImplementedError("""
      scegen_type = {}. Only SceGenDDUSKmeans is supported!""".format(self.pdata.scegen_type))

    # Initialize ctrl.matrices
    ar_DL_pp = np.zeros(shape=(self.n_buses, self.n_buses, n_degs + 1), dtype='float64')
    ar_DL_pq = np.zeros(shape=(self.n_buses, self.n_buses, n_degs + 1), dtype='float64')
    ar_DL_qp = np.zeros(shape=(self.n_buses, self.n_buses, n_degs + 1), dtype='float64')
    ar_DL_qq = np.zeros(shape=(self.n_buses, self.n_buses, n_degs + 1), dtype='float64')

    ar_DP_p = np.zeros(shape=(self.n_buses, self.n_buses, n_degs + 1), dtype='float64')
    ar_DP_q = np.zeros(shape=(self.n_buses, self.n_buses, n_degs + 1), dtype='float64')

    # Initialize insample dump folder log_folder
    log_folder = None
    if 'log_folder' in self.pdata.grid_tables['general'].index:
      log_folder = self.pdata.log_folder
      if log_folder is not None:
        if not os.path.exists(log_folder):
          os.mkdir(log_folder)
        ampl.eval('param fobj2stage {PERIODS};')
        ampl.eval('param pinjection {PERIODS};')
        ampl.eval('param losses {PERIODS};')
  
    df_ada_weights_inv = None
    if self.pdata.ctrl_type == 'droop_polypol':
      if 'fn_adalasso_weights' in self.pdata.grid_tables['general'].index:
        fn_ada = self.pdata.fn_adalasso_weights
        if fn_ada is not None:
          map_adalasso_weights = dict(zip(
            ['DLpp', 'DLpq', 'DLqp', 'DLqq', 'DPp', 'DPq'],
            ['w_DL_pp', 'w_DL_pq', 'w_DL_qp', 'w_DL_qq', 'w_DP_p', 'w_DP_q']
          ))


          df_ada_weights_inv = load_df_result(fn_ada)
          df_ada_weights_inv = df_ada_weights_inv.loc[:, l_cols_D]
          l_tpls = self.__l_cols_D_to_multidx(l_cols_D)

          idx_mul = pd.MultiIndex.from_tuples(l_tpls, names=['name', 'BUSES', 'DEGS'])
          df_ada_weights_inv.columns = idx_mul
          df_ada_weights_inv = df_ada_weights_inv.stack(level=['BUSES', 'DEGS'])
          df_ada_weights_inv.rename(columns=map_adalasso_weights, inplace=True)

    # fix voltage bounds initially
    #   They are unfixed only if the fixed model is unfeasible
    if self.pdata.bilinear_approx == 'bilinear':
      unfeasibility_happend = False
      for i in self.l_buses:
        for t in self.l_sce:
          ampl.getVariable('e_pos')[i, t].setValue(0.)
          ampl.getVariable('e_neg')[i, t].setValue(0.)
          ampl.getVariable('e_pos')[i, t].fix()
          ampl.getVariable('e_neg')[i, t].fix()

    ar_p_nc_condition = None
    ar_q_nc_condition = None
    ar_up_max_condition = None

    for k in range(self.n_win):
      tini_win = trange_win[0]
      # Predict uncertain parameters (perfect mean)
      df_win_scenarios = sce_gen.run(trange_win, trange_win)
      df_win_centroid = sce_gen.centroid()
      df_aux_scenarios.loc[:, 'sto_weights'] = sce_gen.ar_prob

      # Set adaLasso weights
      if df_ada_weights_inv is not None:
        idx_ada_weights_inv_nonzero = df_ada_weights_inv.loc[(tini_win), :].abs() > EPS_FLOAT_ADALASSO
        df_ada_weights = 1 / df_ada_weights_inv.loc[(tini_win), :][idx_ada_weights_inv_nonzero]
        df_ada_weights[~idx_ada_weights_inv_nonzero] = 1. / EPS_FLOAT_ADALASSO
        df_ada_weights = df_ada_weights.abs()
        df_ada_weights_ampl = DataFrame.fromPandas(df_ada_weights)
        ampl.setData(df_ada_weights_ampl)

      # Calculate operational setpoint
      print('aaaaa1')
      df_sol_in_opf = sta_opf.run(df_data=df_win_scenarios,
                    df_sto_weights=df_aux_scenarios.loc[:, 'sto_weights'])
      print('aaaaa2')
      df_sol_in_pflow = sta_pflow.run(df_win_scenarios, df_sol_in_opf)

      v_abs = df_sol_in_pflow.loc[:, self.l_cols_v].mean().values

      # Setting v_set in droop_polypol model
      df_aux_buses.loc[:, 'v_set'] = v_abs

      ar_v_abs_nom = v_abs.reshape(self.n_buses, 1)
      if k == 0:
        v_abs_sim = ar_v_abs_nom
      v_ang = df_sol_in_pflow.loc[0, self.l_cols_vang].values

      # Nominal p, q injection
      ar_loadp_nom = df_win_centroid.loc[[0], self.l_cols_loadp].values
      ar_loadq_nom = df_win_centroid.loc[[0], self.l_cols_loadq].values
      ar_p_nc_nom = - np.transpose(ar_loadp_nom @ ar_Ar_load)
      ar_q_nc_nom = - np.transpose(ar_loadq_nom @ ar_Ar_load)

      ar_dgp_nom = df_sol_in_opf.loc[[0], self.l_cols_dgp].values.astype('float64')
      ar_dgq_nom = df_sol_in_opf.loc[[0], self.l_cols_dgq].values.astype('float64')
      ar_up_nom = np.transpose(ar_dgp_nom @ ar_Ar_dg)
      ar_uq_nom = np.transpose(ar_dgq_nom @ ar_Ar_dg)

      ar_busp_nom = ar_up_nom + ar_p_nc_nom
      ar_busq_nom = ar_uq_nom + ar_q_nc_nom
      df_aux_buses.loc[:, 'p_mean'] = ar_busp_nom
      df_aux_buses.loc[:, 'q_mean'] = ar_busq_nom
      df_aux_buses.loc[:, 'up_mean'] = ar_up_nom
      df_aux_buses.loc[:, 'uq_mean'] = ar_uq_nom

      # Saving nominal set-point
      df_insample_sol.loc[tini_win, self.l_cols_v] = v_abs
      df_insample_sol.loc[tini_win, self.l_cols_vang] = v_ang
      df_insample_sol.loc[tini_win, self.l_cols_dgp] = ar_dgp_nom
      df_insample_sol.loc[tini_win, self.l_cols_dgq] = ar_dgq_nom

      # Construct linear model
      H, a = makeFOT(ar_ybus, v_abs, v_ang, ar_busp_nom.flatten(),
               ar_busq_nom.flatten())
      R = H[:, :self.n_buses]
      X = H[:, self.n_buses:]
      df_aux_links.loc[:, 'R'] = R.flatten()
      df_aux_links.loc[:, 'X'] = X.flatten()

      # OPTI: Avoid construction of ampl.DataFrame at each window
      df_links = DataFrame.fromPandas(df_aux_links)
      ampl.setData(df_links)

      # Set buses x periods data: dp_nc, dq_nc, up_max
      ar_dgpmax = df_win_scenarios.loc[:, self.l_cols_dgpmax].values
      ar_loadp = df_win_scenarios.loc[:, self.l_cols_loadp].values
      ar_loadq = df_win_scenarios.loc[:, self.l_cols_loadq].values
      ar_up_max = np.transpose(ar_dgpmax @ ar_Ar_dg)
      ar_p_nc = np.transpose(- ar_loadp @ ar_Ar_load)
      ar_q_nc = np.transpose(- ar_loadq @ ar_Ar_load)

      ar_p_nc_del = ar_p_nc - ar_p_nc_nom
      ar_q_nc_del = ar_q_nc - ar_q_nc_nom

      ar_p_nc_del_std = ar_p_nc_del.std(axis=1)
      ar_q_nc_del_std = ar_q_nc_del.std(axis=1)

      df_aux_buses_periods.loc[:, 'up_max'] = ar_up_max.flatten()
      df_aux_buses_periods.loc[:, 'dp_nc'] = ar_p_nc_del.flatten()
      df_aux_buses_periods.loc[:, 'dq_nc'] = ar_q_nc_del.flatten()

      ar_up_max_mean = ar_up_max.mean(axis=1)
      ar_up_max_std = ar_up_max.std(axis=1)

      # Update ctrl psi scaling parameters

      ctrl.update_psi_params(ar_p_nc_nom, ar_p_nc_del_std, ar_q_nc_nom, ar_q_nc_del_std,
                   ar_up_max_mean, ar_up_max_std)
      ar_p_nc_scaled = ctrl.scale_p_nc(ar_p_nc)
      ar_q_nc_scaled = ctrl.scale_q_nc(ar_q_nc)
      ar_up_max_scaled = ctrl.scale_up_max(ar_up_max)
      df_aux_buses_periods.loc[:, 'dp_nc_scaled'] = ar_p_nc_scaled.flatten()
      df_aux_buses_periods.loc[:, 'dq_nc_scaled'] = ar_q_nc_scaled.flatten()
      df_aux_buses_periods.loc[:, 'up_max_scaled'] = ar_up_max_scaled.flatten()

      # Fix ctrl coefficient variables for null scaled signals

      if self.pdata.ctrl_type == 'droop_polypol':
        ar_p_nc_condition = ctrl.p_nc_condition
        ar_q_nc_condition = ctrl.q_nc_condition
        ar_up_max_condition = ctrl.up_max_condition
        # OPTI: Avoid in-loop calls to ampl.getVariable(), instead call it aoutside the
        #  loop and assign the result to variables

        for i in set_buses_dgs:
          j = map_lbuses2idxar[i]
          for kk in range(self.pdata.polypol_deg + 1):
            if not ar_p_nc_condition[j, 0]:
              ampl.getVariable('DL_pp_pos')[i, kk].fix(0.)
              ampl.getVariable('DL_pp_neg')[i, kk].fix(0.)
              ampl.getVariable('DL_qp_pos')[i, kk].fix(0.)
              ampl.getVariable('DL_qp_neg')[i, kk].fix(0.)
            if not ar_q_nc_condition[j, 0]:
              ampl.getVariable('DL_pq_pos')[i, kk].fix(0.)
              ampl.getVariable('DL_pq_neg')[i, kk].fix(0.)
              ampl.getVariable('DL_qq_pos')[i, kk].fix(0.)
              ampl.getVariable('DL_qq_neg')[i, kk].fix(0.)
            if not ar_up_max_condition[j, 0]:
              ampl.getVariable('DP_p_pos')[i, kk].fix(0.)
              ampl.getVariable('DP_p_neg')[i, kk].fix(0.)
              ampl.getVariable('DP_q_pos')[i, kk].fix(0.)
              ampl.getVariable('DP_q_neg')[i, kk].fix(0.)

      # Update mccormick bounds
      if self.pdata.bilinear_approx == 'mccormick':
        ar_v_ub_mc = np.minimum(
          self.pdata.buses.loc[self.l_buses, 'vmax'] - v_abs,
          self.pdata.buses.loc[self.l_buses, 'mc_dv_ub']
        )
        ar_v_lb_mc = np.maximum(
          self.pdata.buses.loc[self.l_buses, 'vmin'] - v_abs,
          self.pdata.buses.loc[self.l_buses, 'mc_dv_lb']
        )

      # OPTI: Avoid construction of ampl.DataFrame at each window
      df_buses_periods = DataFrame.fromPandas(df_aux_buses_periods)
      ampl.setData(df_buses_periods)

      df_ampl_buses = DataFrame.fromPandas(df_aux_buses)
      ampl.setData(df_ampl_buses)

      df_ampl_scenarios = DataFrame.fromPandas(df_aux_scenarios)
      ampl.setData(df_ampl_scenarios)

      # ------------------------------------------------------------------------------------ #
      # Set Hotstart to nominal setpoint (deltas -> 0.)
      # ------------------------------------------------------------------------------------ #
      ampl.getVariable('eps').setValue(0.)
      if self.pdata.risk_measure == 'worst_case':
        ampl.getVariable('epi_ub').setValue(0.)
      if (self.pdata.bilinear_approx == 'bilinear' or
          self.pdata.bilinear_approx == 'bilinear_neuman'):
        for i in self.l_buses:
          for t in self.l_sce:
            ampl.getVariable('dv')[i, t].setValue(0.)
      elif (self.pdata.bilinear_approx == 'mccormick' or
          self.pdata.bilinear_approx == 'neuman'):
        for i in self.l_buses:
          for t in self.l_sce:
            ampl.getVariable('dv_pos')[i, t].setValue(0.)
            ampl.getVariable('dv_neg')[i, t].setValue(0.)
      else:
        raise NotSupportedYet('bilinear_approx: {} is not supported!'.format(
          self.pdata.bilinear_approx))

      for i in set_buses_dgs:
        ampl.getVariable('gp')[i].setValue(0.)
        ampl.getVariable('gq')[i].setValue(0.)
        for t in self.l_sce:
          ampl.getVariable('dup')[i, t].setValue(0.)
          ampl.getVariable('duq')[i, t].setValue(0.)

      for i in self.l_buses:
        for t in self.l_sce:
          ampl.getVariable('e_pos')[i, t].setValue(0.)
          ampl.getVariable('e_neg')[i, t].setValue(0.)

      if self.pdata.ctrl_type == 'droop_polypol':
        for i in set_buses_dgs:
          j = map_lbuses2idxar[i]
          for kk in range(self.pdata.polypol_deg + 1):
            if ar_p_nc_condition[j, 0]:
              ampl.getVariable('DL_pp_pos')[i, kk].setValue(0.)
              ampl.getVariable('DL_pp_neg')[i, kk].setValue(0.)
              ampl.getVariable('DL_qp_pos')[i, kk].setValue(0.)
              ampl.getVariable('DL_qp_neg')[i, kk].setValue(0.)
            if ar_q_nc_condition[j, 0]:
              ampl.getVariable('DL_pq_pos')[i, kk].setValue(0.)
              ampl.getVariable('DL_pq_neg')[i, kk].setValue(0.)
              ampl.getVariable('DL_qq_pos')[i, kk].setValue(0.)
              ampl.getVariable('DL_qq_neg')[i, kk].setValue(0.)
            if ar_up_max_condition[j, 0]:
              ampl.getVariable('DP_p_pos')[i, kk].setValue(0.)
              ampl.getVariable('DP_p_neg')[i, kk].setValue(0.)
              ampl.getVariable('DP_q_pos')[i, kk].setValue(0.)
              ampl.getVariable('DP_q_neg')[i, kk].setValue(0.)
      
      # ------------------------------------------------------------------------------------ #
      # Solve model
      # ------------------------------------------------------------------------------------ #
      str_status = ''
      

      ampl.solve()# hack_command(ampl.solve)
      try:
        str_status = ampl.getObjective(fobj_name).result()
        # str_status = re.search(r'= (\w+)', ampl.getOutput('display {}.result;'.format(fobj_name))).group(1)

      except AttributeError:
        str_status = 'unfeasible'




      if str_status != 'solved':
        if self.pdata.bilinear_approx == 'bilinear':
          unfeasibility_happend = True

          for i in self.l_buses:
            for t in self.l_sce:
              ampl.getVariable('e_pos')[i, t].unfix()
              ampl.getVariable('e_neg')[i, t].unfix()

          # Solve the model again
          # hack_command(ampl.solve)
          ampl.solve()

          try:
            str_status = ampl.getObjective(fobj_name).result()
            # str_status = re.search(r'= (\w+)', ampl.getOutput('display {}.result;'.format(fobj_name))).group(1)

          except AttributeError:
            raise ExperimentError(
              'Infeasible tunning instance at window {} (str_status={})'.format(k,
                                                str_status)
            )

        if str_status != 'solved':
          raise ExperimentError('Not optimal solution at window {}'.format(k))

      # Unfix the variables that have been fixed
      if self.pdata.ctrl_type == 'droop_polypol':
        for i in set_buses_dgs:
          j = map_lbuses2idxar[i]
          for kk in range(self.pdata.polypol_deg + 1):
            if not ar_p_nc_condition[j, 0]:
              ampl.getVariable('DL_pp_pos')[i, kk].unfix()
              ampl.getVariable('DL_pp_neg')[i, kk].unfix()
              ampl.getVariable('DL_qp_pos')[i, kk].unfix()
              ampl.getVariable('DL_qp_neg')[i, kk].unfix()
            if not ar_q_nc_condition[j, 0]:
              ampl.getVariable('DL_pq_pos')[i, kk].unfix()
              ampl.getVariable('DL_pq_neg')[i, kk].unfix()
              ampl.getVariable('DL_qq_pos')[i, kk].unfix()
              ampl.getVariable('DL_qq_neg')[i, kk].unfix()
            if not ar_up_max_condition[j, 0]:
              ampl.getVariable('DP_p_pos')[i, kk].unfix()
              ampl.getVariable('DP_p_neg')[i, kk].unfix()
              ampl.getVariable('DP_q_pos')[i, kk].unfix()
              ampl.getVariable('DP_q_neg')[i, kk].unfix()

      # Dump insample solution
      if log_folder:
        # Log scenarios
        df_win_scenarios.to_csv(os.path.join(log_folder, 'df_sce_{}.csv'.format(k)),
                    index_label='index', sep='\t')

        # Calculate fobj2stage for each scenario
        ampl.eval(
          'let {t in PERIODS} fobj2stage[t] := cost_putility * ( sum{i in BUSES} ( ('
          'p_mean[i] + dp_nc[i, t] + dup[i, t]) * sum{j in BUSES} ( R[i, j] * (p_mean[j] + '
          'dp_nc[j, t] + dup[j, t]) ) + (q_mean[i] + dq_nc[i, t] + duq[i, t]) * '
          'sum{j in BUSES} ( R[i, j] * (q_mean[j] + dq_nc[j, t] + duq[j, t]) ) )) - '
          'cost_putility * sum{i in BUSES} (p_mean[i] + dp_nc[i, t] + dup[i, t]) +'
          'cost_vlim * sum{i in BUSES} (e_neg[i, t] + e_pos[i, t]);'
        )

        ampl.eval('let {t in PERIODS} losses[t] := ( sum{i in BUSES} ( (p_mean[i] + dp_nc[i, t] + '
              'dup[i, t]) * sum{j in BUSES} ( R[i, j] * (p_mean[j] + '
          'dp_nc[j, t] + dup[j, t]) ) + (q_mean[i] + dq_nc[i, t] + duq[i, t]) * '
          'sum{j in BUSES} ( R[i, j] * (q_mean[j] + dq_nc[j, t] + duq[j, t]) ) ));')

        ampl.eval('let {t in PERIODS} pinjection[t] := sum{i in BUSES} (p_mean[i] +'
              'dp_nc[i, t] + dup[i, t]);')

        ar_fobj2stage = ampl.getParameter('fobj2stage').getValues().toPandas()[
          'fobj2stage'].values

        ar_pinjection = ampl.getParameter('pinjection').getValues().toPandas()[
          'pinjection'].values

        ar_losses = ampl.getParameter('losses').getValues().toPandas()[
          'losses'].values

        # Construct second stage dataframe
        if (self.pdata.bilinear_approx == 'mccormick' or
            self.pdata.bilinear_approx == 'neuman'):
          df_2stage_vars = ampl.getData('dv_pos', 'dv_neg', 'dup', 'duq').toPandas()
          df_2stage_vars['dv'] = df_2stage_vars['dv_pos'] - df_2stage_vars['dv_neg']
          df_2stage_vars.drop(['dv_pos', 'dv_neg'], inplace=True, axis=1)
        elif (self.pdata.bilinear_approx == 'bilinear' or
            self.pdata.bilinear_approx == 'bilinear_neuman'):
          df_2stage_vars = ampl.getData('dv', 'dup', 'duq').toPandas()

        df_idx_2stage_vars = pd.DataFrame(zip(*df_2stage_vars.index)).transpose().astype(
          'int64')
        idx_mult_2stage_vars = pd.MultiIndex.from_frame(df_idx_2stage_vars,
                                names=['bus', 'sce'])
        df_2stage_vars.index = idx_mult_2stage_vars
        df_2stage_vars = df_2stage_vars.unstack(level='bus')
        df_2stage_vars.columns = df_2stage_vars.columns.map(
          lambda x: ''.join([str(i) for i in x]))

        df_2stage_vars['fobj2stage'] = ar_fobj2stage
        df_2stage_vars['weights'] = sce_gen.ar_prob
        df_2stage_vars['pinjection'] = ar_pinjection
        df_2stage_vars['losses'] = ar_losses

        # Write scenarios insample dataframe
        fn_2stage_vars = os.path.join(log_folder, 'df_vars_insample_{}.csv'.format(k))
        df_2stage_vars.to_csv(fn_2stage_vars, index_label='index')

        # Construct set-point dataframe
        l_cols_setpoint = ['up_mean', 'uq_mean', 'v_set']
        df_setpoint = df_aux_buses.loc[:, l_cols_setpoint]

        # Write set-point
        fn_setpoint = os.path.join(log_folder, 'df_setpoint_{}.csv'.format(k))
        df_setpoint.to_csv(fn_setpoint, index_label='index')

      # Fix e_pos e_neg again if
      #   After writing e_pos e_neg to insample solution,
      #   if they have been unfixed.. fix them again
      if self.pdata.bilinear_approx == 'bilinear':
        if unfeasibility_happend:
          for i in self.l_buses:
            for t in self.l_sce:
              ampl.getVariable('e_pos')[i, t].setValue(0.)
              ampl.getVariable('e_neg')[i, t].setValue(0.)
              ampl.getVariable('e_pos')[i, t].fix()
              ampl.getVariable('e_neg')[i, t].fix()
          unfeasibility_happend = False

      # Obtain results
      Gv_p = np.diag(ampl.getData('gp').toPandas()['gp'].values)
      Gv_q = np.diag(ampl.getData('gq').toPandas()['gq'].values)

      if not np.absolute(np.linalg.eigvals(R @ Gv_p + X @ Gv_q)).max() <= 1.:
        raise ExperimentError('Stability condition is not satisfied!')
      assert ar_up_nom.shape == (self.n_buses, 1)

      # Obtaining affine policy ctrl coefficients
      # OPTI: do only one call to getData with all BUSES x DEGS indexed variables
      ar_DL_pp_aux = (
        ampl.getData('DL_pp_pos').toPandas()['DL_pp_pos'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C') -
        ampl.getData('DL_pp_neg').toPandas()['DL_pp_neg'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C')
      )
      ar_DL_qp_aux = (
        ampl.getData('DL_qp_pos').toPandas()['DL_qp_pos'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C') -
        ampl.getData('DL_qp_neg').toPandas()['DL_qp_neg'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C')
      )

      ar_DL_pq_aux = (
        ampl.getData('DL_pq_pos').toPandas()['DL_pq_pos'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C') -
        ampl.getData('DL_pq_neg').toPandas()['DL_pq_neg'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C')
      )

      ar_DL_qq_aux = (
        ampl.getData('DL_qq_pos').toPandas()['DL_qq_pos'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C') -
        ampl.getData('DL_qq_neg').toPandas()['DL_qq_neg'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C')
      )

      ar_DP_p_aux = (
        ampl.getData('DP_p_pos').toPandas()['DP_p_pos'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C') -
        ampl.getData('DP_p_neg').toPandas()['DP_p_neg'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C')
      )

      ar_DP_q_aux = (
        ampl.getData('DP_q_pos').toPandas()['DP_q_pos'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C') -
        ampl.getData('DP_q_neg').toPandas()['DP_q_neg'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C')
      )

      df_insample_sol.loc[tini_win, l_cols_DL_pp] = ar_DL_pp_aux.flatten()
      df_insample_sol.loc[tini_win, l_cols_DL_pq] = ar_DL_pq_aux.flatten()
      df_insample_sol.loc[tini_win, l_cols_DL_qp] = ar_DL_qp_aux.flatten()
      df_insample_sol.loc[tini_win, l_cols_DL_qq] = ar_DL_qq_aux.flatten()
      df_insample_sol.loc[tini_win, l_cols_DP_p] = ar_DP_p_aux.flatten()
      df_insample_sol.loc[tini_win, l_cols_DP_q] = ar_DP_q_aux.flatten()

      for dd in range(self.pdata.polypol_deg + 1):
        ar_DL_pp[:, :, dd] = np.diag(ar_DL_pp_aux[:, dd])
        ar_DL_pq[:, :, dd] = np.diag(ar_DL_pq_aux[:, dd])
        ar_DL_qp[:, :, dd] = np.diag(ar_DL_qp_aux[:, dd])
        ar_DL_qq[:, :, dd] = np.diag(ar_DL_qq_aux[:, dd])
        ar_DP_p[:, :, dd] = np.diag(ar_DP_p_aux[:, dd])
        ar_DP_q[:, :, dd] = np.diag(ar_DP_q_aux[:, dd])

      ctrl.update(Gv_p, Gv_q, ar_DL_pp, ar_DL_pq, ar_DL_qp, ar_DL_qq, ar_DP_p, ar_DP_q,
            ar_up_nom, ar_uq_nom, v_abs.reshape(self.n_buses, 1))

      df_insample_sol.loc[tini_win, l_cols_gpv] = Gv_p.diagonal()
      df_insample_sol.loc[tini_win, l_cols_gqv] = Gv_q.diagonal()

      # Simulate and obtain solution
      df_win_sim.loc[:, :] = self.df_data.loc[trange_win, :].values
      try:
        df_sim_sol.loc[trange_win, :] = self.simulate_controller_new(
          self.pdata, df_win_sim, ctrl, v_abs_sim).values
      except ExperimentError:
        with open('experiment_log.txt', 'w') as hfile:
          hfile.write('Experiment error at tini_win {}'.format(str(tini_win)))
        exit(1)
      # Shift to next time window
      trange_win = trange_win.shift(self.n_rh)

      # Clean memory
      gc.collect()
    return df_sim_sol, df_insample_sol

  def run_toy_test(self):
    # Config and initialization
    self.solver_path_in = KNITRO_PATH  # IPOPT_PATH
    self.solver_path_sim = KNITRO_PATH  # IPOPT_PATH
    self.mod_in_string = None
    self.mod_sim_string = None
    self.solver_options_in = 'outlev=0' # SOLVER_OPTIONS_KNITRO
    self.solver_options_sim = 'outlev=0'

    """ ------------------------------------------------------------------------------------ """
    """ ########################## Initialize controller setter ############################ """
    """ ------------------------------------------------------------------------------------ """
    # Configure model declaration
    try:
      if self.pdata.risk_measure == 'expected_value':
        if self.pdata.ctrl_type == 'droop_polypol':
          self.mod_in_string = (
            'FOBJ_STOCHASTIC,VOLTAGE_LB2,VOLTAGE_UB2,LINEAR_PFLOW,STABILITY,'
            'PROPOSED_CONTROL_P_BILINEAR,PROPOSED_CONTROL_Q_BILINEAR,e_pos,'
            'e_neg,eps,dv,dup,duq,gp,gq,DL_pp_pos,DL_pq_pos,DL_qp_pos,DL_qq_pos,'
            'DL_pp_neg,DL_pq_neg,DL_qp_neg,DL_qq_neg,DP_p_pos,DP_q_pos,DP_p_neg,'
            'DP_q_neg,PROPOSED_PMAX,PROPOSED_SMAX,PROPOSED_PMIN'
          )
        elif self.pdata.ctrl_type == 'droop':
          self.mod_in_string = (
            """
            FOBJ_STOCHASTIC,VOLTAGE_LB2,VOLTAGE_UB2,LINEAR_PFLOW,STABILITY,
            PROPOSED_CONTROL_P_BILINEAR,PROPOSED_CONTROL_Q_BILINEAR,e_pos,
            e_neg,eps,dv,dup,duq,gp,gq,PROPOSED_PMAX,PROPOSED_SMAX,PROPOSED_PMIN
            """
          )
        else:
          raise InputError("Invalid general configuration parameter: ctrl_type")
      elif self.pdata.risk_measure == 'worst_case':
        if self.pdata.ctrl_type == 'droop_polypol':
          self.mod_in_string = (
            """
            FOBJ_ROBUST,EPIGRAPH_UB,LINEAR_PFLOW,VOLTAGE_UB2,VOLTAGE_LB2,
            STABILITY,PROPOSED_CONTROL_P_BILINEAR,PROPOSED_CONTROL_Q_BILINEAR,
            PROPOSED_SMAX,PROPOSED_PMAX,PROPOSED_PMIN,epi_ub,e_pos,e_neg,eps,dv,dup,duq,gp,gq,
            DL_pp_pos,DL_pq_pos,DL_qp_pos,DL_qq_pos,DL_pp_neg,DL_pq_neg,DL_qp_neg,
            DL_qq_neg,DP_p_pos,DP_q_pos,DP_p_neg,DP_q_neg
            """
          )
        elif self.pdata.ctrl_type == 'droop':
          self.mod_in_string = (
            """
            FOBJ_ROBUST,EPIGRAPH_UB,VOLTAGE_LB2,VOLTAGE_UB2,LINEAR_PFLOW,
            STABILITY,PROPOSED_CONTROL_P_BILINEAR,PROPOSED_CONTROL_Q_BILINEAR,
            PROPOSED_SMAX,PROPOSED_PMAX,PROPOSED_PMIN,epi_ub,e_pos,e_neg,eps,dv,dup,duq,gp,gq
            """
          )
        else:
          raise InputError("Invalid general configuration parameter: ctrl_type")

    except KeyError:
      raise InputError('risk_measure is not defined in the general parameters')

    # Initialize objective function name
    if self.pdata.risk_measure == 'expected_value':
      fobj_name = 'FOBJ_STOCHASTIC'
    elif self.pdata.risk_measure == 'worst_case':
      fobj_name = 'FOBJ_ROBUST'
    else:
      raise InputError('Invalid general param risk_measure!')

    ampl = init_ampl_and_solver(AMPL_FOLDER, self.solver_path_in, self.solver_options_in)
    ampl.read(MODFILE_TOY_BILINEAR)
    ampl.eval('problem tunning: {};'.format(self.mod_in_string))

    # Initialize sets & params
    # Initialize auxiliary containers
    df_aux_scenarios = pd.DataFrame(index=self.l_sce, columns=[
      'sto_weights'], dtype='float64')
    df_aux_buses = pd.DataFrame(index=self.l_buses, columns=[
      'dv_lb', 'dv_ub', 'gp_lb', 'gq_lb', 'up_mean', 'uq_mean',
      'p_mean', 'q_mean', 'smax'], dtype='float64')

    idx_mul_buses_periods = pd.MultiIndex.from_product(
      [self.l_buses, list(self.l_sce)], names=['BUSES', 'PERIODS'])
    df_aux_buses_periods = pd.DataFrame(
      0.,
      index=idx_mul_buses_periods,
      columns=['dp_nc', 'dq_nc', 'up_max', 'dp_nc_scaled', 'dq_nc_scaled', 'up_max_scaled'],
      dtype='float64'
    )

    idx_mul_links = pd.MultiIndex.from_product([self.l_buses, self.l_buses],
                           names=['BUSES', 'BUSES'])

    # OPTI: Utilize sparse dataframes for link related params
    df_aux_links = pd.DataFrame(index=idx_mul_links, columns=['R', 'X'])
    df_aux_links.index.name = 'LINKS'
    # Sets
    ampl.getSet('BUSES').setValues(self.l_buses)
    ampl.getSet('PERIODS').setValues(list(self.l_sce))
    ampl.getSet('LINKS').setValues(idx_mul_links.to_list())

    # General params
    ampl.getParameter('cost_stability').set(self.pdata.cost_stability)
    ampl.getParameter('cost_lasso_v').set(self.pdata.cost_lasso_v)
    ampl.getParameter('cost_lasso_x').set(self.pdata.cost_lasso_x)
    ampl.getParameter('cost_putility').set(self.pdata.cost_putility)
    ampl.getParameter('cost_losses').set(self.pdata.cost_losses)
    ampl.getParameter('cost_vlim').set(self.pdata.cost_vlim)

    n_degs = self.pdata.polypol_deg
    ampl.getParameter('n_deg').set(n_degs)

    # Buses indexed params (fill df_aux_buses)
    df_aux_buses.loc[:, 'v_lb'] = self.pdata.buses.loc[:, 'vmin']
    df_aux_buses.loc[:, 'v_ub'] = self.pdata.buses.loc[:, 'vmax']

    # Fix variables for buses without controlled DGS
    map_lbuses2idxar = dict(zip(self.l_buses, range(self.n_buses)))
    set_buses_dgs = set(self.dgs['bus'].to_list())
    set_buses_no_dgs = set(self.l_buses) - set_buses_dgs

    for i in set_buses_no_dgs:
      ampl.getVariable('gp')[i].fix(0.)
      ampl.getVariable('gq')[i].fix(0.)
      for t in self.l_sce:
        ampl.getVariable('dup')[i, t].fix(0.)
        ampl.getVariable('duq')[i, t].fix(0.)

      if self.pdata.ctrl_type == 'droop_polypol':
        for k in range(self.pdata.polypol_deg + 1):
          ampl.getVariable('DL_pp_pos')[i, k].fix(0.)
          ampl.getVariable('DL_pq_pos')[i, k].fix(0.)
          ampl.getVariable('DL_qp_pos')[i, k].fix(0.)
          ampl.getVariable('DL_qq_pos')[i, k].fix(0.)
          ampl.getVariable('DL_pp_neg')[i, k].fix(0.)
          ampl.getVariable('DL_pq_neg')[i, k].fix(0.)
          ampl.getVariable('DL_qp_neg')[i, k].fix(0.)
          ampl.getVariable('DL_qq_neg')[i, k].fix(0.)

          ampl.getVariable('DP_p_pos')[i, k].fix(0.)
          ampl.getVariable('DP_q_pos')[i, k].fix(0.)
          ampl.getVariable('DP_p_neg')[i, k].fix(0.)
          ampl.getVariable('DP_q_neg')[i, k].fix(0.)

    """ ------------------------------------------------------------------------------------ """
    """ ################################# Rolling horizon ################################## """
    """ ------------------------------------------------------------------------------------ """
    # Set system initial condition
    # FIXME: obtain a point of the manifold given the initial xi
    v_abs_sim = np.ones(shape=(self.n_buses, 1), dtype=float)

    # Construct system matrices
    self.pdata.make_connectivity_mats()
    ar_ybus = self.pdata.ybus.values
    ar_Ar_load = self.pdata.A_load.values[:, 1:]
    ar_Ar_dg = self.pdata.A_dg.values[:, 1:]

    # Set time config
    tini = self.time_map[0]
    tend = self.time_map[self.n_rh - 1]
    trange_win = pd.date_range(start=tini, end=tend, freq=self.dt)

    # Initialize nominal setpoint calculator
    sta_opf = OPFsto(self.pdata)
    sta_pflow = Pflow(self.pdata)

    df_win_sim = pd.DataFrame(index=self.l_t_sim, columns=self.l_data_cols, dtype='float64')
    df_sim_sol = pd.DataFrame(index=self.time_map, columns=self.l_cols_sol, dtype='float64')

    dt_win = self.n_rh * self.dt
    idx_insample_sol = pd.date_range(start=self.time_map[0], freq=dt_win, periods=self.n_win)

    l_cols_gpv = [SNAM_GPV + str(i) for i in self.l_buses]
    l_cols_gqv = [SNAM_GQV + str(i) for i in self.l_buses]
    """ SNAM_DLPP = 'DLpp'
      SNAM_DLPQ = 'DLpq'
      SNAM_DLQP = 'DLqp'
      SNAM_DLQQ = 'DLqq'
      SNAM_DP = 'DPp'
      SNAM_DP = 'DPq'
    """
    l_cols_DL_pp = [SNAM_DLPP + '_' + '_'.join([str(ii), str(jj)])
            for ii in self.l_buses for jj in range(n_degs + 1)]
    l_cols_DL_pq = [SNAM_DLPQ + '_' + '_'.join([str(ii), str(jj)])
            for ii in self.l_buses for jj in range(n_degs + 1)]
    l_cols_DL_qp = [SNAM_DLQP + '_' + '_'.join([str(ii), str(jj)])
            for ii in self.l_buses for jj in range(n_degs + 1)]
    l_cols_DL_qq = [SNAM_DLQQ + '_' + '_'.join([str(ii), str(jj)])
            for ii in self.l_buses for jj in range(n_degs + 1)]
    l_cols_DP_p = [SNAM_DPP + '_' + '_'.join([str(ii), str(jj)])
            for ii in self.l_buses for jj in range(n_degs + 1)]
    l_cols_DP_q = [SNAM_DPQ + '_' + '_'.join([str(ii), str(jj)])
            for ii in self.l_buses for jj in range(n_degs + 1)]

    l_cols_D = (l_cols_DL_pp + l_cols_DL_pq + l_cols_DL_qp + l_cols_DL_qq + l_cols_DP_p +
          l_cols_DP_q)

    l_cols_data_basic = self.l_cols_loadp + self.l_cols_loadq + self.l_cols_dgpmax



    l_cols_fobj2stage = [SNAM_FOBJ_EXP, SNAM_FOBJ_CVAR, SNAM_FOBJ_MAX]
    l_cols_insample_sol = (self.l_cols_v + self.l_cols_vang + self.l_cols_dgp + self.l_cols_dgq
                 + l_cols_gpv + l_cols_gqv + l_cols_D + l_cols_fobj2stage)
    df_insample_sol = pd.DataFrame(index=idx_insample_sol, columns=l_cols_insample_sol)

    # Initialize controller:
    u_snom = ar_Ar_dg.transpose() @ self.dgs.loc[:, ['snom']].values
    df_aux_buses.loc[:, 'smax'] = u_snom

    # Initialize CtrlProposed
    ctrl = CtrlProposed(u_snom, self.pdata.polypol_deg)

    # Initialize SceGenerator
    # sce_gen = SceGenShift(self.df_data, self.n_sce)

    if self.pdata.scegen_type == 'ddus_kmeans':
      sce_gen = SceGenDDUSKmeans(self.df_data, self.n_sce,
                     self.pdata.scegen_n_win_ahead,
                     self.pdata.scegen_n_days,
                     self.pdata.scegen_n_win_before,
                     self.pdata.scegen_n_days_delay)
    else:
      raise NotImplementedError("""
      scegen_type = {}. Only SceGenDDUSKmeans is supported!""".format(self.pdata.scegen_type))

    # Initialize ctrl.matrices
    ar_DL_pp = np.zeros(shape=(self.n_buses, self.n_buses, n_degs + 1), dtype='float64')
    ar_DL_pq = np.zeros(shape=(self.n_buses, self.n_buses, n_degs + 1), dtype='float64')
    ar_DL_qp = np.zeros(shape=(self.n_buses, self.n_buses, n_degs + 1), dtype='float64')
    ar_DL_qq = np.zeros(shape=(self.n_buses, self.n_buses, n_degs + 1), dtype='float64')

    ar_DP_p = np.zeros(shape=(self.n_buses, self.n_buses, n_degs + 1), dtype='float64')
    ar_DP_q = np.zeros(shape=(self.n_buses, self.n_buses, n_degs + 1), dtype='float64')

    # Initialize insample dump folder log_folder
    log_folder = None
    if 'log_folder' in self.pdata.grid_tables['general'].index:
      log_folder = self.pdata.log_folder
      if not os.path.exists(log_folder):
        os.mkdir(log_folder)
      ampl.eval('param fobj2stage {PERIODS};')
      ampl.eval('param pinjection {PERIODS};')
      ampl.eval('param losses {PERIODS};')

    for k in range(self.n_win):
      # Predict uncertain parameters (perfect mean)
      df_win_scenarios = sce_gen.run(trange_win, trange_win)
      df_win_centroid = sce_gen.centroid()
      df_aux_scenarios.loc[:, 'sto_weights'] = sce_gen.ar_prob


      # Calculate operational setpoint
      df_sol_in_opf = sta_opf.run(df_data=df_win_scenarios,
                    df_sto_weights=df_aux_scenarios.loc[:, 'sto_weights'])
      df_sol_in_pflow = sta_pflow.run(df_win_scenarios, df_sol_in_opf)

      v_abs = df_sol_in_pflow.loc[:, self.l_cols_v].mean().values

      # Setting v_set in droop_polypol model
      df_aux_buses.loc[:, 'v_set'] = v_abs

      ar_v_abs_nom = v_abs.reshape(self.n_buses, 1)
      if k == 0:
        v_abs_sim = ar_v_abs_nom
      v_ang = df_sol_in_pflow.loc[0, self.l_cols_vang].values

      # Nominal p, q injection
      ar_loadp_nom = df_win_centroid.loc[[0], self.l_cols_loadp].values
      ar_loadq_nom = df_win_centroid.loc[[0], self.l_cols_loadq].values
      ar_p_nc_nom = - np.transpose(ar_loadp_nom @ ar_Ar_load)
      ar_q_nc_nom = - np.transpose(ar_loadq_nom @ ar_Ar_load)

      ar_dgp_nom = df_sol_in_opf.loc[[0], self.l_cols_dgp].values.astype('float64')
      ar_dgq_nom = df_sol_in_opf.loc[[0], self.l_cols_dgq].values.astype('float64')
      ar_up_nom = np.transpose(ar_dgp_nom @ ar_Ar_dg)
      ar_uq_nom = np.transpose(ar_dgq_nom @ ar_Ar_dg)

      ar_busp_nom = ar_up_nom + ar_p_nc_nom
      ar_busq_nom = ar_uq_nom + ar_q_nc_nom
      df_aux_buses.loc[:, 'p_mean'] = ar_busp_nom
      df_aux_buses.loc[:, 'q_mean'] = ar_busq_nom
      df_aux_buses.loc[:, 'up_mean'] = ar_up_nom
      df_aux_buses.loc[:, 'uq_mean'] = ar_uq_nom

      # Saving nominal set-point
      tini_win = trange_win[0]
      df_insample_sol.loc[tini_win, self.l_cols_v] = v_abs
      df_insample_sol.loc[tini_win, self.l_cols_vang] = v_ang
      df_insample_sol.loc[tini_win, self.l_cols_dgp] = ar_dgp_nom
      df_insample_sol.loc[tini_win, self.l_cols_dgq] = ar_dgq_nom

      # Construct linear model
      H, a = makeFOT(ar_ybus, v_abs, v_ang, ar_busp_nom.flatten(),
               ar_busq_nom.flatten())
      R = H[:, :self.n_buses]
      X = H[:, self.n_buses:]
      df_aux_links.loc[:, 'R'] = R.flatten()
      df_aux_links.loc[:, 'X'] = X.flatten()

      # OPTI: Avoid construction of ampl.DataFrame at each window
      df_links = DataFrame.fromPandas(df_aux_links)
      ampl.setData(df_links)

      # Set buses x periods data: dp_nc, dq_nc, up_max
      ar_dgpmax = df_win_scenarios.loc[:, self.l_cols_dgpmax].values
      ar_loadp = df_win_scenarios.loc[:, self.l_cols_loadp].values
      ar_loadq = df_win_scenarios.loc[:, self.l_cols_loadq].values
      ar_up_max = np.transpose(ar_dgpmax @ ar_Ar_dg)
      ar_p_nc = np.transpose(- ar_loadp @ ar_Ar_load)
      ar_q_nc = np.transpose(- ar_loadq @ ar_Ar_load)

      ar_p_nc_del = ar_p_nc - ar_p_nc_nom
      ar_q_nc_del = ar_q_nc - ar_q_nc_nom

      ar_p_nc_del_std = ar_p_nc_del.std(axis=1)
      ar_q_nc_del_std = ar_q_nc_del.std(axis=1)

      df_aux_buses_periods.loc[:, 'up_max'] = ar_up_max.flatten()
      df_aux_buses_periods.loc[:, 'dp_nc'] = ar_p_nc_del.flatten()
      df_aux_buses_periods.loc[:, 'dq_nc'] = ar_q_nc_del.flatten()

      ar_up_max_mean = ar_up_max.mean(axis=1)
      ar_up_max_std = ar_up_max.std(axis=1)

      # Update ctrl psi scaling parameters

      ctrl.update_psi_params(ar_p_nc_nom, ar_p_nc_del_std, ar_q_nc_nom, ar_q_nc_del_std,
                   ar_up_max_mean, ar_up_max_std)
      ar_p_nc_scaled = ctrl.scale_p_nc(ar_p_nc)
      ar_q_nc_scaled = ctrl.scale_q_nc(ar_q_nc)
      ar_up_max_scaled = ctrl.scale_up_max(ar_up_max)
      df_aux_buses_periods.loc[:, 'dp_nc_scaled'] = ar_p_nc_scaled.flatten()
      df_aux_buses_periods.loc[:, 'dq_nc_scaled'] = ar_q_nc_scaled.flatten()
      df_aux_buses_periods.loc[:, 'up_max_scaled'] = ar_up_max_scaled.flatten()

      # Fix ctrl coefficient variables for null scaled signals
      ar_p_nc_condition = ctrl.p_nc_condition
      ar_q_nc_condition = ctrl.q_nc_condition
      ar_up_max_condition = ctrl.up_max_condition
      # OPTI: Avoid in-loop calls to ampl.getVariable(), instead call it aoutside the
      #  loop and assign the result to variables

      for i in set_buses_dgs:
        j = map_lbuses2idxar[i]
        for kk in range(self.pdata.polypol_deg + 1):
          if not ar_p_nc_condition[j, 0]:
            ampl.getVariable('DL_pp_pos')[i, kk].fix(0.)
            ampl.getVariable('DL_pp_neg')[i, kk].fix(0.)
            ampl.getVariable('DL_qp_pos')[i, kk].fix(0.)
            ampl.getVariable('DL_qp_neg')[i, kk].fix(0.)
          if not ar_q_nc_condition[j, 0]:
            ampl.getVariable('DL_pq_pos')[i, kk].fix(0.)
            ampl.getVariable('DL_pq_neg')[i, kk].fix(0.)
            ampl.getVariable('DL_qq_pos')[i, kk].fix(0.)
            ampl.getVariable('DL_qq_neg')[i, kk].fix(0.)
          if not ar_up_max_condition[j, 0]:
            ampl.getVariable('DP_p_pos')[i, kk].fix(0.)
            ampl.getVariable('DP_p_neg')[i, kk].fix(0.)
            ampl.getVariable('DP_q_pos')[i, kk].fix(0.)
            ampl.getVariable('DP_q_neg')[i, kk].fix(0.)

      # Update mccormick bounds
      if self.pdata.bilinear_approx == 'mccormick':
        ar_v_ub_mc = np.minimum(
          self.pdata.buses.loc[self.l_buses, 'vmax'] - v_abs,
          self.pdata.buses.loc[self.l_buses, 'mc_dv_ub']
        )
        ar_v_lb_mc = np.maximum(
          self.pdata.buses.loc[self.l_buses, 'vmin'] - v_abs,
          self.pdata.buses.loc[self.l_buses, 'mc_dv_lb']
        )

      # OPTI: Avoid construction of ampl.DataFrame at each window
      df_buses_periods = DataFrame.fromPandas(df_aux_buses_periods)
      ampl.setData(df_buses_periods)

      df_ampl_buses = DataFrame.fromPandas(df_aux_buses)
      ampl.setData(df_ampl_buses)

      df_ampl_scenarios = DataFrame.fromPandas(df_aux_scenarios)
      ampl.setData(df_ampl_scenarios)

      # ------------------------------------------------------------------------------------ #
      # Set Hotstart to nominal point (deltas -> 0.)
      # ------------------------------------------------------------------------------------ #
      for i in self.l_buses:
        for t in self.l_sce:
          ampl.getVariable('dv')[i, t].setValue(0.)
          ampl.getVariable('e_pos')[i, t].setValue(0.)
          ampl.getVariable('e_neg')[i, t].setValue(0.)

      for i in set_buses_dgs:
        ampl.getVariable('gp')[i].setValue(0.)
        ampl.getVariable('gq')[i].setValue(0.)
        for t in self.l_sce:
          ampl.getVariable('dup')[i, t].setValue(0.)
          ampl.getVariable('duq')[i, t].setValue(0.)

      for i in set_buses_dgs:
        j = map_lbuses2idxar[i]
        for kk in range(self.pdata.polypol_deg + 1):
          if ar_p_nc_condition[j, 0]:
            ampl.getVariable('DL_pp_pos')[i, kk].setValue(0.)
            ampl.getVariable('DL_pp_neg')[i, kk].setValue(0.)
            ampl.getVariable('DL_qp_pos')[i, kk].setValue(0.)
            ampl.getVariable('DL_qp_neg')[i, kk].setValue(0.)
          if ar_q_nc_condition[j, 0]:
            ampl.getVariable('DL_pq_pos')[i, kk].setValue(0.)
            ampl.getVariable('DL_pq_neg')[i, kk].setValue(0.)
            ampl.getVariable('DL_qq_pos')[i, kk].setValue(0.)
            ampl.getVariable('DL_qq_neg')[i, kk].setValue(0.)
          if ar_up_max_condition[j, 0]:
            ampl.getVariable('DP_p_pos')[i, kk].setValue(0.)
            ampl.getVariable('DP_p_neg')[i, kk].setValue(0.)
            ampl.getVariable('DP_q_pos')[i, kk].setValue(0.)
            ampl.getVariable('DP_q_neg')[i, kk].setValue(0.)

      # ------------------------------------------------------------------------------------ #
      # Solve model
      # ------------------------------------------------------------------------------------ #
      str_status = ''
      # hack_command(ampl.solve)
      ampl.solve()
        # str_status = re.search(r'= (\w+)', ampl.getOutput('display {}.result;'.format(fobj_name))).group(1)

        # try:
        #str_status = re.search(r'= (\w+)', ampl.getOutput(
        #    'display {}.result;'.format(fobj_name))).group(1)
        #except AttributeError:
        #  print('caca')
        #if str_status != 'solved' and str_status != 'limit':
          #raise ExperimentError(
          #  'Infeasible tunning instance at window {} (str_status={})'.format(k, str_status)
          #)

      # ------------------------------------------------------------------------------------ #
      # Dump insample solution if log_folder param exists
      # ------------------------------------------------------------------------------------ #
      if log_folder:
        # Log scenarios
        df_win_scenarios.to_csv(os.path.join(log_folder, 'df_sce_{}.csv'.format(k)),
                    index_label='index', sep='\t')

        # Calculate fobj2stage for each scenario
        ampl.eval(
          'let {t in PERIODS} fobj2stage[t] := cost_putility * ( sum{i in BUSES} ( ('
          'p_mean[i] + dp_nc[i, t] + dup[i, t]) * sum{j in BUSES} ( R[i, j] * (p_mean[j] + '
          'dp_nc[j, t] + dup[j, t]) ) + (q_mean[i] + dq_nc[i, t] + duq[i, t]) * '
          'sum{j in BUSES} ( R[i, j] * (q_mean[j] + dq_nc[j, t] + duq[j, t]) ) )) - '
          'cost_putility * sum{i in BUSES} (p_mean[i] + dp_nc[i, t] + dup[i, t]) +'
          'cost_vlim * sum{i in BUSES} (e_neg[i, t] + e_pos[i, t]);'
        )

        ampl.eval('let {t in PERIODS} losses[t] := ( sum{i in BUSES} ( (p_mean[i] + dp_nc[i, t] + '
              'dup[i, t]) * sum{j in BUSES} ( R[i, j] * (p_mean[j] + '
          'dp_nc[j, t] + dup[j, t]) ) + (q_mean[i] + dq_nc[i, t] + duq[i, t]) * '
          'sum{j in BUSES} ( R[i, j] * (q_mean[j] + dq_nc[j, t] + duq[j, t]) ) ));')

        ampl.eval('let {t in PERIODS} pinjection[t] := sum{i in BUSES} (p_mean[i] +'
              'dp_nc[i, t] + dup[i, t]);')

        ar_fobj2stage = ampl.getParameter('fobj2stage').getValues().toPandas()[
          'fobj2stage'].values

        ar_pinjection = ampl.getParameter('pinjection').getValues().toPandas()[
          'pinjection'].values

        ar_losses = ampl.getParameter('losses').getValues().toPandas()[
          'losses'].values

        # Construct second stage dataframe
        df_2stage_vars = ampl.getData('dv', 'dup', 'duq').toPandas()
        df_idx_2stage_vars = pd.DataFrame(zip(*df_2stage_vars.index)).transpose().astype(
          'int64')
        idx_mult_2stage_vars = pd.MultiIndex.from_frame(df_idx_2stage_vars,
                                names=['bus', 'sce'])
        df_2stage_vars.index = idx_mult_2stage_vars
        df_2stage_vars = df_2stage_vars.unstack(level='bus')
        df_2stage_vars.columns = df_2stage_vars.columns.map(
          lambda x: ''.join([str(i) for i in x]))

        df_2stage_vars['fobj2stage'] = ar_fobj2stage
        df_2stage_vars['weights'] = sce_gen.ar_prob
        df_2stage_vars['pinjection'] = ar_pinjection
        df_2stage_vars['losses'] = ar_losses

        # Write scenarios insample dataframe
        fn_2stage_vars = os.path.join(log_folder, 'df_vars_insample_{}.csv'.format(k))
        df_2stage_vars.to_csv(fn_2stage_vars, index_label='index')

        # Construct set-point dataframe
        l_cols_setpoint = ['up_mean', 'uq_mean', 'v_set']
        df_setpoint = df_aux_buses.loc[:, l_cols_setpoint]

        # Write set-point
        fn_setpoint = os.path.join(log_folder, 'df_setpoint_{}.csv'.format(k))
        df_setpoint.to_csv(fn_setpoint, index_label='index')

      # ------------------------------------------------------------------------------------ #
      # Obtain first stage results and update controller
      # ------------------------------------------------------------------------------------ #
      Gv_p = np.diag(ampl.getData('gp').toPandas()['gp'].values)
      Gv_q = np.diag(ampl.getData('gq').toPandas()['gq'].values)

      if not np.absolute(np.linalg.eigvals(R @ Gv_p + X @ Gv_q)).max() <= 1.:
        raise ExperimentError('Stability condition is not satisfied!')
      assert ar_up_nom.shape == (self.n_buses, 1)

      # Obtaining affine policy ctrl coefficients
      # OPTI: do only one call to getData with all BUSES x DEGS indexed variables
      ar_DL_pp_aux = (
        ampl.getData('DL_pp_pos').toPandas()['DL_pp_pos'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C') -
        ampl.getData('DL_pp_neg').toPandas()['DL_pp_neg'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C')
      )
      ar_DL_qp_aux = (
        ampl.getData('DL_qp_pos').toPandas()['DL_qp_pos'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C') -
        ampl.getData('DL_qp_neg').toPandas()['DL_qp_neg'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C')
      )

      ar_DL_pq_aux = (
        ampl.getData('DL_pq_pos').toPandas()['DL_pq_pos'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C') -
        ampl.getData('DL_pq_neg').toPandas()['DL_pq_neg'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C')
      )

      ar_DL_qq_aux = (
        ampl.getData('DL_qq_pos').toPandas()['DL_qq_pos'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C') -
        ampl.getData('DL_qq_neg').toPandas()['DL_qq_neg'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C')
      )

      ar_DP_p_aux = (
        ampl.getData('DP_p_pos').toPandas()['DP_p_pos'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C') -
        ampl.getData('DP_p_neg').toPandas()['DP_p_neg'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C')
      )

      ar_DP_q_aux = (
        ampl.getData('DP_q_pos').toPandas()['DP_q_pos'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C') -
        ampl.getData('DP_q_neg').toPandas()['DP_q_neg'].values.reshape(
          (self.n_buses, self.pdata.polypol_deg + 1), order='C')
      )

      df_insample_sol.loc[tini_win, l_cols_DL_pp] = ar_DL_pp_aux.flatten()
      df_insample_sol.loc[tini_win, l_cols_DL_pq] = ar_DL_pq_aux.flatten()
      df_insample_sol.loc[tini_win, l_cols_DL_qp] = ar_DL_qp_aux.flatten()
      df_insample_sol.loc[tini_win, l_cols_DL_qq] = ar_DL_qq_aux.flatten()
      df_insample_sol.loc[tini_win, l_cols_DP_p] = ar_DP_p_aux.flatten()
      df_insample_sol.loc[tini_win, l_cols_DP_q] = ar_DP_q_aux.flatten()

      for dd in range(self.pdata.polypol_deg + 1):
        ar_DL_pp[:, :, dd] = np.diag(ar_DL_pp_aux[:, dd])
        ar_DL_pq[:, :, dd] = np.diag(ar_DL_pq_aux[:, dd])
        ar_DL_qp[:, :, dd] = np.diag(ar_DL_qp_aux[:, dd])
        ar_DL_qq[:, :, dd] = np.diag(ar_DL_qq_aux[:, dd])
        ar_DP_p[:, :, dd] = np.diag(ar_DP_p_aux[:, dd])
        ar_DP_q[:, :, dd] = np.diag(ar_DP_q_aux[:, dd])

      ctrl.update(Gv_p, Gv_q, ar_DL_pp, ar_DL_pq, ar_DL_qp, ar_DL_qq, ar_DP_p, ar_DP_q,
            ar_up_nom, ar_uq_nom, v_abs.reshape(self.n_buses, 1))

      df_insample_sol.loc[tini_win, l_cols_gpv] = Gv_p.diagonal()
      df_insample_sol.loc[tini_win, l_cols_gqv] = Gv_q.diagonal()

      # Simulate and obtain solution
      df_win_sim.loc[:, :] = self.df_data.loc[trange_win, :].values
      try:
        df_sim_sol.loc[trange_win, :] = self.simulate_controller_new(
          self.pdata, df_win_sim, ctrl, v_abs_sim).values
      except ExperimentError:
        with open('experiment_log.txt', 'w') as hfile:
          hfile.write('Experiment error at tini_win {}'.format(str(tini_win)))
        exit(1)
      # Shift to next time window
      trange_win = trange_win.shift(self.n_rh)

      # Clean memory
      gc.collect()
    return df_sim_sol, df_insample_sol

  def run_ieee_1547(self):
    assert not self.dgs.empty

    # Init controller
    self.pdata.make_connectivity_mats()
    A_dg_reduced_t = self.pdata.A_dg.values.transpose()[1:]
    u_snom = A_dg_reduced_t @ self.dgs['snom'].to_numpy().reshape(self.dgs.shape[0], 1)
    dt_seconds = self.dt / timedelta(seconds=1)
    ctrl = CtrlIEEE(u_snom, dt_seconds)

    # Initial conditions
    v_abs = np.ones(shape=(ctrl.n, 1), dtype='float64')

    # Simulate controller
    trange = self.pdata.time_map(2)

    df_sim_result = self.simulate_controller_new(self.pdata, self.pdata.df_data.loc[trange, :],
                         ctrl, v_abs)
    return df_sim_result

  def simulate_controller(self, pdata_bucle, df_win_sim, ctrl: Controller, v_abs):
    # Preliminar definitions
    self.solver_path_sim = KNITRO_PATH  # IPOPT_PATH
    self.mod_file_sim = SIMULATE_NCOGNIZANT_MODFILE
    self.solver_options_sim = 'outlev=0'

    # Initialize ampl environment
    ampl = init_ampl_and_solver(AMPL_FOLDER, self.solver_path_sim, self.solver_options_sim)
    ampl.read(self.mod_file_sim)

    # Initialize sets & params
    #   Sets
    ampl.getSet('BUSES0').setValues(self.l_buses0)
    ampl.getSet('BUSES').setValues(self.l_buses)

    #   Params
    ampl.getParameter('SLACK_BUS').set(pdata_bucle.slack_bus)
    ampl.getParameter('SBase_MVA').set(pdata_bucle.sbase_mva)

    map_dgs_rename = {'bus': 'dgBus', 'snom': 'dgSnom'}
    df_dgs_aux = self.dgs.rename(columns=map_dgs_rename)
    df_dgs_aux.index.name = 'DGS'
    df_dgs = DataFrame.fromPandas(df_dgs_aux, ['DGS'])
    ampl.setData(df_dgs, 'DGS')

    map_branches_rename = {'busf': 'lFrom', 'bust': 'lTo'}
    df_branches_aux = self.branches[map_branches_rename.keys()].rename(
      columns=map_branches_rename)
    df_branches_aux['ly'] = self.branches.eval('1/sqrt(x ** 2 + r ** 2)')
    df_branches_aux['ltheta'] = - np.arctan2(self.branches['x'], self.branches['r'])
    df_branches_aux.index.name = 'LINES'
    df_branches = DataFrame.fromPandas(df_branches_aux, ['LINES'])
    ampl.setData(df_branches, 'LINES')

    df_demands = DataFrame('DEMANDS', ['dBus'])
    for i in self.l_loads:
      df_demands.addRow(i, i)
    ampl.setData(df_demands, 'DEMANDS')

    nse = np.zeros(shape=(len(self.l_loads),), dtype=float)
    ampl.getParameter('nse').setValues(nse.tolist())

    A_load_t = pdata_bucle.A_load.values[:, 1:].transpose()
    assert A_load_t.shape == (self.n_buses, self.n_loads)
    A_dg_t = pdata_bucle.A_dg.values.transpose()
    # Initialize solution container
    df_sim_sol = pd.DataFrame(index=df_win_sim.index, columns=self.l_cols_sol, dtype='float64')

    # Simulation loop
    for t in df_win_sim.index:
      # Set params: loadp, loadq, dgpmax
      ampl.getParameter('loadp').setValues(
        df_win_sim.loc[t, self.l_cols_loadp].to_list())
      ampl.getParameter('loadq').setValues(
        df_win_sim.loc[t, self.l_cols_loadq].to_list())
      dgp_max = df_win_sim.loc[t, self.l_cols_dgpmax].values.reshape(self.n_dgs, 1)
      up_max = A_dg_t @ dgp_max
      bus_loadp = (A_load_t @
             df_win_sim.loc[t, self.l_cols_loadp].values.reshape(self.n_loads, 1))
      bus_loadq = (A_load_t @
             df_win_sim.loc[t, self.l_cols_loadq].values.reshape(self.n_loads, 1))

      up, uq = ctrl.ret(v_abs, up_max[1:], - bus_loadp, - bus_loadq)
      dgp = pdata_bucle.A_dg.values[:, 1:] @ up
      dgq = pdata_bucle.A_dg.values[:, 1:] @ uq

      ampl.getParameter('dgp').setValues(dgp.flatten(order='C').tolist())
      ampl.getParameter('dgq').setValues(dgq.flatten(order='C').tolist())

      ampl.solve()
      str_status = re.search(r'(?<== )\w+', ampl.getOutput('display fobj.result;')).group(0)
      if str_status != 'solved':
        raise ExperimentError(
          'Infeasible pflow instance at index {} (str_status={})'.format(t, str_status)
        )

      df_aux_sol = ampl.getVariable('V').getValues().toPandas()
      v_abs[:] = df_aux_sol['V.val'].values[1:].reshape(self.n_buses, 1)
      df_sim_sol.loc[t, self.l_cols_v] = df_aux_sol['V.val'].values[1:]
      df_aux_sol = ampl.getVariable('theta').getValues().toPandas()
      df_sim_sol.loc[t, self.l_cols_vang] = df_aux_sol['theta.val'].values[1:]
      df_sim_sol.loc[t, self.l_cols_dgp] = dgp.flatten()
      df_sim_sol.loc[t, self.l_cols_dgq] = dgq.flatten()

    return df_sim_sol

  def simulate_controller_new(self, pdata_bucle, df_win_sim, ctrl: Controller, v_abs):
    # Preliminar definitions
    self.solver_path_sim = KNITRO_PATH  # IPOPT_PATH
    self.mod_file_sim = PFLOW_BIM_MODFILE
    self.solver_options_sim = 'outlev=0'

    # Initialize ampl environment
    ampl = init_ampl_and_solver(AMPL_FOLDER, self.solver_path_sim, self.solver_options_sim)
    ampl.read(self.mod_file_sim)

    # Initialize sets & params
    #   Sets
    ampl.getSet('BUSES0').setValues(self.l_buses0)
    ampl.getSet('BUSES').setValues(self.l_buses)

    #   Params
    ampl.getParameter('SLACK_BUS').set(pdata_bucle.slack_bus)
    ampl.getParameter('SBase_MVA').set(pdata_bucle.sbase_mva)

    map_dgs_rename = {'bus': 'dgBus'}
    df_dgs_aux = self.dgs.loc[:, map_dgs_rename.keys()].rename(columns=map_dgs_rename)

    if not self.dgs.empty:
      df_dgs_aux.index.name = 'DGS'
      df_dgs = DataFrame.fromPandas(df_dgs_aux, ['DGS'])
      ampl.setData(df_dgs, 'DGS')
    else:
      ampl.getSet('DGS').setValues(self.l_dgs)

    map_branches_rename = {'bust': 'lTo'}
    df_branches_aux = self.branches[map_branches_rename.keys()].rename(
      columns=map_branches_rename)
    df_branches_aux['ly'] = self.branches.eval('1/sqrt(x ** 2 + r ** 2)')
    df_branches_aux['ltheta'] = - np.arctan2(self.branches['x'], self.branches['r'])
    df_branches_aux.index.name = 'LINES'
    df_branches = DataFrame.fromPandas(df_branches_aux, ['BUSES'])
    ampl.setData(df_branches)

    df_demands = DataFrame('DEMANDS', ['dBus'])
    for i in self.l_loads:
      df_demands.addRow(i, i)
    ampl.setData(df_demands, 'DEMANDS')

    A_load_t = pdata_bucle.A_load.values[:, 1:].transpose()
    assert A_load_t.shape == (self.n_buses, self.n_loads)
    A_dg_t = pdata_bucle.A_dg.values.transpose()
    # Initialize solution container
    df_sim_sol = pd.DataFrame(index=df_win_sim.index, columns=self.l_cols_sol, dtype='float64')

    # Simulation loop
    for t in df_win_sim.index:
      # Set params: loadp, loadq, dgpmax
      ampl.getParameter('loadp').setValues(
        df_win_sim.loc[t, self.l_cols_loadp].to_list())
      ampl.getParameter('loadq').setValues(
        df_win_sim.loc[t, self.l_cols_loadq].to_list())
      dgp_max = df_win_sim.loc[t, self.l_cols_dgpmax].values.reshape(self.n_dgs, 1)
      up_max = A_dg_t @ dgp_max
      bus_loadp = (A_load_t @
             df_win_sim.loc[t, self.l_cols_loadp].values.reshape(self.n_loads, 1))
      bus_loadq = (A_load_t @
             df_win_sim.loc[t, self.l_cols_loadq].values.reshape(self.n_loads, 1))

      up, uq = ctrl.ret(v_abs, up_max[1:], - bus_loadp, - bus_loadq)
      dgp = pdata_bucle.A_dg.values[:, 1:] @ up
      dgq = pdata_bucle.A_dg.values[:, 1:] @ uq

      ampl.getParameter('dgp').setValues(dgp.flatten(order='C').tolist())
      ampl.getParameter('dgq').setValues(dgq.flatten(order='C').tolist())

      ampl.solve()

      str_status = re.search(r'(?<== )\w+', ampl.getOutput('display FOBJ.result;')).group(0)
      if str_status != 'solved':
        raise ExperimentError(
          'Infeasible pflow instance at index {} (str_status={})'.format(t, str_status)
        )

      df_aux_sol = ampl.getVariable('V').getValues().toPandas()
      v_abs[:] = df_aux_sol['V.val'].values[1:].reshape(self.n_buses, 1)
      df_sim_sol.loc[t, self.l_cols_v] = df_aux_sol['V.val'].values[1:]
      df_aux_sol = ampl.getVariable('theta').getValues().toPandas()
      df_sim_sol.loc[t, self.l_cols_vang] = df_aux_sol['theta.val'].values[1:]
      df_sim_sol.loc[t, self.l_cols_dgp] = dgp.flatten()
      df_sim_sol.loc[t, self.l_cols_dgq] = dgq.flatten()

    return df_sim_sol
