from os import mkdir
from os.path import join, exists
from data_structure import Adn
from case_generation.toy_problems.create_ieee123bus import create_ieee123bus, scale_ieee123bus
from datetime import datetime, timedelta

# ------------------------------------------------------------------------------------------------ #
# Input
# ------------------------------------------------------------------------------------------------ #
CASE_FOLDER_SEED = '..\cases\ieee123bus_seed'
CASE_FOLDER_BSCALING = '..\cases\ieee123bus_bscaling'
CASE_FOLDER = '..\cases\ieee123bus_factor2'

if not exists(CASE_FOLDER):
	mkdir(CASE_FOLDER)

TIME_CONFIG = {
        'tini': datetime(2018, 7, 27, 0, 0, 0),
        'tiniout': datetime(2018, 8, 15, 0, 0, 0),
        'tend': datetime(2018, 8, 18, 23, 59, 54),
        'dt': timedelta(seconds=6),
        'n_rh': 150
    }

SCALE_ONLY = True

# ------------------------------------------------------------------------------------------------ #
# Run
# ------------------------------------------------------------------------------------------------ #
if not SCALE_ONLY:
	create_ieee123bus(CASE_FOLDER_BSCALING, CASE_FOLDER_SEED)

scale_ieee123bus(CASE_FOLDER, CASE_FOLDER_BSCALING)



#    

#scale_ieee123bus(CASE_FOLDER, CASE_FOLDER_BSCALING, TIME_CONFIG)

#pdata = Adn()
#pdata.read(CASE_FOLDER_BSCALING, 'ieee123bus')
#pdata.df_data = pdata.df_data * 1.05
#pdata.write(CASE_FOLDER, 'ieee123bus')