echo off
REM Set environment variables
set AMPL_FOLDER=C:\Users\PCSIM-SEP-YY\Desktop\jp_sep\ampl

REM Run experiment
call conda activate py37
python run_benchmark_others123bus_multi.py
pause