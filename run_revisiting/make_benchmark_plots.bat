echo off
REM Set environment variables
set AMPL_FOLDER=C:\Users\PCSIM-SEP-YY\Desktop\jp_sep\ampl

REM Run experiment
call conda activate py37
python make_benchmark_plots.py
pause