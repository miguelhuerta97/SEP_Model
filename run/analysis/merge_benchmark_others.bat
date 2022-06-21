echo off
REM Set environment variables
set AMPL_FOLDER=C:\Users\PCSIM-SEP-YY\Desktop\jp_sep\ampl

REM Run experiment
call conda activate py37
python merge_benchmark_others.py
pause