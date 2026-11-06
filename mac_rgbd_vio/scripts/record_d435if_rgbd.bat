@echo off
set SEQ_NAME=%~1
if "%SEQ_NAME%"=="" set SEQ_NAME=exp102_laptop_office_walk
python "%~dp0record_d435if_rgbd.py" %SEQ_NAME% --duration 60 --emitter 1
