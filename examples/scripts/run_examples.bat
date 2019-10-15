del %cd%\failed.txt
set OMP_NUM_THREADS=3
for /r %%i in (*.py) do (
python.exe %%i
if errorlevel 1 @echo %%i >> failed.txt
)
