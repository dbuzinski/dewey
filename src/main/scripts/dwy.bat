@echo off
set PYTHONEXE="__UNDEF__"
where python3 /Q
if %ERRORLEVEL%==0 set PYTHONEXE="python3"
where python /Q
if %ERRORLEVEL%==0 set PYTHONEXE="python"
if %PYTHONEXE%=="__UNDEF__" exit /b 1
%PYTHONEXE% -c "import dewey.cli; dewey.cli.main()" %*