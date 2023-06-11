@echo off

REM Go to the /build/ directory
cd build

REM Run CMake to generate build files
cmake ..

REM Go to the /build-win/ directory
cd ../build-win

REM Run CMake with MinGW Makefiles generator
cmake -G "MinGW Makefiles" ..

REM Execute CMake --build command in /build-win/
cmake --build .

REM Pause to keep the command prompt window open
pause
