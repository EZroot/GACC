cmake_minimum_required(VERSION 3.12)
project(ez-paint-app)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK4 REQUIRED gtk4)

include_directories(${GTK4_INCLUDE_DIRS})
link_directories(${GTK4_LIBRARY_DIRS})

include_directories(components)  # Include the directory containing drawing_area.h

add_executable(ez-paint-app main.cpp components/drawing_area.cpp)
target_link_libraries(ez-paint-app ${GTK4_LIBRARIES})
