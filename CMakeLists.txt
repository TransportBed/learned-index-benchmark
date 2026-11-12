# Top-level build: the Python-free core library (cpp/) plus the pybind11
# extension that exposes it to Python. Driven either directly with cmake (dev)
# or by scikit-build-core when building a wheel.
cmake_minimum_required(VERSION 3.18)
project(lisbench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Native tuning: default portable; opt into -march via cache var (was the
# LISBENCH_NATIVE_ARCH env knob under the old setuptools build).
set(LISBENCH_NATIVE_ARCH "portable" CACHE STRING "Target architecture for -march (portable = none)")
option(LISBENCH_FAST_MATH "Enable -ffast-math in the core" OFF)

add_subdirectory(cpp)

# Resolve the extension against the exact interpreter selected by the caller.
# Without FindPython mode, pybind11 may fall back to an unrelated system/base
# interpreter and emit an ABI-incompatible extension (for example cp313 while
# the experiment environment runs cp312).
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(lisbench_native bindings/module.cpp)
target_link_libraries(lisbench_native PRIVATE lisbench_serving)

install(TARGETS lisbench_native LIBRARY DESTINATION .)
install(DIRECTORY src/ DESTINATION . FILES_MATCHING PATTERN "*.py")
