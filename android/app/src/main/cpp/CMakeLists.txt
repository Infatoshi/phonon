cmake_minimum_required(VERSION 3.22.1)
project(phonon_refiner C CXX)

# llama.cpp source tree, passed in from Gradle (-DLLAMA_DIR=...)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
# One CPU backend per instruction-set level (android_armv8.0 ... armv9.2 on arm64, sse4.2 ... avx512 on x86_64),
# each a loadable module; refiner_jni.cpp loads them and ggml picks the best one the phone supports. A single
# armv8.6+i8mm build died with SIGILL on a Galaxy A16 (Cortex-A55/A76, ARMv8.2) in Test Lab, 2026-11-06.
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
set(GGML_CPU_ARM_ARCH "" CACHE STRING "" FORCE)  # a fixed arch excludes the variants (cached from older builds)
add_subdirectory(${LLAMA_DIR} llama_build)

add_library(phonon_refiner SHARED refiner_jni.cpp)
target_link_libraries(phonon_refiner llama ggml ggml-base log android)

# Kernel benchmark (androidTest/RefinerKernelBenchTest.kt only; the keyboard never loads it)
add_library(phonon_kbench SHARED kbench_jni.cpp)
target_link_libraries(phonon_kbench llama ggml ggml-base log android)
