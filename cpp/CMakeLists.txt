add_library(lisbench_serving STATIC
  src/candidates.cpp
  src/rerank.cpp
  src/search.cpp
)

target_include_directories(lisbench_serving PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(lisbench_serving PUBLIC cxx_std_17)
set_property(TARGET lisbench_serving PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(lisbench_serving PRIVATE -O3)

if(NOT LISBENCH_NATIVE_ARCH STREQUAL "portable")
  target_compile_options(lisbench_serving PRIVATE -march=${LISBENCH_NATIVE_ARCH})
endif()
if(LISBENCH_FAST_MATH)
  target_compile_options(lisbench_serving PRIVATE -ffast-math)
endif()

# OpenMP stays in the module (kept per project decision). Build with gcc so the
# runtime is libgomp, matching the faiss-cpu wheel and keeping a single OpenMP
# runtime in-process.
find_package(OpenMP REQUIRED)
target_link_libraries(lisbench_serving PUBLIC OpenMP::OpenMP_CXX)

