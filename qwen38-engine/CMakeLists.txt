cmake_minimum_required(VERSION 3.20)
project(qwen38_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
# Pascal GP104 only (GTX 1070 Ti). PTX fallbacks are not built.
set(CMAKE_CUDA_ARCHITECTURES 61)

# Host flags for the Ryzen 5 2400G (znver1, AVX2, FMA). CUDA host code gets the same flags via -Xcompiler.
foreach(flag -O3 -march=znver1 -mavx2 -mfma -Wall -Wextra)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${flag}>)
endforeach()
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>)
foreach(flag -march=znver1 -mavx2 -mfma)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

# Baseline llama.cpp tree: only ggml headers and libggml (quant dequantization, decision D1).
set(LLAMA_SRC "$ENV{HOME}/llama.cpp" CACHE PATH "llama.cpp source tree (ggml headers)")
set(LLAMA_BUILD "${LLAMA_SRC}/build-iso3" CACHE PATH "llama.cpp build tree with libggml")

find_library(GGML_LIB ggml PATHS "${LLAMA_BUILD}/bin" REQUIRED)
find_library(GGML_BASE_LIB ggml-base PATHS "${LLAMA_BUILD}/bin" REQUIRED)
find_library(GGML_CPU_LIB ggml-cpu PATHS "${LLAMA_BUILD}/bin" REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(gguf_index STATIC src/packer/gguf_index.cpp)
target_include_directories(gguf_index PUBLIC src/packer "${LLAMA_SRC}/ggml/include")
target_link_libraries(gguf_index PUBLIC ${GGML_LIB} ${GGML_BASE_LIB})

add_library(cpu_ref STATIC src/cpu/cpu_ops.cpp src/cpu/dn_layer.cpp)
target_include_directories(cpu_ref PUBLIC src/cpu src/packer "${LLAMA_SRC}/ggml/include")
target_link_libraries(cpu_ref PUBLIC gguf_index ${GGML_LIB} ${GGML_BASE_LIB})

add_executable(layer01_parity tests/layer01_parity.cpp)
target_link_libraries(layer01_parity PRIVATE cpu_ref pthread)
set_target_properties(layer01_parity PROPERTIES BUILD_RPATH "${LLAMA_BUILD}/bin")


# Reference tensor dumper (test tool, built against the baseline llama.cpp tree).
find_library(LLAMA_LIB llama PATHS "${LLAMA_BUILD}/bin" REQUIRED)
find_library(LLAMA_COMMON_LIB llama-common PATHS "${LLAMA_BUILD}/bin" REQUIRED)
add_executable(dump_ref_tensors tests/tools/dump_ref_tensors.cpp)
target_include_directories(dump_ref_tensors PRIVATE "${LLAMA_SRC}/common" "${LLAMA_SRC}/include"
                           "${LLAMA_SRC}/ggml/include" "${LLAMA_SRC}/vendor")
target_link_libraries(dump_ref_tensors PRIVATE ${LLAMA_COMMON_LIB} ${LLAMA_LIB} ${GGML_LIB} ${GGML_BASE_LIB} pthread)
set_target_properties(dump_ref_tensors PROPERTIES BUILD_RPATH "${LLAMA_BUILD}/bin")

# Stage 2 engine: CPU forward, tokenizer, static arena, streaming residency.
add_library(qmat STATIC src/cpu/qmat.cpp src/cpu/mt_dot.cpp)
target_include_directories(qmat PUBLIC src/cpu src/packer src/common "${LLAMA_SRC}/ggml/include")
target_link_libraries(qmat PUBLIC gguf_index ${GGML_CPU_LIB} ${GGML_LIB} ${GGML_BASE_LIB} pthread)

# Multi-token W4A8 kernel self-test (Point A): mt_dot vs ggml vec_dot on random quantized blocks.
add_executable(mt_dot_test tests/mt_dot_test.cpp src/cpu/mt_dot.cpp src/cpu/qmat.cpp)
target_include_directories(mt_dot_test PRIVATE src/cpu src/common src/packer "${LLAMA_SRC}/ggml/include")
target_link_libraries(mt_dot_test PRIVATE gguf_index ${GGML_CPU_LIB} ${GGML_LIB} ${GGML_BASE_LIB} pthread)

# Point D evidence: scan every blk.<L>.ssm_a tensor and prove no head is -1.0 (identity invalid).
add_executable(ssm_a_scan tests/ssm_a_scan.cpp)
target_link_libraries(ssm_a_scan PRIVATE gguf_index)

# The WebUI is compiled into the binary as a byte array.
set(WEBUI_CPP "${CMAKE_BINARY_DIR}/webui_index.cpp")
add_custom_command(OUTPUT "${WEBUI_CPP}"
  COMMAND ${CMAKE_COMMAND} -DINPUT=${CMAKE_SOURCE_DIR}/webui/index.html -DOUTPUT=${WEBUI_CPP} -DSYMBOL=kWebuiIndex -P ${CMAKE_SOURCE_DIR}/cmake/embed.cmake
  DEPENDS webui/index.html cmake/embed.cmake VERBATIM)
set_source_files_properties(third_party/cpp-httplib/httplib.cpp PROPERTIES COMPILE_OPTIONS "-w")

add_executable(qwen38-engine
  src/engine/main.cpp src/engine/engine.cpp src/engine/tokenizer.cpp src/engine/alloc_guard.cpp
  src/engine/sampler.cpp src/engine/generator.cpp src/engine/server.cpp
  third_party/cpp-httplib/httplib.cpp "${WEBUI_CPP}"
  src/cpu/cpu_ops.cpp)
target_include_directories(qwen38-engine PRIVATE src/engine src/cpu src/packer src/cuda third_party third_party/cpp-httplib "${LLAMA_SRC}/ggml/include")
target_link_libraries(qwen38-engine PRIVATE gpu_cuda qmat gguf_index ${GGML_CPU_LIB} ${GGML_LIB} ${GGML_BASE_LIB} CUDA::cudart pthread)
set_target_properties(qwen38-engine PROPERTIES BUILD_RPATH "${LLAMA_BUILD}/bin")

# Stage 3 foundation: shared host code (iso3 reference, fp16 bit helpers) and its test.
add_library(iso3_ref STATIC src/common/iso3_ref.cpp)
target_include_directories(iso3_ref PUBLIC src/common)
target_compile_options(iso3_ref PRIVATE -ffp-contract=off)

add_executable(foundation_test tests/foundation_test.cpp)
target_compile_options(foundation_test PRIVATE -mf16c)
target_include_directories(foundation_test PRIVATE src/common "${LLAMA_SRC}/ggml/include")
target_link_libraries(foundation_test PRIVATE iso3_ref ${GGML_LIB} ${GGML_BASE_LIB})
set_target_properties(foundation_test PROPERTIES BUILD_RPATH "${LLAMA_BUILD}/bin")

# Stage 3 CUDA kernels (sm_61): quantized GEMV, FP32 operators, iso3 KV store, and the GPU layer runner.
add_library(gpu_cuda STATIC
  src/cuda/gemv.cu src/cuda/dp4a_gemv.cu src/cuda/ops.cu src/cuda/iso3_quant.cu src/cuda/q8_gemv.cu src/cuda/gpu_layers.cu)
target_include_directories(gpu_cuda PUBLIC src/cuda src/common src/packer)
target_link_libraries(gpu_cuda PUBLIC CUDA::cudart)
# The iso3 quantizer must reproduce the CPU bytes exactly, so fused multiply-add contraction is off in that file only.
set_source_files_properties(src/cuda/iso3_quant.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")

add_executable(q8_gemv_parity tests/cuda/q8_gemv_parity.cu)
target_include_directories(q8_gemv_parity PRIVATE "${LLAMA_SRC}/ggml/include")
target_link_libraries(q8_gemv_parity PRIVATE gpu_cuda gguf_index ${GGML_LIB} ${GGML_BASE_LIB} CUDA::cudart)
set_target_properties(q8_gemv_parity PROPERTIES BUILD_RPATH "${LLAMA_BUILD}/bin")

add_executable(gemv_parity tests/cuda/gemv_parity.cu)
target_include_directories(gemv_parity PRIVATE "${LLAMA_SRC}/ggml/include")
target_link_libraries(gemv_parity PRIVATE gpu_cuda qmat gguf_index ${GGML_CPU_LIB} ${GGML_LIB} ${GGML_BASE_LIB} CUDA::cudart pthread)
set_target_properties(gemv_parity PROPERTIES BUILD_RPATH "${LLAMA_BUILD}/bin")

add_executable(gpu_layers_parity tests/cuda/gpu_layers_parity.cpp)
target_link_libraries(gpu_layers_parity PRIVATE gpu_cuda gguf_index CUDA::cudart)
set_target_properties(gpu_layers_parity PROPERTIES BUILD_RPATH "${LLAMA_BUILD}/bin")

add_executable(iso3_gpu_test tests/cuda/iso3_gpu_test.cu)
target_link_libraries(iso3_gpu_test PRIVATE gpu_cuda iso3_ref CUDA::cudart)
