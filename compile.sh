export LLVM_DIR="$(pwd)/external/llvm-15/install"

export CC="${LLVM_DIR}/bin/clang"
export CXX="${LLVM_DIR}/bin/clang++"

# sonicflux: one of OPENGL/VULKAN must be enabled to build GGUI
export TAICHI_CMAKE_ARGS="-DTI_BUILD_TESTS:BOOL=ON
    -DTI_WITH_BACKTRACE:BOOL=ON
    -DTI_WITH_CUDA:BOOL=ON 
    -DTI_WITH_GGUI:BOOL=ON
    -DTI_WITH_METAL:BOOL=OFF
    -DTI_WITH_OPENGL:BOOL=OFF
    -DTI_WITH_VULKAN:BOOL=ON
    -DUSE_MOLD:BOOL=ON
    -DUSE_STDCPP:BOOL=OFF
    -DCMAKE_C_COMPILER_LAUNCHER=sccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=sccache
    -DCLANG_EXECUTABLE=${LLVM_DIR}/bin/clang"
