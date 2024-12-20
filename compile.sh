export LLVM_DIR="$(pwd)/external/llvm-16/install"
export CC="${LLVM_DIR}/bin/clang"
export CXX="${LLVM_DIR}/bin/clang++"

export TAICHI_CMAKE_ARGS="-DTI_BUILD_TESTS:BOOL=ON
    -DTI_WITH_BACKTRACE:BOOL=ON
    -DTI_WITH_CUDA:BOOL=ON 
    -DTI_WITH_METAL:BOOL=OFF 
    -DUSE_MOLD:BOOL=ON
    -DCMAKE_C_COMPILER_LAUNCHER=sccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=sccache"
