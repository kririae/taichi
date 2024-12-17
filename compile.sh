export LLVM_DIR="$(pwd)/external/llvm-15/install"
export CC="${LLVM_DIR}/bin/clang"
export CXX="${LLVM_DIR}/bin/clang++"

export TAICHI_CMAKE_ARGS="-DTI_BUILD_TESTS:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_METAL:BOOL=OFF -DUSE_MOLD:BOOL=ON"

