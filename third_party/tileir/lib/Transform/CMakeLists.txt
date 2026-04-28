set(CUDA_TILE_INSTALL_DIR $ENV{CUDA_TILE_INSTALL_DIR})

add_triton_library(TritonTileIRTransforms
  RewriteAssumeWithCudaTile.cpp
  LiftTTCFToSCF.cpp
  AutoGenMemoryToken.cpp
  PolycubeBlockOrder.cpp

  DEPENDS
  TritonTileIRTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  MLIRControlFlowToSCF
  MLIRControlFlowDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRUBDialect
  TritonIR
  Utils
  ${CUDA_TILE_INSTALL_DIR}/lib/libCudaTileDialect.a
)
