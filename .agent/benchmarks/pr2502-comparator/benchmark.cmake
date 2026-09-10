# Include temporarily at the end of mlir/unittests/CMakeLists.txt.
set(comparator_benchmark "${PROJECT_SOURCE_DIR}/.agent/benchmarks/pr2502-comparator")
execute_process(
  COMMAND git show db95f4817b2498fd5c60e4cf7bf0f23accb81b24:mlir/unittests/Support/IRVerification.cpp
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/baseline-comparator.cpp"
  COMMAND_ERROR_IS_FATAL ANY)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/baseline-comparator.cpp" PROPERTIES
  COMPILE_DEFINITIONS "areModulesEquivalentWithPermutations=areModulesEquivalentBaseline")
add_executable(pr2502-comparator "${comparator_benchmark}/comparator.cpp"
  "${CMAKE_CURRENT_BINARY_DIR}/baseline-comparator.cpp")
target_link_libraries(pr2502-comparator PRIVATE MLIRTestSupport MLIRParser MLIRFuncDialect MLIRQCOUtils)
mqt_mlir_target_use_project_options(pr2502-comparator)
