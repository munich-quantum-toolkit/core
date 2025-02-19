# collect all test files
file(GLOB_RECURSE QIR_BACKEND_TEST_FILES "*.cpp")

# add test executable
package_add_test(mqt-qir-backend-test MQT::QIRBackend ${QIR_BACKEND_TEST_FILES})

# ##################################################################################################
# add tests for example qir circuits in `examples/qir`
# ##################################################################################################

# include helper macros
include(${PROJECT_SOURCE_DIR}/cmake/Macros.cmake)

# add tests for example qir circuits
set(QIR_EXECUTABLES "")
set(QIR_EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples/qir)
file(GLOB QIR_EXAMPLES "${QIR_EXAMPLES_DIR}/*.ll")
foreach(QIR_EXAMPLE ${QIR_EXAMPLES})
  get_filename_component(QIR_EXAMPLE_NAME ${QIR_EXAMPLE} NAME_WE)
  camel_to_dash_lowercase(${QIR_EXAMPLE_NAME} QIR_EXAMPLE_NAME)
  add_qir_circuit(mqt-qir-${QIR_EXAMPLE_NAME}-ll ${QIR_EXAMPLE})
  list(APPEND QIR_EXECUTABLES "$<TARGET_FILE:mqt-qir-${QIR_EXAMPLE_NAME}-ll>")
  add_dependencies(mqt-qir-backend-test mqt-qir-${QIR_EXAMPLE_NAME}-ll)
endforeach()
# transform QIR_EXECUTABLES to comma separated list of string ("...")
string(REPLACE ";" "\",\"" QIR_EXECUTABLES "\"${QIR_EXECUTABLES}\"")
# add all executables as initializer list of std::filesystem::path's
target_compile_definitions(mqt-qir-backend-test PRIVATE TEST_EXECUTABLES=${QIR_EXECUTABLES})
