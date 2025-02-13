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
set(QIR_EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples/qir)
file(GLOB QIR_EXAMPLES "${QIR_EXAMPLES_DIR}/*.ll")
foreach(QIR_EXAMPLE ${QIR_EXAMPLES})
  get_filename_component(QIR_EXAMPLE_NAME ${QIR_EXAMPLE} NAME_WE)
  camel_to_dash_lowercase(${QIR_EXAMPLE_NAME} QIR_EXAMPLE_NAME)
  add_qir_circuit(mqt-qir-${QIR_EXAMPLE_NAME}-ll ${QIR_EXAMPLE})
endforeach()
