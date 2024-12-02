# collect all test files
file(GLOB_RECURSE QIR_BACKEND_TEST_FILES "*.cpp")

# add test executable
package_add_test(mqt-qir-backend-test MQT::QIRBackend ${QIR_BACKEND_TEST_FILES})

# ##################################################################################################
# add tests for example qir circuits in `examples/qir`
# ##################################################################################################

# macro to add a test executable for one qir circuit
macro(ADD_QIR_CIRCUIT target_name circuit_path)
  if(NOT TARGET ${target_name})
    # Add a custom command to compile the .ll file to .o
    get_filename_component(circuit_name ${circuit_path} NAME_WE)
    add_custom_command(
      OUTPUT ${circuit_name}.o
      COMMAND clang -c ${circuit_path} -o ${circuit_name}.o
      DEPENDS ${circuit_path}
      COMMENT "Compiling ${circuit_path} to ${circuit_name}.o")
    add_executable(${target_name} ${circuit_name}.o)
    target_link_libraries(${target_name} PRIVATE MQT::QIRBackend)
  endif()
endmacro()

# function to convert camel case to dash case
function(camel_to_dash_lowercase input output)
  string(REGEX REPLACE "([A-Z])" "-\\1" result "${input}")
  string(TOLOWER "${result}" result)
  string(REGEX REPLACE "^-+" "" result "${result}")
  set(${output}
      "${result}"
      PARENT_SCOPE)
endfunction()

# add tests for example qir circuits
set(QIR_EXAMPLES_DIR ${PROJECT_SOURCE_DIR}/examples/qir)
file(GLOB QIR_EXAMPLES "${QIR_EXAMPLES_DIR}/*.ll")
foreach(QIR_EXAMPLE ${QIR_EXAMPLES})
  get_filename_component(QIR_EXAMPLE_NAME ${QIR_EXAMPLE} NAME_WE)
  camel_to_dash_lowercase(${QIR_EXAMPLE_NAME} QIR_EXAMPLE_NAME)
  add_qir_circuit(mqt-qir-${QIR_EXAMPLE_NAME}-ll ${QIR_EXAMPLE})
endforeach()
