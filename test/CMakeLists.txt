add_subdirectory(qir)

# collect all test files
file(GLOB TEST_FILES "*.cpp")

# add test executable
package_add_test(mqt-mlir-test MQT::MLIR ${TEST_FILES})
