# collect all test files
file(GLOB_RECURSE TEST_FILES "*.cpp")

# add test executable
package_add_test(mqt-mlir-test MQT::MLIR ${TEST_FILES})
