# collect all test files
file(GLOB_RECURSE QIR_BACKEND_TEST_FILES "*.cpp")

# add test executable
package_add_test(mqt-qir-backend-test MQT::QIRBackend ${QIR_BACKEND_TEST_FILES})
