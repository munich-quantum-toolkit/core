OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
h q[0];
bit c = measure q[0];
if (c) {
  x q[0];
} else {
  z q[0];
}
