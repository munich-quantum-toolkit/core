OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
bit[3] c;
measure q[0] -> c[0];
measure q[0] -> c[1];
measure q[0] -> c[2];
