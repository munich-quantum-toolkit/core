OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[1] c0;
bit[2] c1;
measure q[0] -> c0[0];
measure q[1] -> c1[0];
measure q[2] -> c1[1];
