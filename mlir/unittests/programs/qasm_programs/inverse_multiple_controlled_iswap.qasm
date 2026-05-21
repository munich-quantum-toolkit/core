OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
inv @ ctrl(2) @ iswap q[0], q[1], q[2], q[3];
