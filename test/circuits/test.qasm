// i 0 1 2 3
// o 0 1 2 3
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
qubit[2] r;
bit[2] c;
bit[2] d;

gphase(pi/4);
id q[0];
h q[0];
x q;
x q[0];
y q[0];
z q[0];
r(1, 2) q[0];
rx(pi/2) q[0];
rx(pi/4) q[0];
ry(pi/4) q[0];
ry(pi/2) q[0];
rz(-pi/8) q[0];
rz(-pi/0.7854) q[0];
s q[1];
sdg q[1];
sx q[1];
sxdg q[1];
U(0,0,-pi/4) q[1];
U(0,0,pi/4) q[0];
U(pi/2,pi/2,-pi/2) q[0];
U(pi/2,-pi/2,pi/2) q[0];

cx q[0],r[0];
cx q[0],r;
cx q,r[0];
cx q,r;

u3(1,2,3) r[0];
u2(1,2) r[1];
u1(1) q[0];
barrier q;
measure q -> c;
measure r[0] -> d[0];
measure q -> d;
reset q;

ctrl @ z q[0],q[1];
ctrl @ y q[0],q[1];
ctrl @ h q[0],q[1];
ctrl @ s q[0],q[1];
ctrl @ sdg q[0],q[1];
ctrl @ t q[0],q[1];
ctrl @ tdg q[0],q[1];
ctrl(2) @ x q[0],q[1],r[0];
ctrl @ rz(pi/8) q[0],q[1];
ctrl @ u1(pi/8) q[0],q[1];
ctrl @ u3(pi,0,pi) q[0],q[1];
swap q[0],q[1];
