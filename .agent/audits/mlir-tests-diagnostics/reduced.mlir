module {
 func.func @f(%q: !qco.qubit) -> !qco.qubit {
 %a = qco.h %q : !qco.qubit -> !qco.qubit
 %b = qco.h %a : !qco.qubit -> !qco.qubit
 return %b : !qco.qubit
 }
}
