ystade
3 days ago
Maintainer
// Multiple controls
mqtref.ctrl %c0, %c1 {
mqtref.x %q0 // Toffoli gate (CCX)
}
Should the ctrl modifier really support multiple controls at once? Still, I can also wrap control in control, and so forth. At least there should be a fixed canonical form.

Edit: Also the canonical order of pos. and neg. controls should be fixed.

4 replies 1 new
@burgholzer
burgholzer
3 days ago
Maintainer
Author
I think this is touched on somewhere above.

NormalizationPass: Enforces canonical modifier ordering and eliminates redundancies

Reorders nested modifiers to canonical form (ctrl → pow → inv)
Eliminates identity operations (pow(X, 1) → X, inv(inv(X)) → X)
Merges adjacent compatible modifiers
Simplifies power modifiers wherever feasible (pow(RZ(pi/2), 2) → RZ(pi))
Canonicalization ensure that there is a normal form of modifiers, that adjacent modifiers are merged.
I believe that aggregation of controls in a single modifier makes sense because it helps to achieve a compact IR that needs less traversal. I also don't really see a downside of doing that.

@ystade
ystade
3 days ago
Maintainer
Agreed, then just the precedence of neg. controls must be defined.

@burgholzer
burgholzer
3 days ago
Maintainer
Author
Yeah. Essentially, we the normalization pass above should read

(ctrl → negctrl → pow → inv)
I was briefly thinking (and had that in the RFC for a short period) if it would make sense to combine the ctrl and negctrl wrapper operations and a single ctrl wrapper that has both negative and positive controls similar to how we currently have it in our UnitaryOp. But this did not feel as clean to me as the separate solution.
We could also implement a transformation pass that turns negative controls into positive controls sandwhiched by X gates, so that there would only be positive controls anymore.
Naturally, the opposite direction is also imaginable, i.e., aggregating controls sandwhiched by X gates into a negative control.
Any opinions on that?

@ystade
ystade
4 hours ago
Maintainer
Yeah, I was already thinking about the same but could not imagine what flows better with the design of MLIR. My proposal would be to go with it as it stands right now and keep the other option in mind if the current one does not feel ergonomic.

---

In general, we might want to discuss, whether we actually need gates like x, rx, ... or whether we adopt the OpenQASM 3 way and just have on u3 gate plus all modifiers.

Or whether this is a sensible canoncalization pass? And which direction actually is the canonicalisation, to or from only u3 gates?

4 replies 1 new
@burgholzer
burgholzer
3 days ago
Maintainer
Author
Just the U3 gate alone is quite cumbersome to use. And defining everything in terms of the U3 gate also gets annoying rather quickly as it invites numerical errors.
I am also not quite sure I'd like that the treatment of single-qubit gates would be different than for multi-qubit gates, because I doubt, we will describe two-qubit gates by their 17+ parameter KAK decomposition.

Even OpenQASM (and Qiskit) defines a standard library that is being heavily used and includes all kinds of gates. Over there, it's "just" the definition of the unitary of the gates that can be recursively boiled down to the global phase gate and modifiers.

Lastly, dropping the individual gates also makes traits such as "Hermitian" or "Diagonal" much harder to determine. And I would believe these are quite useful in the analysis of programs.

@ystade
ystade
3 days ago
Maintainer
The last point is a very good point. Perhaps, we should have a Canonicalization pass that transforms every gate in the simplest possible (similar to the behaviour when parsing QASM files in MQT Core currently). Maybe even the other way around is needed for passes that only understand u3 gates and nothing else. For the latter, I do not know whether this is even desired.

@burgholzer
burgholzer
3 days ago
Maintainer
Author
This canonicalization pass is something that, to some degree, fits well with the custom gate definitions, which are explained in one of the sections. There could be an inlining pass that inlines gate definitions until it reaches well known basis gates (=our standard library).

At the same time, we have some kind of canonicalization for individual gates as part of MQT Core IR already. Namely, here:

core/src/ir/operations/StandardOperation.cpp

Lines 36 to 186 in 9109b56

OpType StandardOperation::parseU3(fp& theta, fp& phi, fp& lambda) {
if (std::abs(theta) < PARAMETER_TOLERANCE &&
std::abs(phi) < PARAMETER_TOLERANCE) {
parameter = {lambda};
return parseU1(parameter[0]);
}

if (std::abs(theta - PI_2) < PARAMETER_TOLERANCE) {
parameter = {phi, lambda};
return parseU2(parameter[0], parameter[1]);
}

if (std::abs(lambda) < PARAMETER_TOLERANCE) {
lambda = 0.L;
if (std::abs(phi) < PARAMETER_TOLERANCE) {
checkInteger(theta);
checkFractionPi(theta);
parameter = {theta};
return RY;
}
}

if (std::abs(lambda - PI_2) < PARAMETER_TOLERANCE) {
lambda = PI_2;
if (std::abs(phi + PI_2) < PARAMETER_TOLERANCE) {
checkInteger(theta);
checkFractionPi(theta);
parameter = {theta};
return RX;
}

     if (std::abs(phi - PI_2) < PARAMETER_TOLERANCE) {
       phi = PI_2;
       if (std::abs(theta - PI) < PARAMETER_TOLERANCE) {
         parameter.clear();
         return Y;
       }
     }

}

if (std::abs(lambda + PI_2) < PARAMETER_TOLERANCE) {
lambda = -PI_2;
if (std::abs(phi - PI_2) < PARAMETER_TOLERANCE) {
phi = PI_2;
parameter = {-theta};
return RX;
}
}

if (std::abs(lambda - PI) < PARAMETER_TOLERANCE) {
lambda = PI;
if (std::abs(phi) < PARAMETER_TOLERANCE) {
phi = 0.L;
if (std::abs(theta - PI) < PARAMETER_TOLERANCE) {
parameter.clear();
return X;
}
}
}

// parse a real u3 gate
checkInteger(lambda);
checkFractionPi(lambda);
checkInteger(phi);
checkFractionPi(phi);
checkInteger(theta);
checkFractionPi(theta);

return U;
}

OpType StandardOperation::parseU2(fp& phi, fp& lambda) {
if (std::abs(phi) < PARAMETER_TOLERANCE) {
phi = 0.L;
if (std::abs(std::abs(lambda) - PI) < PARAMETER_TOLERANCE) {
parameter.clear();
return H;
}
if (std::abs(lambda) < PARAMETER_TOLERANCE) {
parameter = {PI_2};
return RY;
}
}

if (std::abs(lambda - PI_2) < PARAMETER_TOLERANCE) {
lambda = PI_2;
if (std::abs(phi + PI_2) < PARAMETER_TOLERANCE) {
parameter.clear();
return V;
}
}

if (std::abs(lambda + PI_2) < PARAMETER_TOLERANCE) {
lambda = -PI_2;
if (std::abs(phi - PI_2) < PARAMETER_TOLERANCE) {
parameter.clear();
return Vdg;
}
}

checkInteger(lambda);
checkFractionPi(lambda);
checkInteger(phi);
checkFractionPi(phi);

return U2;
}

OpType StandardOperation::parseU1(fp& lambda) {
if (std::abs(lambda) < PARAMETER_TOLERANCE) {
parameter.clear();
return I;
}
const bool sign = std::signbit(lambda);

if (std::abs(std::abs(lambda) - PI) < PARAMETER_TOLERANCE) {
parameter.clear();
return Z;
}

if (std::abs(std::abs(lambda) - PI_2) < PARAMETER_TOLERANCE) {
parameter.clear();
return sign ? Sdg : S;
}

if (std::abs(std::abs(lambda) - PI_4) < PARAMETER_TOLERANCE) {
parameter.clear();
return sign ? Tdg : T;
}

checkInteger(lambda);
checkFractionPi(lambda);

return P;
}

void StandardOperation::checkUgate() {
if (parameter.empty()) {
return;
}
if (type == P) {
assert(parameter.size() == 1);
type = parseU1(parameter.at(0));
} else if (type == U2) {
assert(parameter.size() == 2);
type = parseU2(parameter.at(0), parameter.at(1));
} else if (type == U) {
assert(parameter.size() == 3);
type = parseU3(parameter.at(0), parameter.at(1), parameter.at(2));
}
}

This is parsing arbitrary U3 gates and translating them to the simplest matching operation. I do see value in that and have relied on that functionality quite a lot in the past.
At the same time, there has actually been (currently unmet) demand for the opposite direction. Namely, here: #910
While this specific issue only talks about phase gates, I'd probably generalize this to U3.
Do both of these points make sense?

@ystade
ystade
4 hours ago
Maintainer
That is exactly along my reasoning.

---

ystade
3 days ago
Maintainer
// Control queries
bool isControlled();
size_t getNumPosControls();
size_t getNumNegControls();
size_t getNumControls();
I do not get it, I thought the single operation does not have controls anymore. Then, the functions above would already implement a transversal of the wrapping control modifiers. But where does this transversal end? At the boundaries of functions/sequences? Or should it go beyond? Furthermore, why are there no function to check for inv or pow modifiers?

7 replies 7 new
@burgholzer
burgholzer
3 days ago
Maintainer
Author
The unitary interface is intended to be implemented by all concepts here.

The base operations would always report 0 controls.
A control modifier reports its own controls + all controls of the contained operation (which again implements the UnitaryInterface).
pow and inv just forward to the contained operation
seq (up for debate) will report 0 controls as it represents a block of operations conceptionally
unitary definition will also report 0 controls
Hope that covered everything. This seems like a well terminating sequence of rules to me.

As for the inv and pow modifiers: neither of them change the number of qubits of the underlying operation. I haven't really found a use case for exposing these.
inv on base operations will be canonicalized away, inv on inv cancels, inv and pow commute, inv on sequence either remains like it is or is canonicalized to reversed sequence of inverted operations; inverse of a unitary definition is up for debate.
Do you have a use case in mind where having generic access to these details in the interface makes sense? And a proposal for the interface function?

@ystade
ystade
3 days ago
Maintainer
A control modifier reports its own controls + all controls of the contained operation (which again implements the UnitaryInterface).
This gives rise to another question: So, the block within a ctrl modifier is only allowed to contain exactly one operation. I was assuming that this can be an arbitrary block...

@ystade
ystade
3 days ago
Maintainer
Do you have a use case in mind where having generic access to these details in the interface makes sense? And a proposal for the interface function?

No, I do have not and what you wrote before makes absolutely sense. We can always add such functionality if we are missing it.

@burgholzer
burgholzer
3 days ago
Maintainer
Author
A control modifier reports its own controls + all controls of the contained operation (which again implements the UnitaryInterface).
This gives rise to another question: So, the block within a ctrl modifier is only allowed to contain exactly one operation. I was assuming that this can be an arbitrary block...

It can be anything that implements the UnitaryInterface.
So it can be a single basis operation, another modifier operation, a unitary, or... a mqtref.seq, which covers the use case of a modifier applying to a group of operations.
I believe this covers all imaginable scenarios.

@ystade
ystade
4 hours ago
Maintainer
Makes sense. However, then it is not entirely clear to me, how to count the controls of a seq.

@DRovara
DRovara
52 minutes ago
Collaborator
I'd say a seq does not have any controls, as its essentially a one-time-use custom gate, right?

@burgholzer
burgholzer
30 minutes ago
Maintainer
Author
I'd say a seq does not have any controls, as its essentially a one-time-use custom gate, right?

Exactly! A sequence will always report 0 controls.

---

ystade
3 days ago
Maintainer
8.1 Matrix-Based Definitions

Is that needed? I am not sure whether we overengineer here our dialect. If I am not mistaken, the u3 gate should already cover all possible unitary gates. When defining gates via matrices, one would also need to implement a validation pass, validating that every matrix is unitary, something that is by default satisfied with the u3 gate.

1 reply
@burgholzer
burgholzer
3 days ago
Maintainer
Author
The u3 gate only covers single-qubit matrices. I'd consider it feasible to represent up to 3-qubit gates (maybe even more) as a unitary and I would imagine that especially the 1- and 2-qubit cases could be popular.
This point was mainly inspired by some discussions at QCE with Ed from the BQSKit team and the motivation is mostly grounded in numerical synthesis, where it is just more efficient and straight forward to describe the unitary of the operation as opposed to a sequence of (known) gates that make up the unitary functionality.

Overall, I think this is not a must-have, but a very-nice-to-have. And it is definitely a unique and unambiguous way of describing the functionality of a gate.

The validation pass is a good idea. Although I'd personally also put some of the responsibility for that on the developer that uses these custom gates.

---

Regarding testing: I think we should consider switching to the CHECK-NEXT directive to ensure that we don't have unexpected statements in our IR.

3 replies
@burgholzer
burgholzer
3 days ago
Maintainer
Author
Yeah. I agree. Maybe even taking this one step further: we should compare entire programs are fully equivalent to what we would expect. We define complete input programs and the passes that should be run on them. Based on that there is a clear expected output program that we could be writing down explicitly.
Maybe with the help of the circuit builder, we can actually not do this on top of the textual IR at all, but rather construct these test cases programmatically in a googletest suite similar to the translation tests.

@DRovara
DRovara
3 days ago
Collaborator
Actually I really like the idea of using the builders to construct MLIR representations of expected outcomes and then using those to check for equivalence in normal unit tests.

In fact, this is the way I have seen such language tests being used outside of the MLIR framework and it just feels really clean and straightforward - if we ever change the representation, the tests won't be affected as the builders should still be able to construct equal programs.

@burgholzer
burgholzer
3 days ago
Maintainer
Author
In fact, this is the way I have seen such language tests being used outside of the MLIR framework and it just feels really clean and straightforward - if we ever change the representation, the tests won't be affected as the builders should still be able to construct equal programs.

That is a good point! And a great argument for stability. I will try to include this in the overall project plan and afterwards will hide this conversation as resolved.

---

denialhaag
3 days ago
Collaborator
We should also think about which, if any, default passes we want to run. The -canonicalize and -remove-dead-values passes seem useful in most cases to me. Slightly related to the comment above, this would, for example, ensure that constants are always defined at the top of the module in our tests.

1 reply
@burgholzer
burgholzer
3 days ago
Maintainer
Author
Yes, totally. Especially if we define further canonicalization for the operations in our dialects.
-canonicalize is a no-brainer in my opinion; we should definitely do that.
Probably the same holds for the -remove-dead-values pass.

---

4.2

mqtref.u3 %q0 {theta = 0.0, phi = 0.0, lambda = 3.14159} // Generic single-qubit gate
Is this naming scheme of parameters actually valid? Does this still correlate with the parameter-arity traits?

The existing solution also (semi-)elegantly handles situations where some parameters may be static and others dynamic operands. Can this still be done with we represent the gates like this?
Or maybe in general, how do such gates still support dynamic operands for angles etc?
Sorry, just saw that this is covered in 4.3

mqtref.rx %q0, %angle : f64
But some extra questions to that:
Do we maybe want some syntactic sugar to distinguish %q0 as a qubit argument from %angle as a parameter?
And why do we need the f64 type indication for the angle, but not the qubit?

1 reply
@burgholzer
burgholzer
3 days ago
Maintainer
Author
I think a lot of the examples are inconsistent at the moment because they are hallucinations.
I'd actually like to keep the parameter handling almost exactly the same compared to now
Syntactic sugar for distinguishing between parameters and qubits makes a lof of sense. I do like how we currently handle this.
The type indication is supposedly just a hallucination.
I'll try to get a little bit more consensus on some of the general design decisions before I try to consolidate the examples and flesh them out more.

---

denialhaag
3 days ago
Collaborator
Just to have it mentioned: We already have a way to translate QuantumComputation IR containing classically controlled operations to mqtref. We do not use custom operations for that, but we rely on the scf dialect. If I'm not mistaken, we have not yet attempted to convert a mqtref module containing classically controlled operations to mqtopt. I'm not sure if this is something we also want to be part of the C++builder.

3 replies 1 new
@burgholzer
burgholzer
3 days ago
Maintainer
Author
Hm. I was, maybe naively, assuming that any program that we can currently express through the QuantumComputation -> MQTRef translation can also be translated to MQTOpt. And I am pretty sure we actually can because this is what @DRovara's qubit reuse relies on. But maybe I am mistaken here.
Please feel free to chime in on this.

In my mind, the C++ builder API would be very, very similar to the QuantumComputation API. It would let you add quantum and classical registers, quantum gates (potentially involving classical controls), measurements and resets. That API would practically be the evolution/continuation of the QuantumComputation API.

@DRovara
DRovara
3 days ago
Collaborator
In MQTOpt we can use scf.if the same way as in MQTDyn, we just need to yield the result afterwards.

@denialhaag
denialhaag
3 days ago
Collaborator
Okay, this sounds good to me then! 🙂

---

DRovara
3 days ago
Collaborator
Regarding section 5: I think this idea of "maintaining simple dataflow analysis" should not be taken for granted. I'm not saying that this is reason enough to refute the whole concept, but let's look at this example:

%q0_0 = mqtopt.alloc : !mqtopt.qubit
%c0_out, %q0_1 = mqtopt.ctrl %c0_in {
%q0_new = mqtopt.x %q0_0 : !mqtopt.qubit
mqtopt.yield %q0_new : !mqtopt.qubit
} : (!mqtopt.qubit, !mqtopt.qubit) -> (!mqtopt.qubit, !mqtopt.qubit)
%q0_2 = mqtopt.x %q0_1 : !mqtopt.qubit
Now, if we run a transformation pass on the final x gate, how do we follow the def-use-chain to reach the alloc?

First, we can just access the definingOp of %q0_1 (the input of the x gate)
Now, we are at mqtopt.ctrl. How do we find out where %q0_1 came from? We programmatically need to search for the yield
Now at the yield we can once again access definingOp to get the x gate above.
Finally, from there, we can again access definingOp to get the alloc.
All in all, this is certainly not impossible, but this idiom will have to be implemented for every single transformation pass.

Further, this is just one example where we might "struggle" that comes to mind immediately. There might be further not so obvious ones too.
For instance, how easy is it if we need to replace controlled operation by a non-controlled one? rewriter.replaceOp will no longer work, because the def-use-chain is no longer consistent due to the break at the yield.

Again, not saying this means that the suggestion is unusable, but it will undoubtedly make stuff at least a bit harder.

2 replies
@burgholzer
burgholzer
3 days ago
Maintainer
Author
@DRovara: Why do sequences require block arguments for inputs and controlled gates do not?
Thinking about it, requiring block arguments for controlled gates as well might eliminate (or at least reduce) the issue

Very good point. I think this is merely an oversight. The control modifiers should definitely also receive block arguments similar to the sequence threading.
Do we also need to perform similar threading for the inv and pow modifier?

@DRovara
DRovara
3 days ago
Collaborator
Good point. Probably we need the same for all of these blocks.

---

DRovara
3 days ago
Collaborator
7.1 UnitaryOpInterface

For mqtopt operations we definitely need a getCorrespondingInput(outQubit) and getCorrespondingOutput(inQubit) function (see my qubit reuse PR) that takes a qubit (either an output or input of the operation) and gets the corresponding qubit at the other side (i.e. the result it turns into or the input it is based on)

Also, I believe getStaticParameters and getDynamicParameters is not enough to correctly identify parameters when we have a parameter mask. It would be cool to already have a function that handles that and gives you the corresponding parameter correctly, whether it be static or dynamic.

3 replies
@burgholzer
burgholzer
3 days ago
Maintainer
Author
For mqtopt operations we definitely need a getCorrespondingInput(outQubit) and getCorrespondingOutput(inQubit) function (see my qubit reuse PR) that takes a qubit (either an output or input of the operation) and gets the corresponding qubit at the other side (i.e. the result it turns into or the input it is based on)

That makes a lot of sense! I will add this in the next iteration on the document!

Also, I believe getStaticParameters and getDynamicParameters is not enough to correctly identify parameters when we have a parameter mask.

Agreed. I'll think a bit about it. Probably something rather close to what we have at the moment would work sufficiently well.

It would be cool to already have a function that handles that and gives you the corresponding parameter correctly, whether it be static or dynamic.

I am not quite sure I follow. What is it that you would expect here?

@DRovara
DRovara
3 days ago
Collaborator
I am not quite sure I follow. What is it that you would expect here?

I'm not quite sure if it makes sense, but something like getParameter(2) returning the second parameter, whether it is an operand or a static attribute.

@burgholzer
burgholzer
3 days ago
Maintainer
Author
Ah. Yeah, something like that would make sense. The only question is what that getter would be returning. In the dynamic case, it would be a mlir::Value, in the other case it would be an f64 attribute.
One could think about returning a std::variant of exactly those types.

---

DRovara
3 days ago
Collaborator
8.2 Composite Definitions

Just for clarity, we should probably also suggest a syntax for mqtopt here (by the way, this example also uses cnot)

// Define a gate as a sequence of existing operations
mqtopt.gate_def @bell_prep %q0 : !mqtopt.qubit, %q1 : !mqtopt.qubit {
%q0_1 = mqtopt.h %q0 : !mqtopt.qubit
%q1_1, %q0_2 = mqtopt.ctrl %q1 {
%q0_new = mqtopt.x %q0_1 : !mqtopt.qubit
mqtopt.yield %q0_new : !mqtopt.qubit
} : (!mqtopt.qubit, !mqtopt.qubit) -> (!mqtopt.qubit, !mqtopt.qubit)
mqtopt.yield %q0_2, %q1_1 :!mqtopt.qubit, !mqtopt.qubit
}

// Apply the composite gate
%q0_1, %q1_1 = mqtopt.apply_gate @bell_prep %q0_0, %q1_0 : !mqtopt.qubit, !mqtopt.qubit
1 reply
@burgholzer
burgholzer
3 days ago
Maintainer
Author
Agreed! I'll add this to the next iteration.
(already replaced the cnot with a cx for now)

---

DRovara
3 days ago
Collaborator
5.1

Is there any way to simplify the syntax for mqtopt?

%c0_out, %q0_out = mqtopt.ctrl %c0_in {
%q0_new = mqtopt.x %q0_in : !mqtopt.qubit
mqtopt.yield %q0_new : !mqtopt.qubit
} : (!mqtopt.qubit, !mqtopt.qubit) -> (!mqtopt.qubit, !mqtopt.qubit)
Do we really need the (!mqtopt.qubit, !mqtopt.qubit) -> (!mqtopt.qubit, !mqtopt.qubit)?

3 replies 2 new
@burgholzer
burgholzer
3 days ago
Maintainer
Author
I suppose we don't need that. Typically, everything behind the colon is optional as long as we can get the type inference right. This is one of the aspects that will probably become clearer when this moves closer to implementation.

@denialhaag
denialhaag
3 days ago
Collaborator
When I tried getting type inference to work in conjunction with AttrSizedResultSegments, I ran into quite some problems, but I'm sure there's a solution to that. 😌

@DRovara
DRovara
8 hours ago
Collaborator
Maybe at least we can make it so the (!mqtopt.qubit, !mqtopt.qubit) -> part is not necessary, similarly to how we did it for other operations.

Actually, this type annotation seems weird anyways (probably the LLM at fault again): (!mqtopt.qubit, !mqtopt.qubit) -> suggests two inputs being used, but the way it is written, only one variable is directly used by mqtopt.ctrl (%c0_in)

---

DRovara
3 days ago
Collaborator
8.3
The syntax seems a bit messed up here.

// Define a parameterized rotation gate
mqtref.gate_def @custom_rotation %q : !mqtref.qubit attributes {params = ["theta", "phi"]} {
mqtref.rz %q {angle = %phi}
mqtref.ry %q {angle = %theta}
mqtref.rz %q {angle = %phi}
}

// Apply with specific parameters
mqtref.apply_gate @custom_rotation %q0 {theta = 1.57 : f64, phi = 0.78 : f64}`
Why can the rotation gates suddenly take a dynamic operand inside curly braces with angle = ...? That's different from how it was in previous sections.

Also, something feels uncomfortable about defining the variables as strings and then just using them as e.g. %phi. I get the idea as to why we would want that, but I wonder if there is a cleaner approach.

1 reply
@burgholzer
burgholzer
3 days ago
Maintainer
Author
Yeah. Thinking about it, this is very much hallucinated and should be more grounded in reality.
I suppose it's more so about the general idea of being able to define a custom gate with parameters.

---

DRovara
3 days ago
Collaborator
Maybe one last general point:

From a language perspective, I love all of these suggestions.
But I find it highly unlikely that we can just make these changes without major overhauls of our existing transformation passes.
I'm pretty sure that the passes we have so far will not only require rewriting but also rethinking, in some contexts.

That doesn't mean we shouldn't do these changes, but it should be clear in advance.

1 reply
@burgholzer
burgholzer
3 days ago
Maintainer
Author
Oh yeah, definitely. This changes the very fundamentals that we operate on.
So changes will definitely be necessary to the transformations as well. Especially in places where operations are constructed.
I'd still hope that some transformations will remain moderately untouched because of the extensive use of the UnitaryInterface that abstracts over many of the details here.

And I also believe that this is now the culmination of spending over 9 months working on MLIR. To me, this feels like it would build a close-to-feature-complete basis architecture and infrastructure that we can rely on in the future.

---

MatthiasReumann
9 hours ago
Collaborator
Hello there! 👀

Some thoughts I had while reading this.

5.1 Control Operations

Should this really be a %c0? I think this is a mistake, right?

// Single control
mqtref.ctrl %c0 { // <--- Shouldn't this be a quantum value, e.g., %q1?
mqtref.x %q0 // Controlled-X (CNOT)
}

... // Similarly for the other examples. 5) Modifier Operations and 7) Unified Interface Design:

I keep asking myself, because I've been working on this recently, how do I apply these changes to the placement and routing passes (MQTOpt). Right now, a UnitaryInterface acts like a gate - it has inputs and outputs. Easy. An IR-Walker can simply ::advance() when encountering it.

With the proposed changes, which I very much like, it's a bit more cumbersome - or at least less intuitive. A (pre-order) traversal would have to skip all elements inside a UnitaryInterface and only work with the interface (getInputs, getOutputs) of the "shell". However, these interface methods will probably have to traverse the IR internally nonetheless.

I've been thinking if it could make sense to define and implement a custom top-down driver for traversing quantum-classical programs canonically 1. This would have the advantage that details such as the one above are hidden away. Following the same argument as in the routing pass this driver will most likely not be pattern based.

Anyhow, either I'm missing something here (and this isn't really a problem) or the current dialects are advantageous over the changes proposed at least for this one particular aspect.

@DRovara was probably also hinting at this:

Further, this is just one example where we might "struggle" that comes to mind immediately. There might be further not so obvious ones too.

Footnotes
The placement and routing passes already implement a naive-form of such a driver, separately. ↩

5 replies 1 new
@DRovara
DRovara
8 hours ago
Collaborator
Should this really be a %c0? I think this is a mistake, right?

I think it's called "c" for "control", not "classical". But yeah, I definitely also had to do a double-take.

@DRovara
DRovara
8 hours ago
Collaborator
Regarding the transformations: I feel like, what we need is to just try out how much more effort it is to write transformations this way. Unfortunately, that can only really be done once we have implemented everything, I guess.

But yeah, the way you would likely have to do is, if you walk along the def-use chain, is to check, at each step, whether prev->getBlock() == current->getBlock(), and, as long as this is not the case, apply current = current->getBlock()->getParentOp() (more or less, don't remember the names specifically).

That repeated check might very well also lead to a reduced efficiency and uglier code, but I don't really see a way around it.

@DRovara
DRovara
5 hours ago
Collaborator
Maybe to this end, as an extension to "7.1 UnitaryOpInterface", we should also add getPredecessors() or getPredecessor(int operandIndex) or getPredecessor(mlir::Value qubit) as well as getSuccessors(...) methods to the UnitaryInterface that handle this automatically (note that for the successors, the result always has to be a collection and cannot be a single item due to branches allowing multiple users of the same variable).

@burgholzer
burgholzer
4 hours ago
Maintainer
Author
Regarding the first point: yeah, the c here was meant for "control" but is also merely an LLM artifact. I'll try to pay attention to this when rewriting the proposal.

As for the transformations: I was hoping that this kind of hierarchical structuring would work very naturally. Especially since we are performing these transformations on the MQTOpt dialect with value semantics. This should still allow us to fairly easily traverse the def-use chains of the hierarchical operations quite similar to how one would traverse an scf.if instruction. Such instructions consume SSA values and produce new ones.
I don't particularly see the problem of relying on the interface methods; in the end, that's what they are for.
But maybe I am overlooking things here.
As Damian already pointed out, I have a feeling that one can only really tell once one tries implementing a rather large part of this proposal.

@DRovara
DRovara
1 hour ago
Collaborator
This should still allow us to fairly easily traverse the def-use chains of the hierarchical operations quite similar to how one would traverse an scf.if instruction

If we define some sort of block arguments similar to your suggestion for mqtopt.seq, then it might actually work quite naturally (but I still didn't think much about it). If we don't then we have to do it the way we handle scf.if.

The thing is, from my understanding working with scf.if so far, contrary to your statement: yes, scf.if produces new values, but it does not explicitly consume any. For that, we have to look at the corresponding yield operation. That's what I meant in another comment about breaking the def-use chain. Whenever we have scf.if, we have two chains, like this:

op1 --> op2 --> scf.yield
scf.if --> op3 --> op4
And we have to "manually" write the traversal methods so that the chains are merged with the correct yield/if

op1 --> op2 --> scf.if/scf.yield --> op3 --> op4
The same thing would also happen with the control/inv/pow regions.

---

flowerthrower
5 hours ago
Collaborator
Disclaimer: I am not an expert on fault-tolerant quantum computing (so take this with a grain of salt). From what I've seen in inconspiquos and FTQC literature, codes like surface/color codes and LDPC stabilizer codes often use logical code blocks—sometimes as geometric patches, sometimes more abstractly. With the MLIR dialect design in mind, several bundled feature areas seem broadly relevant:

Logical Block and Reconfiguration Semantics:
FTQC protocols depend on defining, manipulating, and reconfiguring logical code blocks (e.g., merge/split operations in lattice surgery, code switching, gauge fixing). E.g. inconspiquos has regions of qubits that can be arbitrarily merged and split.
How this could be done with the proposed model:
The dialect's compositional sequences and user-defined gates allow grouping and modular reuse of operations—these can mimic logical blocks and some forms of reconfiguration via higher-level composition. However, native support for explicit logical block annotation and dynamic reconfiguration (merge/split/code transitions) would require further dialect extensions or metadata.

Bulk Operations and Scheduling:
Expressing cycle-based or parallel operations (like batches of parity checks and measurements) is essential for scalability in LDPC and surface code protocols. E.g., one can apply a gate operation to such a region in inconspiquos (under the hood all qubits in that region are addressed)
How this could be done with the proposed model:
Sequences, loop constructs, and composite gates in the dialect can encapsulate repeated or batch operations, but efficient bulk scheduling primitives for large codes may go beyond what's ergonomic now.

Syndrome Processing, Decoder Integration, and Code-Aware Semantics:
Modern FTQC needs first-class support for syndrome acquisition, Pauli-frame tracking, and integration with classical decoders, along with the ability to annotate protocols with code parameters, gate set constraints, and locality/transversality information.
How this could be done with the proposed model:
The dialect's classical register support and compositional logic enable measurement pipelines and feedback routines. Code-aware semantics and decoder integration could be layered on top with attributes, annotations, or additional IR ops. Though many of these things go beyond the concepts of this IR discussion, at least one should keep them in mind.

Overall, I really like the proposed changes and there is not much from my end that has not already been discussed in the above comments. In order to make it as future-proof as possible, however, it could be sensible to talk about the FT requirements with the error correction folks at the chair. Although I acknowledge that it might be difficult for them to fully follow the discussion here—as it is for me (us) to understand all the error correction details.

1 reply 1 new
@burgholzer
burgholzer
4 hours ago
Maintainer
Author
Thanks for the input on this! Certainly an important point.
From a very birds-eye view, I see no major roadblocks for incorporating these features on top of the existing proposal. As identified, one could probably even "abuse" the existing concepts without much adaptation. However, I have a feeling that one might rather want to define separate dialects and/or operations for some of these concepts.
Having a (separate) discussion on this definitely makes sense; probably after another one or two iterations on the actual proposal.
