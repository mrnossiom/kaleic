# Future

## Language Abstract Machine

One thing that helped me to understand the concept of pointer provenance,
  was to realise that each language acts in its own abstract machine.
An abstract machine is emulated through an interpreter.

This is why we have language specifications and undefined behaviour.
Specifications, among other things, define the semantics of this abstract machine.

It is easy to overlook the abstract machine because its model and behaviour can map closely to the underlying implementation.
This is especially true for a language like Rust or C++ that are designed to be systems programming languages.
Whether the language is implemented as an interpreter or compiled down to machine code,
  you need to understand how this translation works to have a better control on performance, or <...>.
By the way, it is easy to argue that machine code is "just" a lower level language that is in turn interpreted by the CPU.

<!-- In the case of pointer provenance: -->
<!-- Rust defines a [model][rust-memory-model] (not finished yet) for memory that uses bytes as the minimal unit of memory. These bytes maybe initialized or not. -->
<!-- This cleanly maps to (things?) like CHERI. -->

[rust-memory-model]: https://doc.rust-lang.org/nightly/reference/memory-model.html

Among the implementations of Rust, one is *miri* (MIR Interpreter).

