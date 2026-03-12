# Pipeline

Current front-end compiler pipeline looks like the following:

> <u>Input</u> is an input, **Bold** is a step, *Italic* is an artefact.
>
> -> is used by, => produces

1. <u>Source code</u> -> [**Lexing & Parsing**][step-parsing] => *AST*

    Understand the form of the user source code.

2. *AST* -> **Item Collection** => *NameEnvironment*

    Go through each item and associate names of items to `ast::NodeId`s.

    It makes sense to collect before **AST Lowering** because

    - this information can be used during the lowering for path resolution
    - it is used to provide diagnostics to user code, AST is closer to what the user wrote

3. *AST* -> **AST Lowering** => *HIR*

    We lower the *AST* to a flat structure (soon™) that is the *HIR*.

    HIR is much more usable in the context of a compiler.

4. *HIR* -> **Item Typing** => *TypeEnvironment*

    Go through each item and associate `hir::NodeId`s to types.

    - `struct Foo {id: u32}` has type `TyKind::Struct {fields: [<NodeId of u32>]}`
    - `fn bar(id: u32)` has type `TyKind::Func {inputs: [<NodeId of u32>], output: ..}`

5. *HIR*, *NameEnvironment*, *TypeEnvironment* -> [**Inference**][step-tycheck] (per function) => *InferResult*

    Introduce a resolution step before to resolve all `ast::Path` in bodies?

6. *HIR*, *InferResult* -> [**Code Generation**][step-codegen] => *Object*

[step-parsing]: ./pipeline/parsing.md
[step-tycheck]: ./pipeline/type-checking.md
[step-codegen]: ./pipeline/code-generation.md
