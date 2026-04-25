# Pipeline

Current front-end compiler pipeline looks like the following:

> <u>Input</u> is an input, **Bold** is a step, *Italic* is an artefact.
>
> -> is used by, => produces

1. <u>Source code</u> -> [**Lexing & Parsing**][step-parsing] => *AST*

    Understand the form of the user source code.

2. *AST* -> **Item Collection**, **Item Resolution** => *NameEnvironment*, *ResolutionMap*

    We first construct the reachable environment and generate `DefId`s for each item.
    Then we go through all the `Path`s and attach meaning to them, i.e. the item or local binding they refer to.

    It makes sense to collect on the AST as it is used to provide diagnostics to
    user code and AST is closer to what the user wrote.

3. *AST*, *ResolutionMap* -> **AST Lowering** => *HIR*

    We lower the *AST* to a flat structure that is the *HIR*. HIR doesn't care
    about structure and is much more convenient further down the (pipe)line.

4. *HIR* -> **Type Collection** => *TypeEnvironment*

    Go through each item and compute their `ty::EarlyItemTy`.

    Once, the collection phase is done, these are immediately resolved to `ty::LateTy`s.

5. *HIR*, *NameEnvironment*, *TypeEnvironment* -> [**Inference**][step-tycheck] (per function) => *InferResult*

6. *HIR*, *InferResult* -> [**Code Generation**][step-codegen] => *Object*

[step-parsing]: ./pipeline/parsing.md
[step-tycheck]: ./pipeline/type-checking.md
[step-codegen]: ./pipeline/code-generation.md
