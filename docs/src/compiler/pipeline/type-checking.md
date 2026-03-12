# Type Checking

## Reading

- [Typechecker Zoo](https://sdiehl.github.io/typechecker-zoo/introduction.html)

  In particular *Algorithm W* was the first implemented algorithm for function-level type inference.

  I intend to implement a form of *System Fω*.

- [Complete and Easy Bidirectional Typechecking for Higher-Rank Polymorphism](https://www.cl.cam.ac.uk/~nk480/bidir.pdf)

- Rust

  Rust uses a form of System Fω with Bidirectional typechecking implemented using a unification (union-find) algorithm.

  See
  [rustc_hir_analysis](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/index.html),
  [ena (docs.rs)](https://docs.rs/ena)
