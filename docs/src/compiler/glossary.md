# Glossary

There are many core types that are used throughout this compiler.
Here is a non-exhaustive list of them and their use.

## Contexts

- `SessionCtx` (scx)

  Holds information needed for the whole compiler session.
  Things such as options, diagnostics, debug output.

- `DiagnosticCtx` (dcx)

  Contains all the diagnostic accumulation and emission related code.

- `TyCtx` (tcx)

  Holds information needed for type inference, type check and their results.

## IDs

- `session::Symbol`

  Created when lexing files. It is an index to interned identifiers and strings.

- `session::Span` and `session::BytePos`

  Used extensively throughout the different intermediate representations to show good diagnostics to the user.

- `ast::NodeId`

  Created during the parsing step.

- `resolve::DefId` <!-- and `resolve::GlobalDefId` (later) -->

  Created during the collection step.

- `hir::NodeId`

  Created during the lowering step.

  - `hir::ExprId`: they wrap normal `NodeId` but ensure that they point to an expression.

## `TyKind`s

- `hir::TyKind`

  Holds the item definition of a type.

- `ty::TyKind<InferKind, RefKind>`

  Holds the concrete definition of a type.

  - `ty::EarlyItemTy` (`ty::TyKind<NoInfer, DefId>`)

  - `ty::InferTy` (`ty::TyKind<Infer, NoRef>`)

    same but keeps an extra field to store type variables in the inference step

  - `ty::LateTy` (`ty::TyKind<NoInfer, NoRef>`)
