//! # Kaleidoscope

pub mod attrs;
pub mod driver;
pub mod pretty_print;
pub mod session;
pub mod symbols;

pub mod ast;
pub mod lexer;
pub mod parser;

pub mod collect;
pub mod hir;
pub mod lowerer;
pub mod resolve;

pub mod doc;
pub mod inference;
pub mod ty;

pub mod codegen;
// TODO: remove ffi module
pub mod ffi;

/// Used when reaching a branch that breaks an assumption made
#[macro_export]
macro_rules! bug {
	($msg:tt) => {
		panic!("ICE: {}", format_args!($msg))
	};
}
