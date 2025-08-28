//! Typed Body IR
//!
//! This is what is handed to codegen
//!

// TODO: remove tbir, use queries on typeck results in codegen using exprids

use crate::{
	ast::{self, BinaryOp, Spanned, UnaryOp},
	hir,
	lexer::LiteralKind,
	session::{Span, Symbol},
	ty,
};

#[derive(Debug, Clone)]
pub struct Block {
	pub stmts: Vec<Stmt>,
	pub ret: Option<Expr>,
	pub ty: ty::TyKind,
	pub span: Span,
	pub id: hir::NodeId,
}

#[derive(Debug, Clone)]
pub struct Stmt {
	pub kind: StmtKind,
	pub span: Span,
	pub id: hir::NodeId,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
	Expr(Expr),
	Let { name: ast::Ident, value: Expr },
	Assign { target: ast::Path, value: Expr },
	Loop { block: Block },
}

#[derive(Debug, Clone)]
pub struct Expr {
	pub kind: ExprKind,
	pub ty: ty::TyKind,
	pub span: Span,
	pub id: hir::NodeId,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
	Literal(LiteralKind, Symbol),
	Access(ast::Path),

	Unary(Spanned<UnaryOp>, Box<Expr>),
	Binary(Spanned<BinaryOp>, Box<Expr>, Box<Expr>),

	FnCall {
		expr: Box<Expr>,
		args: Spanned<Vec<Expr>>,
	},

	If {
		cond: Box<Expr>,
		conseq: Box<Block>,
		altern: Option<Box<Block>>,
	},

	Return(Option<Box<Expr>>),
	Break(Option<Box<Expr>>),
	// TODO: add scope label
	Continue,
}
