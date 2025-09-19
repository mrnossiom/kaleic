//! Higher IR

use std::fmt;

use crate::{
	ast::{self, BinaryOp, Ident, Spanned, UnaryOp},
	lexer::LiteralKind,
	session::{Span, Symbol},
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl fmt::Debug for NodeId {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		// hir node id -> hid
		write!(f, "hid#{}", self.0)
	}
}

#[derive(Debug)]
pub struct Root {
	pub items: Vec<Item>,
}

#[derive(Debug)]
pub struct Item {
	pub kind: ItemKind,
	pub span: Span,
	pub id: NodeId,
}

#[derive(Debug)]
pub struct Struct {
	pub name: Ident,
	pub generics: Vec<Ident>,
	pub fields: Vec<FieldDef>,
}

#[derive(Debug)]
pub struct Enum {
	pub name: Ident,
	pub generics: Vec<Ident>,
	pub variants: Vec<EnumVariant>,
}

#[derive(Debug)]
pub enum ItemKind {
	Type(Type),
	Function(Function),

	Struct(Struct),
	Enum(Enum),

	Trait {
		name: ast::Ident,
		generics: Vec<ast::Ident>,
		members: Vec<TraitItem>,
	},
	TraitImpl {
		type_: ast::Path,
		trait_: ast::Path,
		members: Vec<TraitItem>,
	},
}

#[derive(Debug)]
pub struct Type(pub ast::Ident, pub Option<Box<ast::Ty>>);

#[derive(Debug)]
pub struct Function {
	pub name: ast::Ident,
	pub decl: Box<FnDecl>,
	pub body: Option<Box<Block>>,
	pub abi: Option<Box<Expr>>,
}

#[derive(Debug)]
pub struct EnumVariant {
	pub name: ast::Ident,
	pub fields: Vec<FieldDef>,
	pub span: Span,
}

#[derive(Debug)]
pub struct TraitItem {
	pub kind: TraitItemKind,
	pub span: Span,
}

#[derive(Debug)]
pub enum TraitItemKind {
	Type(Type),
	Function(Function),
}

#[derive(Debug, Clone)]
pub struct FieldDef {
	pub name: ast::Ident,
	pub ty: ast::Ty,
}

#[derive(Debug, Clone)]
pub struct FnDecl {
	pub inputs: Vec<ast::Param>,
	pub output: Box<ast::Ty>,

	pub span: Span,
}

#[derive(Debug)]
pub struct Block {
	pub stmts: Vec<Stmt>,
	pub ret: Option<Box<Expr>>,
	pub span: Span,
	pub id: NodeId,
}

#[derive(Debug)]
pub struct Stmt {
	pub kind: StmtKind,
	pub span: Span,
	pub id: NodeId,
}

#[derive(Debug)]
pub enum StmtKind {
	Expr(Box<Expr>),

	Let {
		ident: ast::Ident,
		// Hinted ty
		ty: Box<ast::Ty>,
		value: Box<Expr>,
	},

	// move these to expr
	Loop(Box<Block>),
}

#[derive(Debug)]
pub struct Expr {
	pub kind: ExprKind,
	pub span: Span,
	pub id: NodeId,
}

#[derive(Debug)]
pub enum ExprKind {
	Access(ast::Path),
	Literal(LiteralKind, Symbol),

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

	Method(Box<Expr>, ast::Ident, Vec<Expr>),
	Field(Box<Expr>, ast::Ident),
	Deref(Box<Expr>),

	Assign {
		target: Box<Expr>,
		value: Box<Expr>,
	},

	Return(Option<Box<Expr>>),
	Break(Option<Box<Expr>>),
	// TODO: add scope label
	Continue,
}
