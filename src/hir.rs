//! **H**igher **IR**

use std::fmt;

use crate::{
	ast::{self, Ident, Spanned},
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ItemId(NodeId);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ExprId(NodeId);

#[derive(Debug)]
pub struct Root {
	// pub attrs: FxHashMap<AttrName, AttrKind>
	pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub struct Item<Kind = ItemKind> {
	pub kind: Kind,
	pub span: Span,
	pub id: NodeId,
}

impl<T> Item<T> {
	pub fn item_id(&self) -> ItemId {
		ItemId(self.id)
	}
}

#[derive(Debug, Clone)]
pub struct Struct {
	pub name: Ident,
	pub generics: ast::Generics,
	pub fields: Vec<FieldDef>,
}

#[derive(Debug, Clone)]
pub struct Enum {
	pub name: Ident,
	pub generics: ast::Generics,
	pub variants: Vec<EnumVariant>,
}

#[derive(Debug, Clone)]
pub enum ItemKind {
	Function(Function),

	Struct(Struct),
	Enum(Enum),
	TypeAlias(TypeAlias),

	Trait {
		name: ast::Ident,
		generics: ast::Generics,
		members: Vec<Item<TraitItemKind>>,
	},
	TraitImpl {
		type_: ast::Path,
		trait_: ast::Path,
		members: Vec<Item<TraitItemKind>>,
	},

	ForeignMod {
		items: Vec<Item<ForeignItemKind>>,
	},
}

#[derive(Debug, Clone)]
pub struct TypeAlias {
	pub name: ast::Ident,
	pub alias: Option<Box<ast::Ty>>,
}

#[derive(Debug, Clone)]
pub struct Function {
	pub name: ast::Ident,
	pub decl: Box<FnDecl>,
	pub body: Option<Box<Block>>,
}

#[derive(Debug, Clone)]
pub struct EnumVariant {
	pub name: ast::Ident,
	pub fields: Vec<FieldDef>,
	pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TraitItemKind {
	TypeAlias(TypeAlias),
	Function(Function),
}

#[derive(Debug, Clone)]
pub enum ForeignItemKind {
	Function(Function),
}

#[derive(Debug, Clone)]
pub struct FieldDef {
	pub name: ast::Ident,
	pub ty: ast::Ty,
}

#[derive(Debug, Clone)]
pub struct FnDecl {
	pub params: Vec<ast::Param>,
	pub ret: Box<ast::Ty>,

	pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Block {
	pub stmts: Vec<Stmt>,
	pub ret: Option<Box<Expr>>,
	pub span: Span,
	pub id: NodeId,
}

#[derive(Debug, Clone)]
pub struct Stmt {
	pub kind: StmtKind,
	pub span: Span,
	pub id: NodeId,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
	Expr {
		expr: Box<Expr>,
	},

	Let {
		name: ast::Ident,
		ty: Option<Box<ast::Ty>>,
		value: Box<Expr>,
		mutable: bool,
	},
}

#[derive(Debug, Clone)]
pub struct Expr {
	pub kind: ExprKind,
	pub span: Span,
	pub id: NodeId,
}

impl Expr {
	pub fn expr_id(&self) -> ExprId {
		ExprId(self.id)
	}
}

#[derive(Debug, Clone)]
pub enum ExprKind {
	LiteralInt {
		sym: Symbol,
	},
	LiteralFloat {
		sym: Symbol,
	},
	LiteralStr {
		sym: Symbol,
	},
	Access {
		path: ast::Path,
	},

	Unary {
		op: Spanned<ast::UnaryOp>,
		expr: Box<Expr>,
	},
	Binary {
		op: Spanned<ast::BinaryOp>,
		left: Box<Expr>,
		right: Box<Expr>,
	},

	// TODO: parse structs, enums, tuples (and records? anon struct)
	Unit,

	If {
		cond: Box<Expr>,
		conseq: Box<Block>,
		altern: Option<Box<Block>>,
	},
	// Match { },
	Loop {
		block: Box<Block>,
	},

	FnCall {
		expr: Box<Expr>,
		args: Spanned<Vec<Expr>>,
	},
	Method {
		expr: Box<Expr>,
		name: ast::Ident,
		params: Vec<Expr>,
	},

	Field {
		expr: Box<Expr>,
		name: ast::Ident,
	},
	Deref {
		expr: Box<Expr>,
	},

	Assign {
		target: Box<Expr>,
		value: Box<Expr>,
	},

	Return {
		expr: Box<Expr>,
	},
	Break {
		expr: Box<Expr>,
		label: Option<Spanned<Ident>>,
	},
	Continue {
		label: Option<Spanned<Ident>>,
	},
}

#[derive(Debug, Clone, Default, Copy)]
pub enum Abi {
	#[default]
	Kalei,
	C,
}

pub trait Visitor {
	fn visit_root(&mut self, root: &Root);

	// fn visit_attrs(&mut self, attrs: &[()]) {
	// 	for attr in attrs {
	// 		self.visit_attr(attr);
	// 	}
	// }

	// fn visit_attr(&mut self, attrs: &());

	fn visit_items(&mut self, items: &[Item]) {
		for item in items {
			self.visit_item(item);
		}
	}

	fn visit_item<T>(&mut self, item: &Item<T>);
}
