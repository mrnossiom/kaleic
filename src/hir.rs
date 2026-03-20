//! **H**igher **IR**

use std::fmt;

use crate::{
	ast::{self, Ident, Spanned},
	resolve,
	session::Span,
	symbols::Symbol,
};

/// `hir::NodeId` are derived from `ast::NodeId`s during lowering.
///
/// If there is a 1:1 translation,
///  the new `hir::NodeId` takes the old inner number,
///  else the lowerer mints a new number.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u32);

impl NodeId {
	pub(crate) fn new(n: u32) -> Self {
		Self(n)
	}
}

impl fmt::Debug for NodeId {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		// hir node id -> hid
		write!(f, "hid#{}", self.0)
	}
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ExprId(NodeId);

#[derive(Debug)]
pub struct Root {
	pub attrs: Vec<Attr>,
	pub items: Vec<Item>,
}

#[derive(Debug)]
pub struct Attr {
	pub path: AttrPath,
	pub meta: AttrMeta,
	pub span: Span,
	pub id: NodeId,
}

#[derive(Debug)]
pub enum AttrMeta {
	None,
	Tuple(Vec<Expr>),
	Map(Vec<Expr>),
	List(Vec<Expr>),
}

#[derive(Debug, Clone)]
pub struct Item<Kind = ItemKind> {
	pub kind: Kind,
	pub span: Span,
	pub id: NodeId,
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
	pub variants: Vec<Variant>,
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
		type_: Path,
		trait_: Path,
		members: Vec<Item<TraitItemKind>>,
	},

	ForeignMod {
		items: Vec<Item<ForeignItemKind>>,
	},
}

#[derive(Debug, Clone)]
pub struct Ty {
	pub kind: TyKind,
	pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TyKind {
	/// See [`Path`]
	Path(Path),

	/// `* <ty>`
	Pointer(Box<Ty>),

	// TODO: replace with tuple definition
	Unit,
}

#[derive(Debug, Clone)]
pub struct PathSegment {
	pub name: Ident,
	pub generics: GenericParams,
	pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Path {
	pub segments: Vec<PathSegment>,
	pub span: Span,
	pub resolved: resolve::Resolution,
}

#[derive(Debug, Clone)]
pub struct GenericParams {
	pub params: Vec<Ty>,
	pub span: Span,
}

#[derive(Debug, Clone)]
pub struct AttrPath {
	pub segments: Vec<Ident>,
	pub span: Span,
	pub resolved: resolve::Resolution,
}

#[derive(Debug, Clone)]
pub struct TypeAlias {
	pub name: ast::Ident,
	pub alias: Option<Box<Ty>>,
}

#[derive(Debug, Clone)]
pub struct Function {
	pub name: ast::Ident,
	pub decl: Box<FnDecl>,
	pub body: Option<Box<Block>>,
}

#[derive(Debug, Clone)]
pub struct Variant {
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
	pub ty: Ty,
}

#[derive(Debug, Clone)]
pub struct FnDecl {
	pub params: Vec<Param>,
	pub ret: Box<Ty>,

	pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Param {
	pub name: Ident,
	pub ty: Ty,
	pub id: NodeId,
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
		ty: Option<Box<Ty>>,
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
		path: Path,
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

pub mod visit {
	use super::*;

	pub trait Visitor: Sized {
		fn visit_root(&mut self, root: &Root) {
			visit_root(self, root);
		}

		fn visit_attr(&mut self, attr: &Attr) {
			visit_attr(self, attr);
		}

		fn visit_item(&mut self, item: &Item) {
			visit_item(self, item);
		}

		fn visit_trait_item(&mut self, item: &Item<TraitItemKind>) {
			visit_trait_item(self, item);
		}

		fn visit_foreign_item(&mut self, item: &Item<ForeignItemKind>) {
			visit_foreign_item(self, item);
		}

		fn visit_expr(&mut self, expr: &Expr) {
			visit_expr(self, expr);
		}

		fn visit_block(&mut self, block: &Block) {
			visit_block(self, block);
		}

		fn visit_stmt(&mut self, stmt: &Stmt) {
			visit_stmt(self, stmt);
		}

		fn visit_ty(&mut self, ty: &Ty) {
			visit_ty(self, ty);
		}

		fn visit_path(&mut self, path: &Path) {
			visit_path(self, path);
		}

		fn visit_path_segment(&mut self, segment: &PathSegment) {
			visit_path_segment(self, segment);
		}

		fn visit_generic_params(&mut self, params: &GenericParams) {
			visit_generic_params(self, params);
		}

		fn visit_param(&mut self, param: &Param) {
			visit_param(self, param);
		}

		fn visit_field_def(&mut self, field: &FieldDef) {
			visit_field_def(self, field);
		}

		fn visit_variant(&mut self, variant: &Variant) {
			visit_variant(self, variant);
		}

		fn visit_generics(&mut self, generics: &ast::Generics) {
			visit_generics(self, generics);
		}

		fn visit_ident(&mut self, ident: &Ident) {
			visit_ident(self, ident);
		}

		fn visit_items(&mut self, items: &[Item]) {
			for item in items {
				self.visit_item(item);
			}
		}

		fn visit_trait_items(&mut self, items: &[Item<TraitItemKind>]) {
			for item in items {
				self.visit_trait_item(item);
			}
		}

		fn visit_foreign_items(&mut self, items: &[Item<ForeignItemKind>]) {
			for item in items {
				self.visit_foreign_item(item);
			}
		}

		fn visit_stmts(&mut self, stmts: &[Stmt]) {
			for stmt in stmts {
				self.visit_stmt(stmt);
			}
		}

		fn visit_attrs(&mut self, attrs: &[Attr]) {
			for attr in attrs {
				self.visit_attr(attr);
			}
		}

		fn visit_variants(&mut self, variants: &[Variant]) {
			for variant in variants {
				self.visit_variant(variant);
			}
		}

		fn visit_fields(&mut self, fields: &[FieldDef]) {
			for field in fields {
				self.visit_field_def(field);
			}
		}
	}

	pub fn visit_root<V: Visitor>(v: &mut V, Root { attrs, items }: &Root) {
		v.visit_attrs(attrs);
		v.visit_items(items);
	}

	pub fn visit_attr<V: Visitor>(
		v: &mut V,
		Attr {
			path,
			meta,
			span: _,
			id: _,
		}: &Attr,
	) {
		for segment in &path.segments {
			v.visit_ident(segment);
		}
		match meta {
			AttrMeta::None => {}
			AttrMeta::Tuple(exprs) | AttrMeta::Map(exprs) | AttrMeta::List(exprs) => {
				for expr in exprs {
					v.visit_expr(expr);
				}
			}
		}
	}

	pub fn visit_item<V: Visitor>(
		v: &mut V,
		Item {
			kind,
			span: _,
			id: _,
		}: &Item,
	) {
		match kind {
			ItemKind::Function(func) => visit_function(v, func),
			ItemKind::Struct(Struct {
				name,
				generics,
				fields,
			}) => {
				v.visit_ident(name);
				v.visit_generics(generics);
				v.visit_fields(fields);
			}
			ItemKind::Enum(Enum {
				name,
				generics,
				variants,
			}) => {
				v.visit_ident(name);
				v.visit_generics(generics);
				v.visit_variants(variants);
			}
			ItemKind::TypeAlias(TypeAlias { name, alias }) => {
				v.visit_ident(name);
				if let Some(ty) = alias {
					v.visit_ty(ty);
				}
			}
			ItemKind::Trait {
				name,
				generics,
				members,
			} => {
				v.visit_ident(name);
				v.visit_generics(generics);
				v.visit_trait_items(members);
			}
			ItemKind::TraitImpl {
				type_,
				trait_,
				members,
			} => {
				v.visit_path(type_);
				v.visit_path(trait_);
				v.visit_trait_items(members);
			}
			ItemKind::ForeignMod { items } => {
				v.visit_foreign_items(items);
			}
		}
	}

	pub fn visit_trait_item<V: Visitor>(
		v: &mut V,
		Item {
			kind,
			span: _,
			id: _,
		}: &Item<TraitItemKind>,
	) {
		match kind {
			TraitItemKind::TypeAlias(TypeAlias { name, alias }) => {
				v.visit_ident(name);
				if let Some(ty) = alias {
					v.visit_ty(ty);
				}
			}
			TraitItemKind::Function(func) => visit_function(v, func),
		}
	}

	pub fn visit_foreign_item<V: Visitor>(
		v: &mut V,
		Item {
			kind,
			span: _,
			id: _,
		}: &Item<ForeignItemKind>,
	) {
		match kind {
			ForeignItemKind::Function(func) => visit_function(v, func),
		}
	}

	pub fn visit_function<V: Visitor>(v: &mut V, Function { name, decl, body }: &Function) {
		v.visit_ident(name);
		let FnDecl {
			params,
			ret,
			span: _,
		} = &**decl;
		for param in params {
			v.visit_param(param);
		}
		v.visit_ty(ret);
		if let Some(body) = body {
			v.visit_block(body);
		}
	}

	pub fn visit_expr<V: Visitor>(
		v: &mut V,
		Expr {
			kind,
			span: _,
			id: _,
		}: &Expr,
	) {
		match kind {
			ExprKind::Access { path } => v.visit_path(path),
			ExprKind::LiteralStr { sym: _ }
			| ExprKind::LiteralInt { sym: _ }
			| ExprKind::LiteralFloat { sym: _ } => {}
			ExprKind::Unary { op: _, expr } => v.visit_expr(expr),
			ExprKind::Binary { op: _, left, right } => {
				v.visit_expr(left);
				v.visit_expr(right);
			}
			ExprKind::FnCall { expr, args } => {
				v.visit_expr(expr);
				for arg in &args.bit {
					v.visit_expr(arg);
				}
			}
			ExprKind::If {
				cond,
				conseq,
				altern,
			} => {
				v.visit_expr(cond);
				v.visit_block(conseq);
				if let Some(altern) = altern {
					v.visit_block(altern);
				}
			}
			ExprKind::Loop { block } => v.visit_block(block),
			ExprKind::Method { expr, name, params } => {
				v.visit_expr(expr);
				v.visit_ident(name);
				for param in params {
					v.visit_expr(param);
				}
			}
			ExprKind::Field { expr, name } => {
				v.visit_expr(expr);
				v.visit_ident(name);
			}
			ExprKind::Deref { expr } => v.visit_expr(expr),
			ExprKind::Assign { target, value } => {
				v.visit_expr(target);
				v.visit_expr(value);
			}
			ExprKind::Return { expr } => {
				v.visit_expr(expr);
			}
			ExprKind::Break { expr, label } => {
				v.visit_expr(expr);
				if let Some(label) = label {
					v.visit_ident(&label.bit);
				}
			}
			ExprKind::Continue { label } => {
				if let Some(label) = label {
					v.visit_ident(&label.bit);
				}
			}
			ExprKind::Unit => {}
		}
	}

	pub fn visit_block<V: Visitor>(
		v: &mut V,
		Block {
			stmts,
			ret,
			span: _,
			id: _,
		}: &Block,
	) {
		v.visit_stmts(stmts);
		if let Some(expr) = ret {
			v.visit_expr(expr);
		}
	}

	pub fn visit_stmt<V: Visitor>(
		v: &mut V,
		Stmt {
			kind,
			span: _,
			id: _,
		}: &Stmt,
	) {
		match kind {
			StmtKind::Let {
				name,
				ty,
				value,
				mutable: _,
			} => {
				v.visit_ident(name);
				if let Some(ty) = ty {
					v.visit_ty(ty);
				}
				v.visit_expr(value);
			}
			StmtKind::Expr { expr } => v.visit_expr(expr),
		}
	}

	pub fn visit_ty<V: Visitor>(v: &mut V, Ty { kind, span: _ }: &Ty) {
		match kind {
			TyKind::Path(path) => v.visit_path(path),
			TyKind::Pointer(ty) => v.visit_ty(ty),
			TyKind::Unit => {}
		}
	}

	pub fn visit_path<V: Visitor>(
		v: &mut V,
		Path {
			segments,
			span: _,
			resolved: _,
		}: &Path,
	) {
		for segment in segments {
			v.visit_path_segment(segment);
		}
	}

	pub fn visit_path_segment<V: Visitor>(
		v: &mut V,
		PathSegment {
			name,
			generics,
			span: _,
		}: &PathSegment,
	) {
		v.visit_ident(name);
		v.visit_generic_params(generics);
	}

	pub fn visit_generic_params<V: Visitor>(
		v: &mut V,
		GenericParams { params, span: _ }: &GenericParams,
	) {
		for param in params {
			v.visit_ty(param);
		}
	}

	pub fn visit_param<V: Visitor>(v: &mut V, Param { name, ty, id: _ }: &Param) {
		v.visit_ident(name);
		v.visit_ty(ty);
	}

	pub fn visit_field_def<V: Visitor>(v: &mut V, FieldDef { name, ty }: &FieldDef) {
		v.visit_ident(name);
		v.visit_ty(ty);
	}

	pub fn visit_variant<V: Visitor>(
		v: &mut V,
		Variant {
			name,
			fields,
			span: _,
		}: &Variant,
	) {
		v.visit_ident(name);
		v.visit_fields(fields);
	}

	pub fn visit_generics<V: Visitor>(
		v: &mut V,
		ast::Generics { idents, span: _ }: &ast::Generics,
	) {
		for generic in idents {
			let ast::Generic { name, id: _ } = generic;
			v.visit_ident(name);
		}
	}

	pub fn visit_ident<V: Visitor>(_v: &mut V, Ident { sym: _, span: _ }: &Ident) {}
}

pub use self::visit::Visitor;
