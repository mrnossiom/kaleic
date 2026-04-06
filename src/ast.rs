//! **A**bstract **S**yntax **T**ree

use std::fmt;

use crate::{session::Span, symbols::Symbol};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct NodeId(u32);

impl fmt::Debug for NodeId {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		// ast node id -> aid
		write!(f, "aid#{}", self.0)
	}
}

impl NodeId {
	pub(crate) fn new(n: u32) -> Self {
		Self(n)
	}
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Ident {
	pub(crate) sym: Symbol,
	pub(crate) span: Span,
}

impl Ident {
	#[must_use]
	pub(crate) const fn new(sym: Symbol, span: Span) -> Self {
		Self { sym, span }
	}
}

impl fmt::Debug for Ident {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "Ident({:?}, {:?})", self.sym, self.span)
	}
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Spanned<T> {
	/// The bit of information that is spanned
	pub(crate) bit: T,

	pub(crate) span: Span,
}

impl<T> Spanned<T> {
	pub(crate) const fn new(bit: T, span: Span) -> Self {
		Self { bit, span }
	}

	pub(crate) const fn with_bit<U>(&self, bit: U) -> Spanned<U> {
		Spanned::new(bit, self.span)
	}

	pub(crate) fn map<U>(&self, map: impl FnOnce(&T) -> U) -> Spanned<U> {
		Spanned::new(map(&self.bit), self.span)
	}
}

#[derive(Debug)]
pub(crate) struct Root {
	pub(crate) attrs: Vec<Attr>,
	pub(crate) items: Vec<Item>,
}

#[derive(Debug)]
pub(crate) struct Attr {
	pub(crate) path: AttrPath,
	pub(crate) meta: AttrMeta,
	pub(crate) kind: AttrKind,
	pub(crate) span: Span,
	pub(crate) id: NodeId,
}

#[derive(Debug)]
pub(crate) struct AttrPath {
	pub(crate) segments: Vec<Ident>,
	pub(crate) span: Span,
	pub(crate) id: NodeId,
}

#[derive(Debug)]
pub(crate) enum AttrMeta {
	/// `#path`
	None,
	/// `#path(foo, bar)`
	Tuple(Vec<Expr>),
	/// `#path{key=value, key2=value2}`
	Map(Vec<Expr>),
	/// `#path[blah1, blah2, blah3]`
	List(Vec<Expr>),
}

/// What should the attr attach to
#[derive(Debug, Clone, Copy)]
pub(crate) enum AttrKind {
	/// `##path`
	Parent,
	/// `#path`
	Next,
}

#[derive(Debug)]
pub(crate) struct Expr {
	pub(crate) attrs: Vec<Attr>,
	pub(crate) kind: ExprKind,
	pub(crate) span: Span,
	pub(crate) id: NodeId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnaryOp {
	/// `!`
	Not,
	// TODO: dissociate lexer token kind from ast constructs, too much tokens are
	// not reachable.
	/// `-`
	Minus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryOp {
	// Arithmetic
	/// `+`
	Plus,
	/// `-`
	Minus,
	/// `*`
	Mul,
	/// `/`
	Div,
	/// `%`
	///
	/// Also commonly known as `Rem`
	#[doc(alias = "Rem")]
	Mod,

	// Bitwise
	/// `&`
	And,
	/// `|`
	Or,
	/// `^`
	Xor,

	/// `<<`
	Shl,
	/// `>>`
	Shr,

	// Compairaison
	/// `>`
	Gt,
	/// `>=`
	Ge,
	/// `<`
	Lt,
	/// `<=`
	Le,

	/// `==`
	EqEq,
	/// `!=`
	Ne,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShortCircuitOp {
	And,
	Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Delimiter {
	Paren,
	Bracket,
	Brace,
	// Lexemes are `Lt` and `Gt`
	Angled,
}

#[derive(Debug)]
pub(crate) enum ExprKind {
	// Atomics
	Access {
		path: Path,
	},
	LiteralStr {
		sym: Symbol,
	},
	LiteralInt {
		sym: Symbol,
	},
	LiteralFloat {
		sym: Symbol,
	},

	// Composition
	/// `( <expr> )`
	Paren {
		expr: Box<Expr>,
	},
	/// `<op> <expr>`
	Unary {
		op: Spanned<UnaryOp>,
		expr: Box<Expr>,
	},
	/// `<left> <op> <right>`
	Binary {
		op: Spanned<BinaryOp>,
		left: Box<Expr>,
		right: Box<Expr>,
	},
	ShortCircuit {
		op: Spanned<ShortCircuitOp>,
		left: Box<Expr>,
		right: Box<Expr>,
	},

	/// `<expr> ( <args>* )`
	FnCall {
		expr: Box<Expr>,
		args: Spanned<Vec<Expr>>,
	},

	/// `if <cond> <conseq> [ else <altern> ]`
	If {
		cond: Box<Expr>,
		conseq: Box<Block>,
		altern: Option<Box<Block>>,
	},
	Match {
		expr: Box<Expr>,
		arms: Vec<()>,
	},
	/// `loop <body>`
	WhileLoop {
		check: Box<Expr>,
		body: Box<Block>,
	},
	/// `while <check> <body>`
	Loop {
		body: Box<Block>,
	},

	/// `<expr> . <ident> ( <expr>* )`
	Method {
		expr: Box<Expr>,
		name: Ident,
		params: Vec<Expr>,
	},
	/// `<expr> . <ident>`
	Field {
		expr: Box<Expr>,
		name: Ident,
	},
	/// `<expr> . *`
	Deref {
		expr: Box<Expr>,
	},

	/// `<target> = <value>`
	Assign {
		target: Box<Expr>,
		value: Box<Expr>,
	},

	/// `return [ <expr> ]`
	Return {
		expr: Option<Box<Expr>>,
	},
	/// `break [ ' <label> ] [ <expr> ]`
	Break {
		expr: Option<Box<Expr>>,
		label: Option<Spanned<Ident>>,
	},
	/// `continue [ ' <label> ]`
	Continue {
		label: Option<Spanned<Ident>>,
	},
}

#[derive(Debug)]
pub(crate) struct Block {
	pub(crate) stmts: Vec<Stmt>,
	pub(crate) span: Span,
	pub(crate) id: NodeId,
}

#[derive(Debug)]
pub(crate) struct FnDecl {
	pub(crate) params: Vec<Param>,
	pub(crate) ret: Option<Ty>,
	pub(crate) span: Span,
}

#[derive(Debug, Clone)]
pub(crate) struct Param {
	pub(crate) name: Ident,
	pub(crate) ty: Ty,
	pub(crate) id: NodeId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Ty {
	pub(crate) kind: TyKind,
	pub(crate) span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TyKind {
	/// See [`Path`]
	Path(Path),

	/// `* <ty>`
	Pointer(Box<Ty>),
	/// `& <ty>`
	// TODO: make references
	// Reference(Box<Ty>),

	// TODO: replace with tuple definition
	Unit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Path {
	pub(crate) segments: Vec<PathSegment>,
	pub(crate) span: Span,
	pub(crate) id: NodeId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PathSegment {
	pub(crate) name: Ident,
	pub(crate) generics: GenericParams,
	pub(crate) span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GenericParams {
	pub(crate) params: Vec<Ty>,
	pub(crate) span: Span,
}

#[derive(Debug)]
pub(crate) struct Item {
	pub(crate) attrs: Vec<Attr>,
	pub(crate) kind: ItemKind,
	pub(crate) span: Span,
	pub(crate) id: NodeId,
}

impl Item {
	pub(crate) fn name(&self) -> Option<Ident> {
		match self.kind {
			ItemKind::ExternImport { name }
			| ItemKind::Module { name, .. }
			| ItemKind::Function(Function { name, .. })
			| ItemKind::TypeAlias(TypeAlias { name, .. })
			| ItemKind::Struct { name, .. }
			| ItemKind::Enum { name, .. }
			| ItemKind::Trait { name, .. } => Some(name),

			ItemKind::Import { .. } | ItemKind::TraitImpl { .. } | ItemKind::ForeignMod { .. } => {
				None
			}
		}
	}
}

/// `type <name> [ = <ty> ] ;`
#[derive(Debug)]
pub(crate) struct TypeAlias {
	pub(crate) name: Ident,
	pub(crate) alias: Option<Box<Ty>>,
}

/// `fn <name> <generics> <decl> <body>|;`
#[derive(Debug)]
pub(crate) struct Function {
	pub(crate) name: Ident,
	pub(crate) generics: Generics,
	pub(crate) decl: FnDecl,
	pub(crate) body: Option<Box<Block>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Generics {
	pub(crate) idents: Vec<Generic>,
	pub(crate) span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Generic {
	pub(crate) name: Ident,
	pub(crate) default: Option<Ty>,
	pub(crate) id: NodeId,
}

#[derive(Debug)]
pub(crate) enum ItemKind {
	/// `extern use <name>`
	ExternImport {
		name: Ident,
	},
	Import {
		tree: ImportTree,
	},
	Module {
		name: Ident,
		items: Vec<Item>,
		inline: bool,
	},

	Function(Function),
	TypeAlias(TypeAlias),
	/// `struct <name> <generics> { <fields>* }`
	Struct {
		name: Ident,
		generics: Generics,
		fields: Vec<FieldDef>,
	},
	/// `enum <name> <generics> { <variant>* }`
	Enum {
		name: Ident,
		generics: Generics,
		variants: Vec<Variant>,
	},
	/// `trait <name> <generics> { <items>* }`
	Trait {
		name: Ident,
		generics: Generics,
		members: Vec<Item>,
	},

	/// `for <type> impl <trait> { <items>* }`
	TraitImpl {
		type_: Path,
		trait_: Path,
		members: Vec<Item>,
	},

	/// `unsafe extern <abi> { <items>* }`
	ForeignMod {
		items: Vec<Item>,
	},
}

/// `<name> : <ty>`
#[derive(Debug)]
pub(crate) struct FieldDef {
	pub(crate) name: Ident,
	pub(crate) ty: Ty,

	pub(crate) span: Span,
}

/// `<name> <kind>`
#[derive(Debug)]
pub(crate) struct Variant {
	pub(crate) name: Ident,
	pub(crate) kind: VariantKind,
	pub(crate) span: Span,
}

#[derive(Debug)]
pub(crate) enum VariantKind {
	/// `ε`
	Bare,
	/// `( <tys>* )`
	Tuple(Vec<Ty>),
	/// `{ <fields>* }`
	Struct(Vec<FieldDef>),
}

#[derive(Debug)]
pub(crate) struct Stmt {
	pub(crate) kind: StmtKind,
	pub(crate) span: Span,
	pub(crate) id: NodeId,
}

#[derive(Debug)]
pub(crate) enum StmtKind {
	/// `let [ mut ] <name> [ : <ty> ] = <expr> ;`
	Let {
		name: Ident,
		ty: Option<Box<Ty>>,
		value: Option<Box<Expr>>,
		mutable: bool,
	},

	/// `<expr> ;`
	Expr(Box<Expr>),

	/// Expression without a semi to return a value at the end of a block
	ExprRet(Box<Expr>),

	/// A single lonely `;`
	Empty,
}

#[derive(Debug)]
pub(crate) enum ImportTree {
	Branches(Vec<Self>),
	Module(Ident, Box<Self>),
	Item(Ident),
	Glob,
}

pub(crate) mod visit {
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

		fn visit_generics(&mut self, generics: &Generics) {
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
			kind,
			span: _,
			id: _,
		}: &Attr,
	) {
		// TODO: visit attr path
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
			attrs,
			kind,
			span: _,
			id: _,
		}: &Item,
	) {
		v.visit_attrs(attrs);
		match kind {
			ItemKind::ExternImport { name } => {
				v.visit_ident(name);
			}
			ItemKind::Import { tree } => {
				todo!()
			}
			ItemKind::Module {
				name,
				items,
				inline: _,
			} => {
				v.visit_ident(name);
				v.visit_items(items);
			}
			ItemKind::Function(Function {
				name,
				generics,
				decl,
				body,
			}) => {
				let FnDecl {
					params,
					ret,
					span: _,
				} = decl;
				v.visit_ident(name);
				v.visit_generics(generics);
				for param in params {
					v.visit_param(param);
				}
				if let Some(ret) = ret {
					v.visit_ty(ret);
				}
				if let Some(body) = body {
					v.visit_block(body);
				}
			}
			ItemKind::TypeAlias(TypeAlias { name, alias }) => {
				v.visit_ident(name);
				if let Some(ty) = alias {
					v.visit_ty(ty);
				}
			}
			ItemKind::Struct {
				name,
				generics,
				fields,
			} => {
				v.visit_ident(name);
				v.visit_generics(generics);
				v.visit_fields(fields);
			}
			ItemKind::Enum {
				name,
				generics,
				variants,
			} => {
				v.visit_ident(name);
				v.visit_generics(generics);
				v.visit_variants(variants);
			}
			ItemKind::Trait {
				name,
				generics,
				members,
			} => {
				v.visit_ident(name);
				v.visit_generics(generics);
				v.visit_items(members);
			}
			ItemKind::TraitImpl {
				type_,
				trait_,
				members,
			} => {
				v.visit_path(type_);
				v.visit_path(trait_);
				v.visit_items(members);
			}
			ItemKind::ForeignMod { items } => {
				v.visit_items(items);
			}
		}
	}

	pub fn visit_expr<V: Visitor>(
		v: &mut V,
		Expr {
			attrs,
			kind,
			span: _,
			id: _,
		}: &Expr,
	) {
		v.visit_attrs(attrs);
		match kind {
			ExprKind::Access { path } => v.visit_path(path),
			ExprKind::LiteralStr { sym: _ }
			| ExprKind::LiteralInt { sym: _ }
			| ExprKind::LiteralFloat { sym: _ } => {}
			ExprKind::Paren { expr }
			| ExprKind::Unary { op: _, expr }
			| ExprKind::Deref { expr } => v.visit_expr(expr),
			ExprKind::Binary { op: _, left, right }
			| ExprKind::ShortCircuit { op: _, left, right } => {
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
			ExprKind::Match { expr, arms: _ } => {
				v.visit_expr(expr);
			}
			ExprKind::WhileLoop { check, body } => {
				v.visit_expr(check);
				v.visit_block(body);
			}
			ExprKind::Loop { body } => v.visit_block(body),
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
			ExprKind::Assign { target, value } => {
				v.visit_expr(target);
				v.visit_expr(value);
			}
			ExprKind::Return { expr } => {
				if let Some(expr) = expr {
					v.visit_expr(expr);
				}
			}
			ExprKind::Break { expr, label } => {
				if let Some(expr) = expr {
					v.visit_expr(expr);
				}
				if let Some(label) = label {
					v.visit_ident(&label.bit);
				}
			}
			ExprKind::Continue { label } => {
				if let Some(label) = label {
					v.visit_ident(&label.bit);
				}
			}
		}
	}

	pub fn visit_block<V: Visitor>(
		v: &mut V,
		Block {
			stmts,
			span: _,
			id: _,
		}: &Block,
	) {
		v.visit_stmts(stmts);
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
				if let Some(value) = value {
					v.visit_expr(value);
				}
			}
			StmtKind::Expr(expr) | StmtKind::ExprRet(expr) => v.visit_expr(expr),
			StmtKind::Empty => {}
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
			id: _,
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

	pub fn visit_field_def<V: Visitor>(v: &mut V, FieldDef { name, ty, span: _ }: &FieldDef) {
		v.visit_ident(name);
		v.visit_ty(ty);
	}

	pub fn visit_variant<V: Visitor>(
		v: &mut V,
		Variant {
			name,
			kind,
			span: _,
		}: &Variant,
	) {
		v.visit_ident(name);
		match kind {
			VariantKind::Bare => {}
			VariantKind::Tuple(tys) => {
				for ty in tys {
					v.visit_ty(ty);
				}
			}
			VariantKind::Struct(fields) => {
				v.visit_fields(fields);
			}
		}
	}

	pub fn visit_generics<V: Visitor>(v: &mut V, Generics { idents, span: _ }: &Generics) {
		for generic in idents {
			let Generic {
				name,
				default,
				id: _,
			} = generic;
			v.visit_ident(name);
			if let Some(ty) = default {
				v.visit_ty(ty);
			}
		}
	}

	pub fn visit_ident<V: Visitor>(_v: &mut V, Ident { sym: _, span: _ }: &Ident) {}
}

pub(crate) use self::visit::Visitor;
