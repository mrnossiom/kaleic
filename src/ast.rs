//! **A**bstract **S**yntax **T**ree

use std::fmt;

use crate::{session::Span, symbols::Symbol};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u32);

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
pub struct Ident {
	pub sym: Symbol,
	pub span: Span,
}

impl Ident {
	#[must_use]
	pub const fn new(sym: Symbol, span: Span) -> Self {
		Self { sym, span }
	}
}

impl fmt::Debug for Ident {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "Ident({:?}, {:?})", self.sym, self.span)
	}
}

#[derive(Debug, Clone, Copy)]
pub struct Spanned<T> {
	/// The bit of information that is spanned
	pub bit: T,

	pub span: Span,
}

impl<T> Spanned<T> {
	pub const fn new(bit: T, span: Span) -> Self {
		Self { bit, span }
	}

	pub const fn with_bit<U>(&self, bit: U) -> Spanned<U> {
		Spanned::new(bit, self.span)
	}

	pub fn map<U>(&self, map: impl FnOnce(&T) -> U) -> Spanned<U> {
		Spanned::new(map(&self.bit), self.span)
	}
}

#[derive(Debug)]
pub struct Root {
	pub attrs: Vec<Attr>,
	pub items: Vec<Item>,
}

#[derive(Debug)]
pub struct Attr {
	pub path: Path,
	pub meta: AttrMeta,
	pub span: Span,
	pub id: NodeId,
}

#[derive(Debug)]
pub enum AttrMeta {
	None,
	/// `#path(foo, bar)`
	Tuple(Vec<Expr>),
	/// `#path{key=value, key2=value2}`
	Map(Vec<Expr>),
	/// `#path[blah1, blah2, blah3]`
	List(Vec<Expr>),
}

#[derive(Debug)]
pub struct Expr {
	// TODO
	// pub attrs: Vec<Attr>,
	pub kind: ExprKind,
	pub span: Span,
	pub id: NodeId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
	/// `!`
	Not,
	// TODO: dissociate lexer token kind from ast constructs, too much tokens are
	// not reachable.
	/// `-`
	Minus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
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
pub enum ShortCircuitOp {
	And,
	Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delimiter {
	Paren,
	Bracket,
	Brace,
	// Lexemes are `Lt` and `Gt`
	Angled,
}

#[derive(Debug)]
pub enum ExprKind {
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
pub struct Block {
	pub stmts: Vec<Stmt>,
	pub span: Span,
	pub id: NodeId,
}

#[derive(Debug)]
pub struct FnDecl {
	pub params: Vec<Param>,
	pub ret: Option<Ty>,

	pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Param {
	pub name: Ident,
	pub ty: Ty,
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
	/// `& <ty>`
	Reference(Box<Ty>),

	// TODO: replace with tuple definition
	Unit,
}

#[derive(Debug, Clone)]
pub struct Path {
	pub segments: Vec<PathSegment>,
	pub span: Span,
	pub id: NodeId,
}

#[derive(Debug, Clone)]
pub struct PathSegment {
	pub name: Ident,
	pub generics: Vec<Ty>,
	pub span: Span,
}

impl Path {
	// TODO: remove simple api
	pub fn simple(&self) -> Ident {
		assert_eq!(self.segments.len(), 1);
		self.segments[0].name
	}

	pub(crate) fn is_match(&self, path: &[Symbol]) -> bool {
		self.segments.len() == path.len()
			&& self
				.segments
				.iter()
				.zip(path)
				.all(|(segment, sym)| segment.name.sym == *sym && segment.generics.is_empty())
	}
}

#[derive(Debug)]
pub struct Item {
	pub attrs: Vec<Attr>,
	pub kind: ItemKind,
	pub span: Span,
	pub id: NodeId,
}

/// `type <name> [ = <ty> ] ;`
#[derive(Debug)]
pub struct TypeAlias {
	pub name: Ident,
	pub alias: Option<Box<Ty>>,
}

/// `fn <name> <decl> <body>|;`
#[derive(Debug)]
pub struct Function {
	pub name: Ident,
	pub decl: FnDecl,
	pub body: Option<Box<Block>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generics(pub Vec<Ident>);

#[derive(Debug)]
pub enum ItemKind {
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
pub struct FieldDef {
	pub name: Ident,
	pub ty: Ty,

	pub span: Span,
}

/// `<name> <kind>`
#[derive(Debug)]
pub struct Variant {
	pub name: Ident,
	pub kind: VariantKind,
	pub span: Span,
}

#[derive(Debug)]
pub enum VariantKind {
	/// `ε`
	Bare,
	/// `( <tys>* )`
	Tuple(Vec<Ty>),
	/// `{ <fields>* }`
	Struct(Vec<FieldDef>),
}

#[derive(Debug)]
pub struct Stmt {
	pub kind: StmtKind,
	pub span: Span,
	pub id: NodeId,
}

#[derive(Debug)]
pub enum StmtKind {
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

pub trait Visitor {
	fn visit_root(&mut self, root: &Root);

	fn visit_attrs(&mut self, attrs: &[Attr]) {
		for attr in attrs {
			self.visit_attr(attr);
		}
	}

	fn visit_attr(&mut self, attrs: &Attr);

	fn visit_items(&mut self, items: &[Item]) {
		for item in items {
			self.visit_item(item);
		}
	}

	fn visit_item(&mut self, item: &Item);
}
