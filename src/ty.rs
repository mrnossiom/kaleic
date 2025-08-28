use std::{cell::RefCell, collections::HashMap, fmt, sync::atomic::AtomicU32};

use crate::{
	ast::{self, Ident},
	bug, errors, hir,
	inference::InferTag,
	resolve::Environment,
	session::{SessionCtx, Span, Symbol},
	tbir,
};

#[derive(Debug)]
pub struct TyCtx<'scx> {
	pub scx: &'scx SessionCtx,

	pub environment: RefCell<Option<Environment>>,

	pub(crate) infer_tag_count: AtomicU32,
}

impl<'scx> TyCtx<'scx> {
	#[must_use]
	pub fn new(scx: &'scx SessionCtx) -> Self {
		Self {
			scx,
			environment: RefCell::default(),
			infer_tag_count: AtomicU32::default(),
		}
	}
}

/// Context actions
impl TyCtx<'_> {
	#[must_use]
	#[tracing::instrument(level = "trace", skip(self, decl, body,))]
	pub fn typeck_fn(&self, name: Ident, decl: &FnDecl, body: &hir::Block) -> tbir::Block {
		let env = self.environment.borrow_mut().take().unwrap();
		// defer put back

		let mut inferer = Inferer::new(self, decl, body, &env.values);
		inferer.infer_fn();

		let mut expr_tys = HashMap::default();

		for ((span, node_id), ty_infer) in inferer.expr_type {
			match ty_infer.as_no_infer() {
				Ok(ty) => {
					expr_tys.insert(node_id, ty);
				}
				Err((mut tag, infer)) => loop {
					let Some(ty) = inferer.infer_map.get(&tag) else {
						// set default types for expression that can be inferred via literals
						match infer {
							Infer::Integer => {
								expr_tys
									.insert(node_id, TyKind::Primitive(PrimitiveKind::SignedInt));
							}
							Infer::Float => {
								expr_tys.insert(node_id, TyKind::Primitive(PrimitiveKind::Float));
							}
							Infer::Generic | Infer::Explicit => {
								let report = errors::ty::report_unconstrained(span);
								self.scx.dcx().emit_build(report);
							}
						}
						break;
					};
					match ty.clone().as_no_infer() {
						Ok(ty) => {
							expr_tys.insert(node_id, ty);
							break;
						}
						Err((next_tag, _)) => tag = next_tag,
					}
				},
			}
		}

		let tbir_builder = TbirBuilder {
			body: inferer.body,
			expr_tys,
		};
		let body = tbir_builder.build_body();

		// put back
		self.environment.borrow_mut().replace(env);

		body
	}
}

#[derive(Debug)]
pub struct Inferer<'tcx> {
	pub(crate) tcx: &'tcx TyCtx<'tcx>,
	pub(crate) item_env: &'tcx HashMap<Symbol, TyKind>,

	pub(crate) decl: &'tcx FnDecl,
	pub(crate) body: &'tcx hir::Block,

	pub(crate) local_env: HashMap<Symbol, Vec<TyKind<Infer>>>,
	// get this span out of here once we have an easy NodeId -> Span way
	pub(crate) expr_type: HashMap<(Span, hir::NodeId), TyKind<Infer>>,
	pub(crate) infer_map: HashMap<InferTag, TyKind<Infer>>,
}

impl<'tcx> Inferer<'tcx> {
	#[must_use]
	pub fn new(
		tcx: &'tcx TyCtx,
		decl: &'tcx FnDecl,
		body: &'tcx hir::Block,
		item_env: &'tcx HashMap<Symbol, TyKind>,
	) -> Self {
		Self {
			tcx,
			item_env,

			decl,
			body,

			local_env: HashMap::default(),
			expr_type: HashMap::default(),

			infer_map: HashMap::default(),
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
	pub name: ast::Ident,
	pub ty: TyKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnDecl {
	pub inputs: Vec<Param>,
	pub output: TyKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Struct {
	pub name: ast::Ident,
	// TODO?: lift generics in tykind variant?
	pub generics: Vec<Ident>,
	pub fields: Vec<FieldDef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDef {
	pub name: ast::Ident,
	pub ty: TyKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Enum {
	pub name: ast::Ident,
	// TODO?: lift generics in tykind variant?
	pub generics: Vec<Ident>,
	pub variants: Vec<Variant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variant {
	pub name: Ident,
	// pub kind: VariantKind,
	pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TyKind<InferKind = NoInfer> {
	Primitive(PrimitiveKind),
	Pointer(Box<Self>),

	Fn(Box<FnDecl>),
	// TODO: merge both in an adt construct?
	Struct(Box<Struct>),
	Enum(Box<Enum>),

	Infer(InferTag, InferKind),
	Error,
}

impl<T: fmt::Display> fmt::Display for TyKind<T> {
	// Should fit in the sentence "found {}"
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Primitive(kind) => write!(f, "primitive {kind}"),
			Self::Pointer(ty) => write!(f, "*{ty}"),
			// TODO: expand args in display
			Self::Fn(_) => write!(f, "a function"),
			Self::Struct(_struct) => write!(f, "a struct"),
			Self::Enum(_enum) => write!(f, "an enum"),
			Self::Infer(_, infer) => infer.fmt(f),
			// TODO
			Self::Error => bug!("error ty kind should never be shown to end-user"),
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimitiveKind {
	Void,
	Never,

	Bool,
	UnsignedInt,
	SignedInt,
	Float,

	Str,
}

impl fmt::Display for PrimitiveKind {
	// Should fit in the sentence "found primitive {}"
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Void => write!(f, "void"),
			Self::Never => write!(f, "never"),

			Self::Bool => write!(f, "bool"),
			Self::UnsignedInt => write!(f, "uint"),
			Self::SignedInt => write!(f, "sint"),
			Self::Float => write!(f, "float"),

			Self::Str => write!(f, "str"),
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoInfer {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Infer {
	Integer,
	Float,

	Generic,
	Explicit,
}

impl fmt::Display for Infer {
	// Should fit in the sentence "found {}"
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Integer => write!(f, "{{integer}}"),
			Self::Float => write!(f, "{{float}}"),
			Self::Generic | Self::Explicit => write!(f, "_"),
		}
	}
}

impl TyKind<NoInfer> {
	#[must_use]
	pub fn as_infer(self) -> TyKind<Infer> {
		match self {
			Self::Primitive(kind) => TyKind::Primitive(kind),
			Self::Pointer(ty) => TyKind::Pointer(Box::new(ty.as_infer())),
			Self::Fn(fn_) => TyKind::Fn(fn_),
			// Self::Adt(()) => TyKind::Adt(()),
			Self::Struct(struct_) => TyKind::Struct(struct_),
			Self::Enum(enum_) => TyKind::Enum(enum_),
			Self::Error => TyKind::Error,
		}
	}
}

impl TyKind<Infer> {
	pub fn as_no_infer(self) -> Result<TyKind<NoInfer>, (InferTag, Infer)> {
		match self {
			Self::Primitive(kind) => Ok(TyKind::Primitive(kind)),
			Self::Pointer(kind) => Ok(TyKind::Pointer(Box::new(kind.as_no_infer()?))),
			Self::Fn(fn_) => Ok(TyKind::Fn(fn_)),
			// Self::Adt(()) => Ok(TyKind::Adt(())),
			Self::Struct(struct_) => Ok(TyKind::Struct(struct_)),
			Self::Enum(enum_) => Ok(TyKind::Enum(enum_)),
			Self::Infer(tag, infer) => Err((tag, infer)),
			Self::Error => Ok(TyKind::Error),
		}
	}
}

struct TbirBuilder<'body> {
	body: &'body hir::Block,

	expr_tys: HashMap<hir::NodeId, TyKind>,
}

/// TBIR construction
impl TbirBuilder<'_> {
	fn build_body(&self) -> tbir::Block {
		self.build_block(self.body)
	}
	fn build_block(&self, block: &hir::Block) -> tbir::Block {
		let ret = block.ret.as_ref().map(|expr| self.build_expr(expr));
		let ty = ret
			.as_ref()
			.map_or(TyKind::Primitive(PrimitiveKind::Void), |expr| {
				expr.ty.clone()
			});

		tbir::Block {
			stmts: block
				.stmts
				.iter()
				.map(|stmt| self.build_stmt(stmt))
				.collect(),
			ret,
			ty,
			span: block.span,
			id: block.id,
		}
	}

	fn build_stmt(&self, stmt: &hir::Stmt) -> tbir::Stmt {
		let kind = match &stmt.kind {
			hir::StmtKind::Expr(expr) => tbir::StmtKind::Expr(self.build_expr(expr)),
			hir::StmtKind::Let {
				ident,
				ty: _,
				value,
			} => tbir::StmtKind::Let {
				name: *ident,
				value: self.build_expr(value),
			},
			hir::StmtKind::Loop { block } => tbir::StmtKind::Loop {
				block: self.build_block(block),
			},
		};
		tbir::Stmt {
			kind,
			span: stmt.span,
			id: stmt.id,
		}
	}

	fn build_expr(&self, expr: &hir::Expr) -> tbir::Expr {
		let kind = match &expr.kind {
			hir::ExprKind::Access(path) => tbir::ExprKind::Access(path.clone()),
			hir::ExprKind::Literal(kind, sym) => tbir::ExprKind::Literal(*kind, *sym),

			hir::ExprKind::Unary(op, expr) => {
				tbir::ExprKind::Unary(*op, Box::new(self.build_expr(expr)))
			}
			hir::ExprKind::Binary(op, left, right) => tbir::ExprKind::Binary(
				*op,
				Box::new(self.build_expr(left)),
				Box::new(self.build_expr(right)),
			),
			hir::ExprKind::FnCall { expr, args } => {
				let nargs = args.bit.iter().map(|arg| self.build_expr(arg)).collect();
				tbir::ExprKind::FnCall {
					expr: Box::new(self.build_expr(expr)),
					args: args.with_bit(nargs),
				}
			}
			hir::ExprKind::If {
				cond,
				conseq,
				altern,
			} => tbir::ExprKind::If {
				cond: Box::new(self.build_expr(cond)),
				conseq: Box::new(self.build_block(conseq)),
				altern: altern
					.as_ref()
					.map(|altern| Box::new(self.build_block(altern))),
			},

			hir::ExprKind::Method(_expr, _name, _args) => todo!(),
			hir::ExprKind::Field(_expr, _name) => todo!(),
			hir::ExprKind::Deref(_expr) => todo!(),

			hir::ExprKind::Assign {
				target: _,
				value: _,
			} => todo!(),

			hir::ExprKind::Return(expr) => {
				tbir::ExprKind::Return(expr.as_ref().map(|expr| Box::new(self.build_expr(expr))))
			}
			hir::ExprKind::Break(expr) => {
				tbir::ExprKind::Break(expr.as_ref().map(|expr| Box::new(self.build_expr(expr))))
			}

			hir::ExprKind::Continue => tbir::ExprKind::Continue,
		};

		let ty = match self.expr_tys.get(&expr.id) {
			Some(ty) => ty.clone(),
			None => bug!("all expression should have a type by now"),
		};

		tbir::Expr {
			kind,
			ty,
			span: expr.span,
			id: expr.id,
		}
	}
}
