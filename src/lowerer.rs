//! AST to HIR lowering logic

use std::sync::atomic::{AtomicU32, Ordering};

use cranelift::codegen::FxHashMap;

use crate::{
	ast::{self, Spanned},
	errors, hir,
	session::{DcxHandle, Diagnostic, DiagnosticCtx, SessionCtx, Span},
};

pub fn lower_root(scx: &SessionCtx, source: &ast::Root) -> hir::Root {
	let mut l = Lowerer::new(scx);
	let hir = source.lower(&mut l);
	scx.aid_hid_map.put(l.aid_hid_map);
	hir
}

pub trait Lower {
	type Out;

	fn lower(&self, l: &mut Lowerer) -> Self::Out;

	fn lower_box(&self, l: &mut Lowerer) -> Box<Self::Out> {
		Box::new(self.lower(l))
	}
}

#[derive(Debug)]
pub struct Lowerer<'scx> {
	scx: &'scx SessionCtx,

	aid_hid_map: FxHashMap<ast::NodeId, hir::NodeId>,
}

impl<'scx> Lowerer<'scx> {
	pub fn new(scx: &'scx SessionCtx) -> Self {
		Self {
			scx,
			aid_hid_map: FxHashMap::default(),
		}
	}

	/// Mint a new [`hir::NodeId`] giving the corresponding [`ast::NodeId`] is possible
	fn make_node_id(&mut self, aid: impl Into<Option<ast::NodeId>>) -> hir::NodeId {
		static NEXT_NODE_ID: AtomicU32 = AtomicU32::new(0);
		let hid = hir::NodeId::new(NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed));

		if let Some(aid) = aid.into() {
			self.aid_hid_map.insert(aid, hid);
		}

		hid
	}

	fn lower_opt<O, L: Lower<Out = O>>(&mut self, opt: Option<&L>) -> Option<O> {
		opt.map(|item| item.lower(self))
	}

	fn lower_opt_box<O, L: Lower<Out = O>>(&mut self, opt: Option<&L>) -> Option<Box<O>> {
		self.lower_opt(opt).map(Box::new)
	}
}

trait IterLower<Out, L: Lower<Out = Out>>: Iterator<Item = L> + Sized {
	fn lower_iter(self, l: &mut Lowerer) -> impl Iterator<Item = Out> {
		self.map(|item| item.lower(l))
	}
}

impl<Out, L: Lower<Out = Out>, I: Iterator<Item = L> + Sized> IterLower<Out, L> for I {}

trait IterDiagnostics<T>: Iterator<Item = Result<T, Diagnostic>> + Sized {
	fn collect_diagnostics(self, dcx: &DiagnosticCtx) -> impl Iterator<Item = T> {
		self.into_iter().filter_map(|item| match item {
			Ok(item) => Some(item),
			Err(diag) => {
				dcx.emit(&diag);
				None
			}
		})
	}
}

impl<T, I: Iterator<Item = Result<T, Diagnostic>> + Sized> IterDiagnostics<T> for I {}

impl<Out: 'static, T: Lower<Out = Out>> Lower for &T {
	type Out = Out;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		(*self).lower(l)
	}
}

impl Lower for ast::Root {
	type Out = hir::Root;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { attrs, items } = &self;
		Self::Out {
			attrs: attrs.iter().lower_iter(l).collect(),
			items: items.iter().lower_iter(l).collect(),
		}
	}
}

impl Lower for ast::Attr {
	type Out = hir::Attr;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self {
			path,
			meta,
			span,
			id,
		} = &self;
		Self::Out {
			path: lower_attr_path(l, path),
			meta: meta.lower(l),
			span: *span,
			id: l.make_node_id(*id),
		}
	}
}
impl Lower for ast::AttrMeta {
	type Out = hir::AttrMeta;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		match self {
			Self::None => hir::AttrMeta::None,
			Self::Tuple(exprs) => hir::AttrMeta::Tuple(exprs.iter().lower_iter(l).collect()),
			Self::Map(exprs) => hir::AttrMeta::Map(exprs.iter().lower_iter(l).collect()),
			Self::List(exprs) => hir::AttrMeta::List(exprs.iter().lower_iter(l).collect()),
		}
	}
}

impl Lower for ast::Item {
	type Out = hir::Item;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self {
			kind,
			attrs,
			span,
			id,
		} = &self;

		Self::Out {
			kind: kind.lower(l),
			span: *span,
			id: l.make_node_id(*id),
		}
	}
}

impl Lower for ast::ItemKind {
	type Out = hir::ItemKind;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		match &self {
			Self::Function(func) => hir::ItemKind::Function(func.lower(l)),

			Self::TypeAlias(ast::TypeAlias { name, alias }) => {
				hir::ItemKind::TypeAlias(hir::TypeAlias {
					name: *name,
					alias: alias.clone(),
				})
			}
			Self::Struct {
				name,
				generics,
				fields,
			} => hir::ItemKind::Struct(hir::Struct {
				name: *name,
				generics: generics.clone(),
				fields: fields.iter().lower_iter(l).collect(),
			}),
			Self::Enum {
				name,
				generics,
				variants,
			} => hir::ItemKind::Enum(hir::Enum {
				name: *name,
				generics: generics.clone(),
				variants: variants.iter().lower_iter(l).collect(),
			}),

			Self::Trait {
				name,
				generics,
				members,
			} => {
				let scx = l.scx;
				hir::ItemKind::Trait {
					name: *name,
					generics: generics.clone(),
					members: members
						.iter()
						.lower_iter(l)
						.map(TryFrom::try_from)
						.collect_diagnostics(scx.dcx())
						.collect(),
				}
			}
			Self::TraitImpl {
				type_,
				trait_,
				members,
			} => {
				let scx = l.scx;
				hir::ItemKind::TraitImpl {
					type_: type_.lower(l),
					trait_: trait_.lower(l),
					members: members
						.iter()
						.lower_iter(l)
						.map(TryFrom::try_from)
						.collect_diagnostics(scx.dcx())
						.collect(),
				}
			}

			Self::ForeignMod { items } => {
				let scx = l.scx;
				hir::ItemKind::ForeignMod {
					items: items
						.iter()
						.lower_iter(l)
						.map(TryFrom::try_from)
						.collect_diagnostics(scx.dcx())
						.collect(),
				}
			}
		}
	}
}

impl TryFrom<hir::Item> for hir::Item<hir::TraitItemKind> {
	type Error = Diagnostic;
	fn try_from(value: hir::Item) -> Result<Self, Self::Error> {
		let hir::Item { kind, span, id } = value;
		let kind = match kind {
			hir::ItemKind::Function(func) => hir::TraitItemKind::Function(func),
			hir::ItemKind::TypeAlias(ty) => hir::TraitItemKind::TypeAlias(ty),
			_ => {
				let diag = Diagnostic::new(errors::lowerer::incorrect_item_in_trait(span));
				return Err(diag);
			}
		};
		Ok(Self { kind, span, id })
	}
}

impl TryFrom<hir::Item> for hir::Item<hir::ForeignItemKind> {
	type Error = Diagnostic;
	fn try_from(value: hir::Item) -> Result<Self, Self::Error> {
		let hir::Item { kind, span, id } = value;
		let kind = if let hir::ItemKind::Function(func) = kind {
			hir::ForeignItemKind::Function(func)
		} else {
			// FIXME: adapt diagnostic
			let diag = Diagnostic::new(errors::lowerer::incorrect_item_in_trait(span));
			return Err(diag);
		};
		Ok(Self { kind, span, id })
	}
}

impl Lower for ast::Function {
	type Out = hir::Function;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { name, decl, body } = &self;

		Self::Out {
			name: *name,
			decl: decl.lower_box(l),
			body: l.lower_opt_box(body.as_deref()),
		}
	}
}

impl Lower for ast::Block {
	type Out = hir::Block;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { stmts, span, id } = &self;

		let mut out_stmts = Vec::new();
		let mut ret = None;

		let mut ast_stmts = &stmts[..];
		while let [stmt, tail @ ..] = ast_stmts {
			ast_stmts = tail;

			let stmt = match stmt.lower(l) {
				Some(StmtOrRet::Stmt(stmt)) => stmt,
				Some(StmtOrRet::Ret(expr)) if tail.is_empty() => {
					ret = Some(Box::new(expr));
					continue;
				}
				Some(StmtOrRet::Ret(expr)) => {
					let report = errors::lowerer::no_semicolon_mid_block(expr.span);
					l.scx.dcx().emit_build(report);

					// recover like there was a semicolon
					hir::Stmt {
						span: expr.span,
						kind: hir::StmtKind::Expr {
							expr: Box::new(expr),
						},
						id: l.make_node_id(None),
					}
				}
				None => continue,
			};

			out_stmts.push(stmt);
		}

		Self::Out {
			stmts: out_stmts,
			ret,
			span: *span,
			id: l.make_node_id(*id),
		}
	}
}

pub enum StmtOrRet {
	Stmt(hir::Stmt),
	Ret(hir::Expr),
}

impl Lower for ast::Stmt {
	type Out = Option<StmtOrRet>;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { kind, span, id } = &self;
		let kind = match &kind {
			ast::StmtKind::Let {
				name: ident,
				ty,
				value,
				mutable,
			} => hir::StmtKind::Let {
				name: *ident,
				ty: ty.clone(),
				// TODO: handle variable with no init value
				value: value.as_ref().unwrap().lower_box(l),
				mutable: *mutable,
			},
			ast::StmtKind::Expr(expr) => hir::StmtKind::Expr {
				expr: expr.lower_box(l),
			},
			ast::StmtKind::ExprRet(expr) => {
				return Some(StmtOrRet::Ret(expr.lower(l)));
			}
			ast::StmtKind::Empty => return None,
		};

		Some(StmtOrRet::Stmt(hir::Stmt {
			kind,
			span: *span,
			id: l.make_node_id(*id),
		}))
	}
}

impl Lower for ast::TypeAlias {
	type Out = hir::TypeAlias;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { name, alias } = &self;
		let _ = l;
		hir::TypeAlias {
			name: *name,
			alias: alias.clone(),
		}
	}
}

impl Lower for ast::FnDecl {
	type Out = hir::FnDecl;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { params, ret, span } = &self;
		let _ = l;
		let output = ret.clone().unwrap_or_else(|| ast::Ty {
			kind: ast::TyKind::Unit,
			span: span.end(),
		});
		hir::FnDecl {
			params: params.clone(),
			ret: Box::new(output),
			span: *span,
		}
	}
}

impl Lower for ast::FieldDef {
	type Out = hir::FieldDef;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { name, ty, span } = &self;
		let _ = l;
		hir::FieldDef {
			name: *name,
			ty: ty.clone(),
		}
	}
}

impl Lower for ast::Variant {
	type Out = hir::EnumVariant;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { name, kind, span } = &self;
		let fields = match &kind {
			ast::VariantKind::Bare => vec![],
			ast::VariantKind::Tuple(fields) => fields
				.iter()
				.enumerate()
				.map(|(i, ty)| hir::FieldDef {
					name: ast::Ident::new(l.scx.symbols.intern(&format!("{i}")), ty.span),
					ty: ty.clone(),
				})
				.collect(),
			ast::VariantKind::Struct(fields) => fields.iter().lower_iter(l).collect(),
		};

		hir::EnumVariant {
			name: *name,
			fields,
			span: *span,
		}
	}
}

impl Lower for ast::Expr {
	type Out = hir::Expr;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { kind, span, id } = &self;
		let kind = match kind {
			ast::ExprKind::Access { path } => hir::ExprKind::Access {
				path: path.lower(l),
			},
			ast::ExprKind::LiteralStr { sym } => hir::ExprKind::LiteralStr { sym: *sym },
			ast::ExprKind::LiteralInt { sym } => hir::ExprKind::LiteralInt { sym: *sym },
			ast::ExprKind::LiteralFloat { sym } => hir::ExprKind::LiteralFloat { sym: *sym },

			ast::ExprKind::Paren { expr } => return expr.lower(l),
			ast::ExprKind::Unary { op, expr } => lower_unary(l, *op, expr),
			ast::ExprKind::Binary { op, left, right } => lower_binary(l, *op, left, right),
			ast::ExprKind::ShortCircuit { op, left, right } => {
				lower_short_circuit(l, *op, left, right)
			}

			ast::ExprKind::FnCall { expr, args } => hir::ExprKind::FnCall {
				expr: Box::new(expr.lower(l)),
				args: args.with_bit(args.bit.iter().lower_iter(l).collect()),
			},
			ast::ExprKind::If {
				cond,
				conseq,
				altern,
			} => hir::ExprKind::If {
				cond: cond.lower_box(l),
				conseq: conseq.lower_box(l),
				altern: l.lower_opt_box(altern.as_deref()),
			},
			ast::ExprKind::Match { expr, arms } => todo!(),

			ast::ExprKind::Loop { body } => hir::ExprKind::Loop {
				block: body.lower_box(l),
			},
			ast::ExprKind::WhileLoop { check, body } => lower_while_loop(l, check, body),

			ast::ExprKind::Method { expr, name, params } => hir::ExprKind::Method {
				expr: Box::new(expr.lower(l)),
				name: *name,
				params: params.iter().lower_iter(l).collect(),
			},
			ast::ExprKind::Field { expr, name } => hir::ExprKind::Field {
				expr: Box::new(expr.lower(l)),
				name: *name,
			},
			ast::ExprKind::Deref { expr } => hir::ExprKind::Deref {
				expr: expr.lower_box(l),
			},

			ast::ExprKind::Assign { target, value } => hir::ExprKind::Assign {
				target: Box::new(target.lower(l)),
				value: Box::new(value.lower(l)),
			},

			ast::ExprKind::Return { expr } => hir::ExprKind::Return {
				expr: l
					.lower_opt_box(expr.as_deref())
					.unwrap_or_else(|| Box::new(make_unit(l, *span))),
			},
			ast::ExprKind::Break { expr, label } => hir::ExprKind::Break {
				expr: l
					.lower_opt_box(expr.as_deref())
					.unwrap_or_else(|| Box::new(make_unit(l, *span))),
				label: *label,
			},
			ast::ExprKind::Continue { label } => hir::ExprKind::Continue { label: *label },
		};

		hir::Expr {
			kind,
			span: *span,
			id: l.make_node_id(*id),
		}
	}
}

/// Lower an AST `while cond { body }` to an HIR `loop { if cond { body } else { break } }`
fn lower_while_loop(l: &mut Lowerer, cond: &ast::Expr, body: &ast::Block) -> hir::ExprKind {
	let break_expr = hir::Expr {
		kind: hir::ExprKind::Break {
			expr: Box::new(make_unit(l, Span::DUMMY)),
			label: None,
		},
		span: body.span,
		id: l.make_node_id(None),
	};
	let altern_blk = hir::Block {
		stmts: Vec::new(),
		ret: Some(Box::new(break_expr)),
		span: body.span,
		id: l.make_node_id(None),
	};

	let if_expr = hir::Expr {
		kind: hir::ExprKind::If {
			cond: cond.lower_box(l),
			conseq: body.lower_box(l),
			altern: Some(Box::new(altern_blk)),
		},
		span: body.span,
		id: l.make_node_id(None),
	};
	let loop_blk = hir::Block {
		stmts: Vec::new(),
		ret: Some(Box::new(if_expr)),

		span: body.span,
		id: l.make_node_id(None),
	};

	hir::ExprKind::Loop {
		block: Box::new(loop_blk),
	}
}

impl Lower for ast::Path {
	type Out = hir::Path;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		todo!()
	}
}

fn lower_unary(l: &mut Lowerer, op: Spanned<ast::UnaryOp>, expr: &ast::Expr) -> hir::ExprKind {
	// TODO: same as lower_binary
	hir::ExprKind::Unary {
		op,
		expr: expr.lower_box(l),
	}
}

fn lower_binary(
	l: &mut Lowerer,
	op: Spanned<ast::BinaryOp>,
	left: &ast::Expr,
	right: &ast::Expr,
) -> hir::ExprKind {
	// TODO: lower to interface call
	// `a + b` becomes `Add.add(a, b)` or `<a as Add>.add(b)`
	// e.g. ExprKind::FnCall { expr: to_core_func(op), args: vec![left, right] }

	hir::ExprKind::Binary {
		op,
		left: left.lower_box(l),
		right: right.lower_box(l),
	}
}

fn lower_short_circuit(
	l: &mut Lowerer,
	op: Spanned<ast::ShortCircuitOp>,
	left: &ast::Expr,
	right: &ast::Expr,
) -> hir::ExprKind {
	let (altern, conseq) = match op.bit {
		// foo() and bar()
		// → if foo() { bar() } else { false }
		ast::ShortCircuitOp::And => {
			let kind = hir::ExprKind::Access { path: todo!() };
			let expr = hir::Expr {
				kind,
				span: right.span,
				id: l.make_node_id(None),
			};
			let right_block = hir::Block {
				stmts: Vec::new(),
				ret: Some(Box::new(expr)),
				span: right.span,
				id: l.make_node_id(None),
			};
			let left_block = hir::Block {
				stmts: Vec::new(),
				ret: Some(right.lower_box(l)),
				span: left.span,
				id: l.make_node_id(None),
			};
			(Box::new(left_block), Box::new(right_block))
		}
		// foo() or bar()
		// → if foo() { true } else { bar() }
		ast::ShortCircuitOp::Or => {
			let kind = hir::ExprKind::Access { path: todo!() };
			let expr = hir::Expr {
				kind,
				span: right.span,
				id: l.make_node_id(None),
			};
			let right_block = hir::Block {
				stmts: Vec::new(),
				ret: Some(Box::new(expr)),
				span: right.span,
				id: l.make_node_id(None),
			};
			let left_block = hir::Block {
				stmts: Vec::new(),
				ret: Some(right.lower_box(l)),
				span: left.span,
				id: l.make_node_id(None),
			};
			(Box::new(right_block), Box::new(left_block))
		}
	};

	hir::ExprKind::If {
		cond: left.lower_box(l),
		conseq,
		altern: Some(altern),
	}
}

fn lower_attr_path(
	l: &mut Lowerer<'_>,
	ast::Path { segments, span, id }: &ast::Path,
) -> hir::AttrPath {
	let segments = segments
		.iter()
		.map(
			|ast::PathSegment {
			     name,
			     generics,
			     span,
			 }| {
				if !generics.params.is_empty() {
					let report = errors::lowerer::generic_in_attr_path(generics.span);
					l.scx.dcx().emit_build(report);
				}
				*name
			},
		)
		.collect();

	hir::AttrPath {
		segments,
		span: *span,
		resolved: todo!(),
	}
}

fn make_unit(l: &mut Lowerer<'_>, span: Span) -> hir::Expr {
	hir::Expr {
		kind: hir::ExprKind::Unit,
		span,
		id: l.make_node_id(None),
	}
}
