//! AST to HIR lowering logic

use std::{
	fmt::Write,
	sync::atomic::{AtomicU32, Ordering},
};

use rustc_hash::FxHashMap;

use crate::{
	ast::{self, Spanned},
	errors,
	hir::{self, Path, PathSegment},
	pretty_print,
	resolve::{DefId, LangItem, Resolution},
	session::{DcxHandle, Diagnostic, DiagnosticCtx, PrintKind, SessionCtx, Span},
	symbols::sym,
};

pub(crate) fn lower_root(
	scx: &SessionCtx,
	source: &ast::Root,
	resolution_map: &FxHashMap<ast::NodeId, Resolution>,
	node_id_to_def_id: &FxHashMap<ast::NodeId, DefId>,
	lang_items: &FxHashMap<LangItem, DefId>,
) -> hir::Root {
	let mut l = Lowerer::new(scx, resolution_map, node_id_to_def_id, lang_items);
	let hir = source.lower(&mut l);
	scx.node_id_to_hir_id.put(l.node_id_to_hir_id);

	scx.register_artefact(&PrintKind::HigherIr, "hir.txt", |artefact| {
		write!(artefact, "{hir:#?}")
	});
	scx.register_artefact(&PrintKind::HigherIrPretty, "hir-pretty.txt", |artefact| {
		pretty_print::pretty_print(&hir, artefact)
	});

	hir
}

pub(crate) trait Lower {
	type Out;

	fn lower(&self, l: &mut Lowerer) -> Self::Out;

	fn lower_box(&self, l: &mut Lowerer) -> Box<Self::Out> {
		Box::new(self.lower(l))
	}
}

#[derive(Debug)]
pub(crate) struct Lowerer<'scx> {
	scx: &'scx SessionCtx,

	resolution_map: &'scx FxHashMap<ast::NodeId, Resolution>,
	node_id_to_def_id: &'scx FxHashMap<ast::NodeId, DefId>,
	lang_items: &'scx FxHashMap<LangItem, DefId>,

	node_id_to_hir_id: FxHashMap<ast::NodeId, hir::NodeId>,
}

impl<'scx> Lowerer<'scx> {
	pub(crate) fn new(
		scx: &'scx SessionCtx,
		resolution_map: &'scx FxHashMap<ast::NodeId, Resolution>,
		node_id_to_def_id: &'scx FxHashMap<ast::NodeId, DefId>,
		lang_items: &'scx FxHashMap<LangItem, DefId>,
	) -> Self {
		Self {
			scx,
			resolution_map,
			node_id_to_def_id,
			lang_items,

			node_id_to_hir_id: FxHashMap::default(),
		}
	}

	/// Mint a new [`hir::NodeId`] giving the corresponding [`ast::NodeId`] is possible
	fn make_node_id(&mut self, aid: impl Into<Option<ast::NodeId>>) -> hir::NodeId {
		static NEXT_NODE_ID: AtomicU32 = AtomicU32::new(0);
		let hid = hir::NodeId::new(NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed));

		if let Some(aid) = aid.into() {
			self.node_id_to_hir_id.insert(aid, hid);
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
		hir::Root {
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
		hir::Attr {
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
		hir::Item {
			kind: kind.lower(l),
			span: *span,
			def_id: *l.node_id_to_def_id.get(id).unwrap(),
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
					alias: l.lower_opt_box(alias.as_deref()),
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
		let hir::Item {
			kind,
			span,
			def_id: id,
		} = value;
		let kind = match kind {
			hir::ItemKind::Function(func) => hir::TraitItemKind::Function(func),
			hir::ItemKind::TypeAlias(ty) => hir::TraitItemKind::TypeAlias(ty),
			_ => {
				let diag = Diagnostic::new(errors::lowerer::incorrect_item_in_trait(span));
				return Err(diag);
			}
		};
		Ok(Self {
			kind,
			span,
			def_id: id,
		})
	}
}

impl TryFrom<hir::Item> for hir::Item<hir::ForeignItemKind> {
	type Error = Diagnostic;
	fn try_from(value: hir::Item) -> Result<Self, Self::Error> {
		let hir::Item {
			kind,
			span,
			def_id: id,
		} = value;
		let kind = if let hir::ItemKind::Function(func) = kind {
			hir::ForeignItemKind::Function(func)
		} else {
			// FIXME: adapt diagnostic
			let diag = Diagnostic::new(errors::lowerer::incorrect_item_in_trait(span));
			return Err(diag);
		};
		Ok(Self {
			kind,
			span,
			def_id: id,
		})
	}
}

impl Lower for ast::Function {
	type Out = hir::Function;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self {
			name,
			generics,
			decl,
			body,
		} = &self;
		hir::Function {
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

		hir::Block {
			stmts: out_stmts,
			ret,
			span: *span,
			id: l.make_node_id(*id),
		}
	}
}

pub(crate) enum StmtOrRet {
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
				ty: l.lower_opt_box(ty.as_deref()),
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
			alias: l.lower_opt_box(alias.as_deref()),
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
			params: params.iter().lower_iter(l).collect(),
			ret: output.lower_box(l),
			span: *span,
		}
	}
}

impl Lower for ast::Param {
	type Out = hir::Param;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { name, ty, id } = &self;
		hir::Param {
			name: *name,
			ty: ty.lower(l),
			id: l.make_node_id(*id),
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
			ty: ty.lower(l),
		}
	}
}

impl Lower for ast::Variant {
	type Out = hir::Variant;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { name, kind, span } = &self;
		let fields = match &kind {
			ast::VariantKind::Bare => vec![],
			ast::VariantKind::Tuple(fields) => fields
				.iter()
				.enumerate()
				.map(|(i, ty)| {
					let field_name = l.scx.symbols.intern(&i.to_string());
					hir::FieldDef {
						name: ast::Ident::new(field_name, ty.span),
						ty: ty.lower(l),
					}
				})
				.collect(),
			ast::VariantKind::Struct(fields) => fields.iter().lower_iter(l).collect(),
		};

		hir::Variant {
			name: *name,
			fields,
			span: *span,
		}
	}
}

impl Lower for ast::Expr {
	type Out = hir::Expr;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self {
			attrs,
			kind,
			span,
			id,
		} = &self;
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
		span: Span::DUMMY,
		id: l.make_node_id(None),
	};
	let altern_blk = hir::Block {
		stmts: Vec::new(),
		ret: Some(Box::new(break_expr)),
		span: Span::DUMMY,
		id: l.make_node_id(None),
	};

	let if_expr = hir::Expr {
		kind: hir::ExprKind::If {
			cond: cond.lower_box(l),
			conseq: body.lower_box(l),
			altern: Some(Box::new(altern_blk)),
		},
		span: Span::DUMMY,
		id: l.make_node_id(None),
	};
	let loop_blk = hir::Block {
		stmts: Vec::new(),
		ret: Some(Box::new(if_expr)),
		span: Span::DUMMY,
		id: l.make_node_id(None),
	};

	hir::ExprKind::Loop {
		block: Box::new(loop_blk),
	}
}

impl Lower for ast::Path {
	type Out = hir::Path;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { segments, span, id } = &self;
		let res = l.resolution_map[id];
		hir::Path {
			segments: segments.iter().lower_iter(l).collect(),
			span: *span,
			resolved: res,
		}
	}
}

impl Lower for ast::PathSegment {
	type Out = hir::PathSegment;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self {
			name,
			generics,
			span,
		} = &self;
		hir::PathSegment {
			name: *name,
			generics: generics.lower(l),
			span: *span,
		}
	}
}

impl Lower for ast::GenericParams {
	type Out = hir::GenericParams;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { params, span } = &self;
		hir::GenericParams {
			params: params.iter().lower_iter(l).collect(),
			span: *span,
		}
	}
}

impl Lower for ast::Ty {
	type Out = hir::Ty;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { kind, span } = &self;
		hir::Ty {
			kind: kind.lower(l),
			span: *span,
		}
	}
}

impl Lower for ast::TyKind {
	type Out = hir::TyKind;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		match self {
			Self::Path(path) => hir::TyKind::Path(path.lower(l)),
			Self::Pointer(ty) => hir::TyKind::Pointer(ty.lower_box(l)),
			Self::Unit => hir::TyKind::Unit,
		}
	}
}

fn lower_unary(l: &mut Lowerer, op: Spanned<ast::UnaryOp>, expr: &ast::Expr) -> hir::ExprKind {
	// let lang_item = match op.bit {
	// 	ast::UnaryOp::Not => todo!(),
	// 	ast::UnaryOp::Minus => todo!(),
	// };

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

	// let lang_item = match op.bit {
	// 	ast::BinaryOp::Plus => TraitLangItem::Add,
	// 	ast::BinaryOp::Minus => TraitLangItem::Sub,
	// 	ast::BinaryOp::Mul => TraitLangItem::Sub,
	// 	ast::BinaryOp::Div => TraitLangItem::Sub,
	// 	_ => todo!(),
	// };

	hir::ExprKind::Binary {
		op,
		left: left.lower_box(l),
		right: right.lower_box(l),
	}
}

// TODO: very verbose for no reason
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
			let left_block = hir::Block {
				stmts: Vec::new(),
				ret: Some(right.lower_box(l)),
				span: left.span,
				id: l.make_node_id(None),
			};

			let name = ast::Ident {
				sym: sym::false_,
				span: Span::DUMMY,
			};
			let path_segment = PathSegment {
				name,
				generics: hir::GenericParams {
					params: vec![],
					span: Span::DUMMY,
				},
				span: Span::DUMMY,
			};
			let path = Path {
				segments: vec![path_segment],
				span: Span::DUMMY,
				resolved: Resolution::Def(todo!()),
			};
			let kind = hir::ExprKind::Access { path };
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
			(Box::new(left_block), Box::new(right_block))
		}
		// foo() or bar()
		// → if foo() { true } else { bar() }
		ast::ShortCircuitOp::Or => {
			let name = ast::Ident {
				sym: sym::true_,
				span: Span::DUMMY,
			};
			let path_segment = PathSegment {
				name,
				generics: hir::GenericParams {
					params: vec![],
					span: Span::DUMMY,
				},
				span: Span::DUMMY,
			};
			let path = Path {
				segments: vec![path_segment],
				span: Span::DUMMY,
				resolved: Resolution::Def(todo!()),
			};

			let kind = hir::ExprKind::Access { path };
			let expr = hir::Expr {
				kind,
				span: right.span,
				id: l.make_node_id(None),
			};
			let left_block = hir::Block {
				stmts: Vec::new(),
				ret: Some(Box::new(expr)),
				span: left.span,
				id: l.make_node_id(None),
			};

			let right_block = hir::Block {
				stmts: Vec::new(),
				ret: Some(right.lower_box(l)),
				span: right.span,
				id: l.make_node_id(None),
			};
			(Box::new(left_block), Box::new(right_block))
		}
	};

	hir::ExprKind::If {
		cond: left.lower_box(l),
		conseq,
		altern: Some(altern),
	}
}

fn lower_attr_path(l: &Lowerer<'_>, ast::Path { segments, span, id }: &ast::Path) -> hir::AttrPath {
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
		resolved: todo!("add resolution kind for attributes"),
	}
}

fn make_unit(l: &mut Lowerer<'_>, span: Span) -> hir::Expr {
	hir::Expr {
		kind: hir::ExprKind::Unit,
		span,
		id: l.make_node_id(None),
	}
}
