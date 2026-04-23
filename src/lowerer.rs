//! AST to HIR lowering logic

use std::{fmt::Write, num::NonZero};

use rustc_hash::FxHashMap;

use crate::{
	ast,
	ast::Spanned,
	collect::{DefId, LangItem},
	hir,
	hir::{Path, PathSegment},
	pretty_print,
	resolve::{PartialRes, Res},
	session::{ArtefactKind, DcxHandle, Diagnostic, DiagnosticCtx, SessionCtx, Span},
	symbols::sym,
};

pub(crate) fn lower_root(scx: &SessionCtx, root: &ast::Root) -> hir::Root {
	let resolutions = scx.resolutions.borrow();
	let node_id_to_def_id = scx.node_id_to_def_id.borrow();
	let lang_items = scx.lang_items.borrow();

	let mut l = Lowerer::new(scx, &resolutions, &node_id_to_def_id, &lang_items);
	let hir = l.lower_root(root);

	scx.register_artefact(&ArtefactKind::HigherIr(()), |artefact| {
		write!(artefact, "{hir:#?}")
	});
	scx.register_artefact(&ArtefactKind::HigherIrPretty(()), |artefact| {
		pretty_print::pretty_print(&hir, artefact)
	});

	scx.node_id_to_hir_id.put(l.node_id_to_hir_id);
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

	resolutions: &'scx FxHashMap<ast::NodeId, PartialRes>,
	node_id_to_def_id: &'scx FxHashMap<ast::NodeId, DefId>,
	lang_items: &'scx FxHashMap<LangItem, DefId>,

	node_id_to_hir_id: FxHashMap<ast::NodeId, hir::NodeId>,
	next_node_id: NonZero<u32>,
}

impl<'scx> Lowerer<'scx> {
	pub(crate) fn new(
		scx: &'scx SessionCtx,
		resolutions: &'scx FxHashMap<ast::NodeId, PartialRes>,
		node_id_to_def_id: &'scx FxHashMap<ast::NodeId, DefId>,
		lang_items: &'scx FxHashMap<LangItem, DefId>,
	) -> Self {
		Self {
			scx,
			resolutions,
			node_id_to_def_id,
			lang_items,

			node_id_to_hir_id: FxHashMap::default(),
			next_node_id: NonZero::new(1).unwrap(),
		}
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

impl Lowerer<'_> {
	/// Mint a new [`hir::NodeId`]
	fn create_node_id(&mut self) -> hir::NodeId {
		let node_id = hir::NodeId::new(self.next_node_id);
		self.next_node_id = self.next_node_id.checked_add(1).unwrap();
		node_id
	}

	fn lower_node_id(&mut self, id: ast::NodeId) -> hir::NodeId {
		let next_id = self.create_node_id();
		*self.node_id_to_hir_id.entry(id).or_insert_with(|| next_id)
	}
}

impl Lowerer<'_> {
	fn lower_root(&mut self, ast::Root { attrs, items }: &ast::Root) -> hir::Root {
		hir::Root {
			attrs: attrs.iter().map(|attr| self.lower_attr(attr)).collect(),
			items: items.iter().map(|item| self.lower_item(item)).collect(),
		}
	}

	fn lower_attr(
		&mut self,
		ast::Attr {
			path,
			meta,
			kind: _,
			span,
			id,
		}: &ast::Attr,
	) -> hir::Attr {
		hir::Attr {
			path: self.lower_attr_path(path),
			meta: self.lower_attr_meta(meta),
			span: *span,
			id: self.lower_node_id(*id),
		}
	}

	fn lower_attr_path(
		&mut self,
		ast::AttrPath { segments, span, id }: &ast::AttrPath,
	) -> hir::AttrPath {
		hir::AttrPath {
			segments: segments.clone(),
			span: *span,
		}
	}

	fn lower_attr_meta(&mut self, meta: &ast::AttrMeta) -> hir::AttrMeta {
		match meta {
			ast::AttrMeta::None => hir::AttrMeta::None,
			ast::AttrMeta::Tuple(exprs) => {
				hir::AttrMeta::Tuple(exprs.iter().lower_iter(self).collect())
			}
			ast::AttrMeta::Map(exprs) => {
				hir::AttrMeta::Map(exprs.iter().lower_iter(self).collect())
			}
			ast::AttrMeta::List(exprs) => {
				hir::AttrMeta::List(exprs.iter().lower_iter(self).collect())
			}
		}
	}

	fn lower_item(
		&mut self,
		ast::Item {
			attrs,
			kind,
			span,
			id,
		}: &ast::Item,
	) -> hir::Item {
		hir::Item {
			kind: self.lower_item_kind(kind),
			span: *span,
			def_id: self.node_id_to_def_id[id],
		}
	}

	fn lower_item_kind(&mut self, kind: &ast::ItemKind) -> hir::ItemKind {
		match kind {
			ast::ItemKind::ExternImport { .. } | ast::ItemKind::Import { .. } => {
				todo!("noop?")
			}
			ast::ItemKind::Module {
				name,
				items,
				inline,
			} => {
				todo!("need flat hir")
			}

			ast::ItemKind::Function(func) => hir::ItemKind::Function(self.lower_function(func)),

			ast::ItemKind::TypeAlias(ast::TypeAlias { name, alias }) => {
				hir::ItemKind::TypeAlias(hir::TypeAlias {
					name: *name,
					alias: self.lower_opt_box(alias.as_deref()),
				})
			}
			ast::ItemKind::Struct {
				name,
				generics,
				fields,
			} => hir::ItemKind::Struct(hir::Struct {
				name: *name,
				generics: generics.lower(self),
				fields: fields.iter().lower_iter(self).collect(),
			}),
			ast::ItemKind::Enum {
				name,
				generics,
				variants,
			} => hir::ItemKind::Enum(hir::Enum {
				name: *name,
				generics: generics.lower(self),
				variants: variants.iter().lower_iter(self).collect(),
			}),

			ast::ItemKind::Trait {
				name,
				generics,
				members,
			} => {
				let scx = self.scx;
				hir::ItemKind::Trait {
					name: *name,
					generics: generics.lower(self),
					members: members
						.iter()
						.map(|member| self.lower_item(member))
						.map(TryFrom::try_from)
						.collect_diagnostics(scx.dcx())
						.collect(),
				}
			}
			ast::ItemKind::TraitImpl {
				type_,
				trait_,
				members,
			} => {
				let scx = self.scx;
				hir::ItemKind::TraitImpl {
					type_: self.lower_path(type_),
					trait_: self.lower_path(trait_),
					members: members
						.iter()
						.map(|member| self.lower_item(member))
						.map(TryFrom::try_from)
						.collect_diagnostics(scx.dcx())
						.collect(),
				}
			}

			ast::ItemKind::ForeignMod { items } => {
				let scx = self.scx;
				hir::ItemKind::ForeignMod {
					items: items
						.iter()
						.map(|member| self.lower_item(member))
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
				let diag = Diagnostic::new(errors::incorrect_item_in_trait(span));
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
			let diag = Diagnostic::new(errors::incorrect_item_in_trait(span));
			return Err(diag);
		};
		Ok(Self {
			kind,
			span,
			def_id: id,
		})
	}
}

impl Lowerer<'_> {
	fn lower_function(
		&mut self,
		ast::Function {
			name,
			generics,
			decl,
			body,
		}: &ast::Function,
	) -> hir::Function {
		hir::Function {
			name: *name,
			decl: decl.lower_box(self),
			body: body
				.as_ref()
				.map(|body| self.lower_block(body))
				.map(Box::new),
		}
	}

	fn lower_block(&mut self, ast::Block { stmts, span, id }: &ast::Block) -> hir::Block {
		let mut out_stmts = Vec::new();
		let mut ret = None;

		let mut ast_stmts = &stmts[..];
		while let [stmt, tail @ ..] = ast_stmts {
			ast_stmts = tail;

			let stmt = match stmt.lower(self) {
				Some(StmtOrRet::Stmt(stmt)) => stmt,
				Some(StmtOrRet::Ret(expr)) if tail.is_empty() => {
					ret = Some(Box::new(expr));
					continue;
				}
				Some(StmtOrRet::Ret(expr)) => {
					let report = errors::no_semicolon_mid_block(expr.span);
					self.scx.dcx().emit_build(report);

					// recover like there was a semicolon
					hir::Stmt {
						span: expr.span,
						kind: hir::StmtKind::Expr {
							expr: Box::new(expr),
						},
						id: self.create_node_id(),
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
			id: self.lower_node_id(*id),
		}
	}

	fn lower_path(&mut self, ast::Path { segments, span, id }: &ast::Path) -> hir::Path {
		let partial_res = &self.resolutions[id];

		let res = if partial_res.unresolved_segments == 0 {
			partial_res.res.map_local(|id| self.node_id_to_hir_id[&id])
		} else {
			todo!();
			Res::Error
		};

		hir::Path {
			segments: segments.iter().lower_iter(self).collect(),
			span: *span,
			res,
		}
	}

	fn lower_qualified_path(
		&mut self,
		ast::Path { segments, span, id }: &ast::Path,
	) -> hir::QualifiedPath {
		let partial_res = &self.resolutions[id];

		if partial_res.unresolved_segments == 0 {
			hir::QualifiedPath::Resolved(hir::Path {
				segments: segments.iter().lower_iter(self).collect(),
				span: *span,
				res: partial_res.res.map_local(|id| self.node_id_to_hir_id[&id]),
			})
		} else {
			hir::QualifiedPath::TypeRelative {
				def_id: todo!(),
				segment: todo!(),
			}
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
			id: l.lower_node_id(*id),
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
			id: l.lower_node_id(*id),
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
				qpath: l.lower_qualified_path(path),
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
				conseq: Box::new(l.lower_block(conseq)),
				altern: altern
					.as_ref()
					.map(|altern| l.lower_block(altern))
					.map(Box::new),
			},
			ast::ExprKind::Match { expr, arms } => todo!(),

			ast::ExprKind::Loop { body } => hir::ExprKind::Loop {
				body: Box::new(l.lower_block(body)),
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
			id: l.lower_node_id(*id),
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
		id: l.create_node_id(),
	};
	let altern_blk = hir::Block {
		stmts: Vec::new(),
		ret: Some(Box::new(break_expr)),
		span: Span::DUMMY,
		id: l.create_node_id(),
	};

	let if_expr = hir::Expr {
		kind: hir::ExprKind::If {
			cond: cond.lower_box(l),
			conseq: Box::new(l.lower_block(body)),
			altern: Some(Box::new(altern_blk)),
		},
		span: Span::DUMMY,
		id: l.create_node_id(),
	};
	let loop_blk = hir::Block {
		stmts: Vec::new(),
		ret: Some(Box::new(if_expr)),
		span: Span::DUMMY,
		id: l.create_node_id(),
	};

	hir::ExprKind::Loop {
		body: Box::new(loop_blk),
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
			Self::Path(path) => hir::TyKind::Path(l.lower_qualified_path(path)),
			Self::Pointer(ty) => hir::TyKind::Pointer(ty.lower_box(l)),
			Self::Unit => hir::TyKind::Unit,
		}
	}
}

impl Lower for ast::Generics {
	type Out = hir::Generics;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { idents, span } = &self;
		hir::Generics {
			idents: idents.iter().lower_iter(l).collect(),
			span: *span,
		}
	}
}

impl Lower for ast::Generic {
	type Out = hir::Generic;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { name, default, id } = &self;
		hir::Generic {
			name: *name,
			default: l.lower_opt(default.as_ref()),
			id: l.lower_node_id(*id),
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
	// `a + b` becomes `Add.add(a, b)` or `<_ as Add>::add(a, b)`
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
				id: l.create_node_id(),
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
			let qpath = hir::QualifiedPath::Resolved(Path {
				segments: vec![path_segment],
				span: Span::DUMMY,
				res: Res::Def(todo!()),
			});
			let kind = hir::ExprKind::Access { qpath };
			let expr = hir::Expr {
				kind,
				span: right.span,
				id: l.create_node_id(),
			};
			let right_block = hir::Block {
				stmts: Vec::new(),
				ret: Some(Box::new(expr)),
				span: right.span,
				id: l.create_node_id(),
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
			let qpath = hir::QualifiedPath::Resolved(Path {
				segments: vec![path_segment],
				span: Span::DUMMY,
				res: Res::Def(todo!()),
			});

			let kind = hir::ExprKind::Access { qpath };
			let expr = hir::Expr {
				kind,
				span: right.span,
				id: l.create_node_id(),
			};
			let left_block = hir::Block {
				stmts: Vec::new(),
				ret: Some(Box::new(expr)),
				span: left.span,
				id: l.create_node_id(),
			};

			let right_block = hir::Block {
				stmts: Vec::new(),
				ret: Some(right.lower_box(l)),
				span: right.span,
				id: l.create_node_id(),
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

fn make_unit(l: &mut Lowerer<'_>, span: Span) -> hir::Expr {
	hir::Expr {
		kind: hir::ExprKind::Unit,
		span,
		id: l.create_node_id(),
	}
}

mod errors {
	use ariadne::{Label, ReportKind};

	use crate::session::{Report, ReportBuilder, Span};

	pub fn no_semicolon_mid_block(expr_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, expr_span)
			.with_message("expression is missing a semicolon but is not at the end")
			.with_label(Label::new(expr_span.end()).with_message("here"))
			.with_message("you may need to add a semicolon at the end of the expression")
	}

	pub fn incorrect_item_in_trait(item_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, item_span)
			.with_message("invalid item in trait definition".to_string())
			.with_label(Label::new(item_span).with_message("found an item that was unexpected"))
			.with_help("only type definitions and functions are allowed")
	}

	pub fn generic_in_attr_path(generics: Span) -> ariadne::ReportBuilder<Span, ReportKind> {
		Report::build(ReportKind::Error, generics)
			.with_message("attribute paths cannot contain generics".to_string())
			.with_label(Label::new(generics).with_message("remove these generics"))
	}
}
