//! AST to HIR lowering logic

use std::sync::atomic::{self, AtomicU32};

use crate::{
	ast::{self, Spanned},
	errors, hir, lexer,
	session::{SessionCtx, Span},
};

pub fn lower_root(scx: &SessionCtx, source: &ast::Root) -> hir::Root {
	let mut l = Lowerer::new(scx);
	source.lower(&mut l)
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

	next_node_id: AtomicU32,
}

impl<'scx> Lowerer<'scx> {
	pub const fn new(scx: &'scx SessionCtx) -> Self {
		Self {
			scx,
			next_node_id: AtomicU32::new(0),
		}
	}

	fn make_new_node_id(&self) -> hir::NodeId {
		let hid = self.next_node_id.fetch_add(1, atomic::Ordering::Relaxed);
		hir::NodeId(hid)
	}

	fn lower_iter<O, T: Lower<Out = O>>(
		&mut self,
		iter: impl Iterator<Item = T>,
	) -> impl Iterator<Item = O> {
		iter.map(|item| item.lower(self))
	}

	fn lower_opt<O, L: Lower<Out = O>>(&mut self, opt: Option<&L>) -> Option<O> {
		opt.map(|item| item.lower(self))
	}

	fn lower_opt_box<O, L: Lower<Out = O>>(&mut self, opt: Option<&L>) -> Option<Box<O>> {
		self.lower_opt(opt).map(Box::new)
	}
}

impl<O, T: Lower<Out = O>> Lower for &T {
	type Out = O;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		(*self).lower(l)
	}
}

impl Lower for ast::Root {
	type Out = hir::Root;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { items } = &self;
		let items = l.lower_iter(items.iter()).collect();
		Self::Out { items }
	}
}

impl Lower for ast::NodeId {
	type Out = hir::NodeId;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		// TODO: store hid provenance
		let _ = self;

		l.make_new_node_id()
	}
}

impl Lower for ast::Item {
	type Out = hir::Item;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { kind, span, id } = &self;
		let kind = match &kind {
			ast::ItemKind::Function(func) => hir::ItemKind::Function(func.lower(l)),

			ast::ItemKind::TypeAlias(ast::TypeAlias { name, alias }) => {
				hir::ItemKind::TypeAlias(hir::TypeAlias {
					name: *name,
					alias: alias.clone(),
				})
			}
			ast::ItemKind::Struct {
				name,
				generics,
				fields,
			} => hir::ItemKind::Struct(hir::Struct {
				name: *name,
				generics: generics.clone(),
				fields: l.lower_iter(fields.iter()).collect(),
			}),
			ast::ItemKind::Enum {
				name,
				generics,
				variants,
			} => hir::ItemKind::Enum(hir::Enum {
				name: *name,
				generics: generics.clone(),
				variants: l.lower_iter(variants.iter()).collect(),
			}),

			// TODO
			ast::ItemKind::Trait {
				name,
				generics,
				members,
			} => hir::ItemKind::Trait {
				name: *name,
				generics: generics.clone(),
				members: l.lower_iter(members.iter()).collect(),
			},
			ast::ItemKind::TraitImpl {
				type_,
				trait_,
				members,
			} => hir::ItemKind::TraitImpl {
				type_: type_.clone(),
				trait_: trait_.clone(),
				members: l.lower_iter(members.iter()).collect(),
			},
		};
		Self::Out {
			kind,
			span: *span,
			id: id.lower(l),
		}
	}
}

impl Lower for ast::Function {
	type Out = hir::Function;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self {
			name,
			decl,
			body,
			abi,
		} = &self;

		let abi = if let Some(abi) = abi {
			let abi = match abi.kind {
				ast::ExprKind::Literal {
					lit: lexer::LiteralKind::Str,
					sym,
				} => sym,
				_ => todo!("invalid abi expr"),
			};

			match l.scx.symbols.resolve(abi).as_str() {
				"c" => hir::Abi::C,
				_ => todo!("no such abi"),
			}
		} else {
			hir::Abi::default()
		};

		Self::Out {
			name: *name,
			decl: decl.lower_box(l),
			body: l.lower_opt_box(body.as_deref()),
			abi,
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
						kind: hir::StmtKind::Expr(Box::new(expr)),
						id: l.make_new_node_id(),
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
			id: id.lower(l),
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
			ast::StmtKind::Loop { body } => hir::StmtKind::Loop(body.lower_box(l)),
			ast::StmtKind::WhileLoop { check, body } => lower_while_loop(l, check, body),

			ast::StmtKind::Let {
				ident,
				ty,
				value,
				mutable,
			} => hir::StmtKind::Let {
				name: *ident,
				ty: Box::new(ty.as_ref().map_or_else(
					|| ast::Ty {
						kind: ast::TyKind::ImplicitInfer,
						span: ident.span.end(),
					},
					|ty| ty.as_ref().clone(),
				)),
				// TODO: handle variable with no init value
				value: value.as_ref().unwrap().lower_box(l),
				mutable: *mutable,
			},
			ast::StmtKind::Expr(expr) => hir::StmtKind::Expr(expr.lower_box(l)),
			ast::StmtKind::ExprRet(expr) => {
				return Some(StmtOrRet::Ret(expr.lower(l)));
			}
			ast::StmtKind::Empty => return None,
		};

		Some(StmtOrRet::Stmt(hir::Stmt {
			kind,
			span: *span,
			id: id.lower(l),
		}))
	}
}

impl Lower for ast::TraitItem {
	type Out = hir::TraitItem;
	fn lower(&self, l: &mut Lowerer) -> Self::Out {
		let Self { kind, span } = &self;
		let kind = match &kind {
			ast::TraitItemKind::Type(ty) => hir::TraitItemKind::Type(ty.lower(l)),
			ast::TraitItemKind::Function(func) => hir::TraitItemKind::Function(func.lower(l)),
		};
		Self::Out { kind, span: *span }
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
			inputs: params.clone(),
			output: Box::new(output),
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
					name: ast::Ident::new(l.scx.symbols.intern(&format!("{i}")), Span::DUMMY),
					ty: ty.clone(),
				})
				.collect(),
			ast::VariantKind::Struct(fields) => l.lower_iter(fields.iter()).collect(),
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
			ast::ExprKind::Access { path } => hir::ExprKind::Access { path: path.clone() },
			ast::ExprKind::Literal { lit, sym } => hir::ExprKind::Literal {
				lit: *lit,
				sym: *sym,
			},

			ast::ExprKind::Paren { expr } => expr.lower(l).kind,
			ast::ExprKind::Unary { op, expr } => lower_unary(l, *op, expr),
			ast::ExprKind::Binary { op, left, right } => lower_binary(l, *op, left, right),
			ast::ExprKind::ShortCircuit { op, left, right } => {
				lower_short_circuit(l, *op, left, right)
			}

			ast::ExprKind::FnCall { expr, args } => hir::ExprKind::FnCall {
				expr: Box::new(expr.lower(l)),
				args: args.with_bit(l.lower_iter(args.bit.iter()).collect()),
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

			ast::ExprKind::Method { expr, name, params } => hir::ExprKind::Method {
				expr: Box::new(expr.lower(l)),
				name: *name,
				params: l.lower_iter(params.iter()).collect(),
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
				expr: l.lower_opt_box(expr.as_deref()),
			},
			ast::ExprKind::Break { expr, label } => hir::ExprKind::Break {
				expr: l.lower_opt_box(expr.as_deref()),
				label: todo!(),
			},
			ast::ExprKind::Continue { label } => hir::ExprKind::Continue { label: todo!() },
		};

		hir::Expr {
			kind,
			span: *span,
			id: id.lower(l),
		}
	}
}

/// Lower an AST `while cond { body }` to an HIR `loop { if cond { body } else { break } }`
fn lower_while_loop(l: &mut Lowerer, cond: &ast::Expr, body: &ast::Block) -> hir::StmtKind {
	let break_expr = hir::Expr {
		kind: hir::ExprKind::Break {
			expr: None,
			label: None,
		},
		span: body.span,
		id: l.make_new_node_id(),
	};
	let altern_blk = hir::Block {
		stmts: Vec::new(),
		ret: Some(Box::new(break_expr)),
		span: body.span,
		id: l.make_new_node_id(),
	};

	let if_expr = hir::Expr {
		kind: hir::ExprKind::If {
			cond: cond.lower_box(l),
			conseq: body.lower_box(l),
			altern: Some(Box::new(altern_blk)),
		},
		span: body.span,
		id: l.make_new_node_id(),
	};
	let loop_blk = hir::Block {
		stmts: Vec::new(),
		ret: Some(Box::new(if_expr)),

		span: body.span,
		id: l.make_new_node_id(),
	};

	hir::StmtKind::Loop(Box::new(loop_blk))
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
				id: l.make_new_node_id(),
			};
			let right_block = hir::Block {
				stmts: Vec::new(),
				ret: Some(Box::new(expr)),
				span: right.span,
				id: l.make_new_node_id(),
			};
			let left_block = hir::Block {
				stmts: Vec::new(),
				ret: Some(right.lower_box(l)),
				span: left.span,
				id: l.make_new_node_id(),
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
				id: l.make_new_node_id(),
			};
			let right_block = hir::Block {
				stmts: Vec::new(),
				ret: Some(Box::new(expr)),
				span: right.span,
				id: l.make_new_node_id(),
			};
			let left_block = hir::Block {
				stmts: Vec::new(),
				ret: Some(right.lower_box(l)),
				span: left.span,
				id: l.make_new_node_id(),
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
