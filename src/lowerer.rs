//! AST to HIR lowering logic

use std::sync::atomic::{self, AtomicU32};

use crate::{
	ast::{self, Spanned},
	errors,
	hir::{
		Block, Enum, EnumVariant, Expr, ExprKind, FieldDef, FnDecl, Function, Item, ItemKind,
		NodeId, Root, Stmt, StmtKind, Struct, TraitItem, TraitItemKind, Type,
	},
	session::{SessionCtx, Span},
};

#[derive(Debug)]
pub struct LowerCtx<'scx> {
	pub scx: &'scx SessionCtx,
}

impl<'scx> LowerCtx<'scx> {
	#[must_use]
	pub fn new(scx: &'scx SessionCtx) -> Self {
		Self { scx }
	}
}

impl LowerCtx<'_> {
	pub fn lower_root(&self, ast: &ast::Root) -> Root {
		tracing::trace!("lower_root");

		let lowerer = Lowerer::new(self);
		lowerer.lower_items(ast)
	}
}

// pub trait Lower {
// 	type Out;
// 	fn lower(&self, cx: ()) -> Self::Out;
// }

// impl Lower for ast::Root {
// 	type Out = hir::Root;
// 	fn lower(&self, cx: ()) -> Self::Out {}
// }

#[derive(Debug)]
pub struct Lowerer<'lcx> {
	lcx: &'lcx LowerCtx<'lcx>,

	next_node_id: AtomicU32,
}

impl<'lcx> Lowerer<'lcx> {
	pub const fn new(lcx: &'lcx LowerCtx) -> Self {
		Self {
			lcx,
			next_node_id: AtomicU32::new(0),
		}
	}

	fn make_node_id(&self, aid: ast::NodeId) -> NodeId {
		// TODO: store hid provenance
		let _ = aid;

		let hid = self.next_node_id.fetch_add(1, atomic::Ordering::Relaxed);
		NodeId(hid)
	}

	fn make_new_node_id(&self) -> NodeId {
		let hid = self.next_node_id.fetch_add(1, atomic::Ordering::Relaxed);
		NodeId(hid)
	}
}

impl Lowerer<'_> {
	#[must_use]
	pub fn lower_items(&self, file: &ast::Root) -> Root {
		let items = file
			.items
			.iter()
			.map(|item| self.lower_item(item))
			.collect();

		Root { items }
	}

	fn lower_item(&self, item: &ast::Item) -> Item {
		let kind = match &item.kind {
			ast::ItemKind::Function(ast::Function {
				name,
				decl,
				body,
				abi,
			}) => ItemKind::Function(Function {
				name: *name,
				decl: Box::new(self.lower_fn_decl(decl)),
				body: body.as_ref().map(|block| Box::new(self.lower_block(block))),
				abi: abi.as_ref().map(|expr| Box::new(self.lower_expr(expr))),
			}),

			ast::ItemKind::Type(ast::Type { name, alias }) => {
				ItemKind::Type(Type(*name, alias.clone()))
			}
			ast::ItemKind::Struct {
				name,
				generics,
				fields,
			} => ItemKind::Struct(Struct {
				name: *name,
				generics: generics.clone(),
				fields: fields
					.iter()
					.map(|field| self.lower_field_def(field))
					.collect(),
			}),
			ast::ItemKind::Enum {
				name,
				generics,
				variants,
			} => ItemKind::Enum(Enum {
				name: *name,
				generics: generics.clone(),
				variants: variants
					.iter()
					.map(|variant| self.lower_variant(variant))
					.collect(),
			}),

			// TODO
			ast::ItemKind::Trait {
				name,
				generics,
				members,
			} => ItemKind::Trait {
				name: *name,
				generics: generics.clone(),
				members: members
					.iter()
					.map(|member| self.lower_trait_member(member))
					.collect(),
			},
			ast::ItemKind::TraitImpl {
				type_,
				trait_,
				members,
			} => ItemKind::TraitImpl {
				type_: type_.clone(),
				trait_: trait_.clone(),
				members: members
					.iter()
					.map(|member| self.lower_trait_member(member))
					.collect(),
			},
		};
		Item {
			kind,
			span: item.span,
			id: self.make_node_id(item.id),
		}
	}

	fn lower_trait_member(&self, member: &ast::TraitItem) -> TraitItem {
		let kind = match &member.kind {
			ast::TraitItemKind::Type(ty) => TraitItemKind::Type(Type(ty.name, ty.alias.clone())),
			ast::TraitItemKind::Function(ast::Function {
				name,
				decl,
				body,
				abi,
			}) => TraitItemKind::Function(Function {
				name: *name,
				decl: Box::new(self.lower_fn_decl(&decl)),
				body: body.as_ref().map(|block| Box::new(self.lower_block(block))),
				abi: abi.as_ref().map(|abi| Box::new(self.lower_expr(abi))),
			}),
		};
		TraitItem {
			kind,
			span: member.span,
		}
	}

	fn lower_fn_decl(&self, decl: &ast::FnDecl) -> FnDecl {
		let output = decl.ret.clone().unwrap_or_else(|| ast::Ty {
			kind: ast::TyKind::Unit,
			span: decl.span.end(),
		});
		FnDecl {
			inputs: decl.params.clone(),
			output: Box::new(output),
			span: decl.span,
		}
	}

	fn lower_field_def(&self, field: &ast::FieldDef) -> FieldDef {
		FieldDef {
			name: field.name,
			ty: field.ty.clone(),
		}
	}

	fn lower_variant(&self, variant: &ast::Variant) -> EnumVariant {
		let fields = match &variant.kind {
			ast::VariantKind::Bare => vec![],
			ast::VariantKind::Tuple(fields) => fields
				.iter()
				.enumerate()
				.map(|(i, ty)| FieldDef {
					name: ast::Ident::new(
						self.lcx.scx.symbols.intern(&format!("{i}")),
						Span::DUMMY,
					),
					ty: ty.clone(),
				})
				.collect(),
			ast::VariantKind::Struct(fields) => fields
				.iter()
				.map(|field| self.lower_field_def(field))
				.collect(),
		};

		EnumVariant {
			name: variant.name,
			fields,
			span: variant.span,
		}
	}

	fn lower_block(&self, block: &ast::Block) -> Block {
		let mut stmts = Vec::new();
		let mut ret = None;

		let mut ast_stmts = &block.stmts[..];
		while let [stmt, tail @ ..] = ast_stmts {
			ast_stmts = tail;

			let kind = match &stmt.kind {
				ast::StmtKind::Loop { body } => StmtKind::Loop {
					block: Box::new(self.lower_block(body)),
				},
				ast::StmtKind::WhileLoop { check, body } => self.lower_while_loop(check, body),

				ast::StmtKind::Let {
					name: ident,
					ty,
					value,
				} => StmtKind::Let {
					ident: *ident,
					ty: Box::new(ty.as_ref().map_or_else(
						|| ast::Ty {
							kind: ast::TyKind::Infer,
							span: ident.span.end(),
						},
						|ty| ty.as_ref().clone(),
					)),
					value: Box::new(self.lower_expr(value)),
				},

				ast::StmtKind::Expr(expr) => {
					let expr = Box::new(self.lower_expr(expr));
					StmtKind::Expr(expr)
				}

				ast::StmtKind::ExprRet(expr) => {
					let expr = Box::new(self.lower_expr(expr));

					// correct case
					if tail.is_empty() {
						ret = Some(expr);
						break;
					}

					let report = errors::lowerer::no_semicolon_mid_block(expr.span);
					self.lcx.scx.dcx().emit_build(report);

					// recover like there was a semicolon
					StmtKind::Expr(expr)
				}

				ast::StmtKind::Empty => continue,
			};

			stmts.push(Stmt {
				kind,
				span: stmt.span,
				id: self.make_node_id(stmt.id),
			});
		}

		Block {
			stmts,
			ret,
			span: block.span,
			id: self.make_node_id(block.id),
		}
	}

	/// Lower an AST `while cond { body }` to an HIR `loop { if cond { body } else { break } }`
	fn lower_while_loop(&self, cond: &ast::Expr, body: &ast::Block) -> StmtKind {
		let break_expr = Expr {
			kind: ExprKind::Break(None),
			span: body.span,
			id: self.make_new_node_id(),
		};
		let altern_blk = Block {
			stmts: Vec::new(),
			ret: Some(Box::new(break_expr)),
			span: body.span,
			id: self.make_new_node_id(),
		};

		let if_expr = Expr {
			kind: ExprKind::If {
				cond: Box::new(self.lower_expr(cond)),
				conseq: Box::new(self.lower_block(body)),
				altern: Some(Box::new(altern_blk)),
			},
			span: body.span,
			id: self.make_new_node_id(),
		};
		let loop_blk = Block {
			stmts: Vec::new(),
			ret: Some(Box::new(if_expr)),

			span: body.span,
			id: self.make_new_node_id(),
		};

		StmtKind::Loop {
			block: Box::new(loop_blk),
		}
	}

	fn lower_expr(&self, expr: &ast::Expr) -> Expr {
		let kind = match &expr.kind {
			ast::ExprKind::Access(path) => ExprKind::Access(path.clone()),
			ast::ExprKind::Literal(lit, ident) => ExprKind::Literal(*lit, *ident),

			ast::ExprKind::Paren(expr) => self.lower_expr(expr).kind,
			ast::ExprKind::Unary { op, expr } => self.lower_unary(*op, expr),
			ast::ExprKind::Binary { op, left, right } => self.lower_binary(*op, left, right),
			ast::ExprKind::FnCall { expr, args } => ExprKind::FnCall {
				expr: Box::new(self.lower_expr(expr)),
				args: args.with_bit(args.bit.iter().map(|e| self.lower_expr(e)).collect()),
			},
			ast::ExprKind::If {
				cond,
				conseq,
				altern,
			} => ExprKind::If {
				cond: Box::new(self.lower_expr(cond)),
				conseq: Box::new(self.lower_block(conseq)),
				altern: altern.as_ref().map(|a| Box::new(self.lower_block(a))),
			},

			ast::ExprKind::Method(expr, name, args) => ExprKind::Method(
				Box::new(self.lower_expr(expr)),
				*name,
				args.iter().map(|e| self.lower_expr(e)).collect(),
			),
			ast::ExprKind::Field(expr, name) => {
				ExprKind::Field(Box::new(self.lower_expr(expr)), *name)
			}
			ast::ExprKind::Deref(expr) => ExprKind::Deref(Box::new(self.lower_expr(expr))),

			ast::ExprKind::Assign { target, value } => ExprKind::Assign {
				target: Box::new(self.lower_expr(target)),
				value: Box::new(self.lower_expr(value)),
			},

			ast::ExprKind::Return(expr) => {
				ExprKind::Return(expr.as_ref().map(|expr| Box::new(self.lower_expr(expr))))
			}
			ast::ExprKind::Break(expr) => {
				ExprKind::Break(expr.as_ref().map(|expr| Box::new(self.lower_expr(expr))))
			}
			ast::ExprKind::Continue => ExprKind::Continue,
		};

		Expr {
			kind,
			span: expr.span,
			id: self.make_node_id(expr.id),
		}
	}

	fn lower_unary(&self, op: Spanned<ast::UnaryOp>, expr: &ast::Expr) -> ExprKind {
		ExprKind::Unary(op, Box::new(self.lower_expr(expr)))
	}

	fn lower_binary(
		&self,
		op: Spanned<ast::BinaryOp>,
		left: &ast::Expr,
		right: &ast::Expr,
	) -> ExprKind {
		// TODO: lower to interface call
		// `a + b` becomes `Add.add(a, b)` or `<a as Add>.add(b)`
		// e.g. ExprKind::FnCall { expr: to_core_func(op), args: vec![left, right] }

		ExprKind::Binary(
			op,
			Box::new(self.lower_expr(left)),
			Box::new(self.lower_expr(right)),
		)
	}
}
