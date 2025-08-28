use std::sync::atomic::Ordering;

use crate::{
	ast::{self, UnaryOp},
	errors, hir, lexer,
	ty::{FnDecl, Infer, Inferer, Param, PrimitiveKind, TyCtx, TyKind},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InferTag(u32);

impl TyCtx<'_> {
	fn next_infer_tag(&self) -> InferTag {
		InferTag(self.infer_tag_count.fetch_add(1, Ordering::Relaxed))
	}

	fn lower_ty(&self, ty: &ast::Ty) -> TyKind<Infer> {
		match &ty.kind {
			ast::TyKind::Path(path) => self.lower_path_ty(path),
			ast::TyKind::Pointer(ty) => TyKind::Pointer(Box::new(self.lower_ty(ty))),
			ast::TyKind::Unit => TyKind::Primitive(PrimitiveKind::Void),
			ast::TyKind::Infer => TyKind::Infer(self.next_infer_tag(), Infer::Explicit),
		}
	}

	// TODO: not pub
	pub fn lower_fn_decl(&self, decl: &hir::FnDecl) -> FnDecl {
		// TODO: diag no infer ty in functions
		let inputs = decl
			.inputs
			.iter()
			.map(|ast::Param { name, ty }| {
				let ty = if let Ok(ty) = self.lower_ty(ty).as_no_infer() {
					ty
				} else {
					let report = errors::ty::function_cannot_infer_signature(name.span);
					self.scx.dcx().emit_build(report);
					TyKind::Error
				};
				Param { name: *name, ty }
			})
			.collect();

		let ty = if let Ok(ty) = self.lower_ty(&decl.output).as_no_infer() {
			ty
		} else {
			let report = errors::ty::function_cannot_infer_signature(decl.output.span);
			self.scx.dcx().emit_build(report);
			TyKind::Error
		};
		FnDecl { inputs, output: ty }
	}

	fn lower_path_ty(&self, path: &ast::Path) -> TyKind<Infer> {
		// TODO: remove these constraints
		assert_eq!(path.segments.len(), 1);
		assert_eq!(path.generics.len(), 0);

		let path = path.segments[0];
		match self.scx.symbols.resolve(path.sym).as_str() {
			"_" => TyKind::Infer(self.next_infer_tag(), Infer::Explicit),

			"void" => TyKind::Primitive(PrimitiveKind::Void),
			"never" => TyKind::Primitive(PrimitiveKind::Never),

			"bool" => TyKind::Primitive(PrimitiveKind::Bool),
			"uint" => TyKind::Primitive(PrimitiveKind::UnsignedInt),
			"sint" => TyKind::Primitive(PrimitiveKind::SignedInt),
			"float" => TyKind::Primitive(PrimitiveKind::Float),

			"str" => TyKind::Primitive(PrimitiveKind::Str),

			_ => {
				let report = errors::ty::type_unknown(path.span);
				self.scx.dcx().emit_build(report);
				TyKind::Error
			}
		}
	}
}
impl Inferer<'_> {
	fn resolve_var_ty(&self, var: &ast::Path) -> TyKind<Infer> {
		// TODO: resolve full path
		let var = var.segments[0];

		if let Some(ty) = self
			.local_env
			.get(&var.sym)
			.and_then(|ty_kinds| ty_kinds.last())
		{
			ty.clone()
		} else {
			let report = errors::ty::variable_not_in_scope(var.span);
			self.tcx.scx.dcx().emit_build(report);
			TyKind::Error
		}
	}
	pub fn infer_fn(&mut self) {
		// TODO: remove
		for (fn_, decl) in self.item_env {
			self.local_env.insert(*fn_, vec![decl.clone().as_infer()]);
		}

		// init context with function arguments

		self.decl.inputs.iter().for_each(|Param { name, ty }| {
			self.local_env
				.entry(name.sym)
				.or_default()
				.push(ty.clone().as_infer());
		});

		let ret_ty = self.infer_block(self.body);
		self.unify(&self.decl.output.clone().as_infer(), &ret_ty);
	}

	fn infer_block(&mut self, block: &hir::Block) -> TyKind<Infer> {
		for stmt in &block.stmts {
			self.infer_stmt(stmt);
		}

		let expected_ret_ty = block
			.ret
			.as_ref()
			.map_or(TyKind::Primitive(PrimitiveKind::Void), |expr| {
				self.infer_expr(expr)
			});

		#[expect(clippy::let_and_return)]
		expected_ret_ty
	}

	fn infer_stmt(&mut self, stmt: &hir::Stmt) {
		match &stmt.kind {
			hir::StmtKind::Expr(expr) => {
				self.infer_expr(expr);
			}
			hir::StmtKind::Let { ident, value, ty } => {
				let explicit_ty = self.tcx.lower_ty(ty);
				let expr_ty = self.infer_expr(value);
				self.unify(&explicit_ty, &expr_ty);

				self.local_env.entry(ident.sym).or_default().push(expr_ty);
			}
			hir::StmtKind::Loop { block } => {
				let block_ty = self.infer_block(block);
				self.unify(&TyKind::Primitive(PrimitiveKind::Void), &block_ty);
			}
		}
	}

	fn infer_expr(&mut self, expr: &hir::Expr) -> TyKind<Infer> {
		let ty = match &expr.kind {
			hir::ExprKind::Access(path) => self.resolve_var_ty(path),
			hir::ExprKind::Literal(lit, _ident) => match lit {
				lexer::LiteralKind::Integer => {
					TyKind::Infer(self.tcx.next_infer_tag(), Infer::Integer)
				}
				lexer::LiteralKind::Float => TyKind::Infer(self.tcx.next_infer_tag(), Infer::Float),
				lexer::LiteralKind::Str => TyKind::Primitive(PrimitiveKind::Str),
			},

			hir::ExprKind::Unary(op, expr) => {
				let expr_ty = self.infer_expr(expr);

				match op.bit {
					UnaryOp::Not => {
						self.unify(&TyKind::Primitive(PrimitiveKind::Bool), &expr_ty);
						TyKind::Primitive(PrimitiveKind::Bool)
					}
					UnaryOp::Minus => {
						self.unify(&TyKind::Primitive(PrimitiveKind::UnsignedInt), &expr_ty);
						TyKind::Primitive(PrimitiveKind::UnsignedInt)
					}
				}
			}
			hir::ExprKind::Binary(op, left, right) => {
				let left = self.infer_expr(left);
				let right = self.infer_expr(right);

				// TODO: allow with bools, and other int types
				self.unify(&TyKind::Primitive(PrimitiveKind::UnsignedInt), &left);
				self.unify(&TyKind::Primitive(PrimitiveKind::UnsignedInt), &right);

				#[allow(clippy::enum_glob_use)]
				let expected = {
					use ast::BinaryOp::*;
					match op.bit {
						Plus | Minus | Mul | Div | Mod | And | Or | Xor | Shl | Shr => {
							TyKind::Primitive(PrimitiveKind::UnsignedInt)
						}
						Gt | Ge | Lt | Le | EqEq | Ne => TyKind::Primitive(PrimitiveKind::Bool),
					}
				};

				expected
			}
			hir::ExprKind::FnCall { expr, args } => {
				let expr_ty = self.infer_expr(expr);

				let TyKind::Fn(func) = expr_ty else {
					let report =
						errors::ty::tried_to_call_non_function(expr.span, args.span, &expr_ty);
					self.tcx.scx.dcx().emit_build(report);
					return TyKind::Error;
				};

				if func.inputs.len() != args.bit.len() {
					let report = errors::ty::function_nb_args_mismatch(
						args.span,
						func.inputs.len(),
						args.bit.len(),
					);
					self.tcx.scx.dcx().emit_build(report);
				}

				for (Param { ty: expected, .. }, actual) in func.inputs.iter().zip(args.bit.iter())
				{
					let actual_ty = self.infer_expr(actual);
					self.unify(&expected.clone().as_infer(), &actual_ty);
				}

				func.output.as_infer()
			}
			hir::ExprKind::If {
				cond,
				conseq,
				altern,
			} => {
				let cond_ty = self.infer_expr(cond);
				self.unify(&TyKind::Primitive(PrimitiveKind::Bool), &cond_ty);

				let conseq_ty = self.infer_block(conseq);
				// if no `else` part, then it must return Unit
				let altern_ty = altern
					.as_ref()
					.map_or(TyKind::Primitive(PrimitiveKind::Void), |altern| {
						self.infer_block(altern)
					});

				self.unify(&conseq_ty, &altern_ty)
			}

			hir::ExprKind::Method(_expr, _name, _args) => todo!(),
			hir::ExprKind::Field(_expr, _name) => todo!(),
			hir::ExprKind::Deref(_expr) => todo!("ensure expr ty is pointer"),

			hir::ExprKind::Assign { target: _, value } => {
				let target_ty = self.resolve_var_ty(todo!());
				let value_ty = self.infer_expr(value);
				self.unify(&target_ty, &value_ty);
			}

			hir::ExprKind::Return(_) | hir::ExprKind::Break(_) | hir::ExprKind::Continue => {
				TyKind::Primitive(PrimitiveKind::Never)
			}
		};

		// TODO
		assert!(
			self.expr_type
				.insert((expr.span, expr.id), ty.clone())
				.is_none()
		);

		ty
	}
}

/// Unification
#[expect(clippy::match_same_arms)]
impl Inferer<'_> {
	fn unify(&mut self, expected: &TyKind<Infer>, actual: &TyKind<Infer>) -> TyKind<Infer> {
		tracing::trace!(?expected, ?actual, "unify");
		match (expected, actual) {
			(TyKind::Infer(tag, infer), ty) | (ty, TyKind::Infer(tag, infer)) => {
				self.unify_infer(*tag, *infer, ty)
			}
			// infer and never have different meaning but both coerces to anything
			(TyKind::Primitive(PrimitiveKind::Never), ty)
			| (ty, TyKind::Primitive(PrimitiveKind::Never)) => ty.clone(),
			// we try to recover further by inferring errors
			(TyKind::Error, ty) | (ty, TyKind::Error) => ty.clone(),

			(_, _) if expected == actual => expected.clone(),

			(_, _) => {
				let report = errors::ty::unification_mismatch(expected, actual);
				self.tcx.scx.dcx().emit_build(report);
				TyKind::Error
			}
		}
	}

	fn unify_infer(&mut self, tag: InferTag, infer: Infer, other: &TyKind<Infer>) -> TyKind<Infer> {
		tracing::trace!(?tag, ?infer, ?other, "unify_infer");
		let unified = match (infer, other) {
			(
				Infer::Integer,
				ty @ TyKind::Primitive(PrimitiveKind::UnsignedInt | PrimitiveKind::SignedInt),
			) => ty.clone(),
			(Infer::Float, ty @ TyKind::Primitive(PrimitiveKind::Float)) => ty.clone(),
			(Infer::Generic | Infer::Explicit, ty) => ty.clone(),

			(_, TyKind::Infer(tag, actual_infer)) => {
				if infer == *actual_infer {
					TyKind::Infer(*tag, *actual_infer)
				} else {
					let report = errors::ty::infer_unification_mismatch(infer, *actual_infer);
					self.tcx.scx.dcx().emit_build(report);
					TyKind::Error
				}
			}
			(_, ty) => {
				let report = errors::ty::infer_ty_unification_mismatch(infer, ty);
				self.tcx.scx.dcx().emit_build(report);
				TyKind::Error
			}
		};

		self.infer_map.insert(tag, unified.clone());

		unified
	}
}
