use std::sync::atomic::Ordering;

use crate::{
	ast::{self, UnaryOp},
	errors,
	hir::{self, ExprKind},
	lexer,
	ty::{Infer, Inferer, Param, PrimitiveKind, TyCtx, TyKind},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InferTag(u32);

impl TyCtx<'_> {
	pub fn next_infer_tag(&self) -> InferTag {
		InferTag(self.infer_tag_count.fetch_add(1, Ordering::Relaxed))
	}
}
impl Inferer<'_> {
	fn resolve_var_ty(&self, var: &ast::Path) -> TyKind<Infer> {
		let var = var.simple();
		if let Some(ty) = self
			.local_env
			.get(&var.sym)
			.and_then(|ty_kinds| ty_kinds.last())
		{
			// search in the locals defined, respecting shadowing
			ty.clone()
		} else if let Some(ty) = self.name_env.values.get(&var.sym) {
			// search values in the whole project
			self.ty_env.get(&ty.id).unwrap().clone().as_infer()
		} else {
			let report = errors::ty::variable_not_in_scope(var.span);
			self.tcx.scx.dcx().emit_build(report);
			TyKind::Error
		}
	}

	pub fn infer_fn(&mut self) {
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
			hir::StmtKind::Let {
				name: ident,
				value,
				ty,
				mutable,
			} => {
				let explicit_ty = self.tcx.lower_ty(&ty);
				let expr_ty = self.infer_expr(value);
				self.unify(&explicit_ty, &expr_ty);

				self.local_env.entry(ident.sym).or_default().push(expr_ty);
			}
			hir::StmtKind::Loop(block) => {
				let block_ty = self.infer_block(block);
				self.unify(&TyKind::Primitive(PrimitiveKind::Void), &block_ty);
			}
		}
	}

	fn infer_expr(&mut self, expr: &hir::Expr) -> TyKind<Infer> {
		let ty = match &expr.kind {
			hir::ExprKind::Access { path } => self.resolve_var_ty(path),
			hir::ExprKind::Literal { lit, sym } => match lit {
				lexer::LiteralKind::Integer => {
					TyKind::Infer(self.tcx.next_infer_tag(), Infer::Integer)
				}
				lexer::LiteralKind::Float => TyKind::Infer(self.tcx.next_infer_tag(), Infer::Float),
				lexer::LiteralKind::Str => TyKind::Primitive(PrimitiveKind::Str),
			},

			hir::ExprKind::Unary { op, expr } => {
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
			hir::ExprKind::Binary { op, left, right } => {
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

			hir::ExprKind::Method { expr, name, params } => todo!(),
			hir::ExprKind::Field { expr, name: ident } => todo!(),
			hir::ExprKind::Deref { expr } => todo!("ensure expr ty is pointer"),

			hir::ExprKind::Assign { target, value } => {
				let ExprKind::Access { path } = &target.kind else {
					todo!("invalid lvalue")
				};

				let target_ty = self.resolve_var_ty(path);
				let value_ty = self.infer_expr(value);
				self.unify(&target_ty, &value_ty)
			}

			hir::ExprKind::Return { .. }
			| hir::ExprKind::Break { .. }
			| hir::ExprKind::Continue { .. } => TyKind::Primitive(PrimitiveKind::Never),
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
			(TyKind::Ref(expected_ref), ty) => {
				let expected = self.ty_env[expected_ref].clone().as_infer();
				self.unify(&expected, ty)
			}
			(ty, TyKind::Ref(actual_ref)) => {
				let actual = self.ty_env[actual_ref].clone().as_infer();
				self.unify(ty, &actual)
			}

			(TyKind::Infer(tag, infer), ty) | (ty, TyKind::Infer(tag, infer)) => {
				self.unify_infer(*tag, *infer, ty)
			}
			// infer and never have different meaning but both coerces to anything
			// TODO: enforce that functions that return never cannot return anything else
			// this in incorrect
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
