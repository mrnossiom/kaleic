use std::{ops::Deref, rc::Rc};

use rustc_hash::FxHashMap;

use crate::{
	ast::{self, UnaryOp},
	errors,
	hir::{self, ExprId, ExprKind, Function, Visitor, visit},
	resolve::{DefId, NameEnvironment, Resolution},
	session::{DcxHandle, Span},
	ty::{self, Infer, InferExprTy, InferKind, LateTy, Param, PrimitiveKind, TyCtx, TyKind},
};

/// Type Variable Id
///
/// local to a function body
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TypeVarId(u32);

#[derive(Debug)]
pub(crate) struct Inferer<'tcx> {
	pub(crate) tcx: &'tcx TyCtx<'tcx>,
	pub(crate) name_env: &'tcx NameEnvironment,
	pub(crate) type_env: &'tcx FxHashMap<DefId, Rc<LateTy>>,

	pub(crate) local_env: FxHashMap<hir::NodeId, Rc<InferExprTy>>,
	pub(crate) return_ty: InferExprTy,
	pub(crate) expr_type: FxHashMap<ExprId, InferExprTy>,
	pub(crate) infer_map: FxHashMap<TypeVarId, InferExprTy>,

	next_ty_var_id: u32,
	ty_var_expr_map: FxHashMap<TypeVarId, Span>,

	// TODO: support labels
	pub(crate) loops: Vec<InferExprTy>,
}

impl<'tcx> Inferer<'tcx> {
	#[must_use]
	pub(crate) fn new(
		tcx: &'tcx TyCtx,
		decl: &'tcx ty::FnDecl<LateTy>,
		body: &'tcx hir::Block,
		name_env: &'tcx NameEnvironment,
		type_env: &'tcx FxHashMap<DefId, Rc<LateTy>>,
	) -> Self {
		Self {
			tcx,
			name_env,
			type_env,

			local_env: FxHashMap::default(),
			return_ty: TyKind::Error,
			expr_type: FxHashMap::default(),
			infer_map: FxHashMap::default(),

			next_ty_var_id: 0,
			ty_var_expr_map: FxHashMap::default(),

			loops: Vec::new(),
		}
	}

	fn make_type_variable(&mut self, span: Span) -> TypeVarId {
		let id = TypeVarId(self.next_ty_var_id);
		self.ty_var_expr_map.insert(id, span);
		self.next_ty_var_id = self.next_ty_var_id.strict_add(1);
		id
	}

	fn make_infer_ty(&mut self, span: Span, kind: InferKind) -> InferExprTy {
		TyKind::Infer(Infer {
			tvid: self.make_type_variable(span),
			kind,
		})
	}
}

pub(crate) fn infer_root(tcx: &TyCtx, hir: &hir::Root) {
	let mut visitor = InferVisitor { tcx };
	visitor.visit_root(hir);
}

/// Compute inference for every function body
struct InferVisitor<'tcx> {
	tcx: &'tcx TyCtx<'tcx>,
}

impl Visitor for InferVisitor<'_> {
	fn visit_item(&mut self, hir::Item { kind, span, def_id }: &hir::Item) {
		match kind {
			hir::ItemKind::Function(Function { name, decl, body }) => {
				let Some(body) = body.as_ref() else {
					return;
				};

				let type_env = self.tcx.type_env.borrow();
				let TyKind::Fn(decl) = type_env.get(def_id).unwrap().as_ref() else {
					todo!()
				};

				let function_expr_tys = typeck_fn(self.tcx, *name, decl, body);
				self.tcx.typeck_results.put_key(*def_id, function_expr_tys);
			}

			hir::ItemKind::ForeignMod { items } => self.visit_foreign_items(items),

			hir::ItemKind::Trait {
				name,
				generics,
				members,
			} => todo!(),
			hir::ItemKind::TraitImpl {
				type_,
				trait_,
				members,
			} => todo!(),

			hir::ItemKind::Struct(_) | hir::ItemKind::Enum(_) | hir::ItemKind::TypeAlias(_) => {
				// TODO: do nothing?
			}
		}
	}
}

#[must_use]
fn typeck_fn(
	tcx: &TyCtx,
	name: ast::Ident,
	decl: &ty::FnDecl<LateTy>,
	body: &hir::Block,
) -> FxHashMap<ExprId, LateTy> {
	let type_env = tcx.type_env.borrow();

	let mut inferer = Inferer::new(tcx, decl, body, tcx.name_env, &type_env);
	inferer.infer_fn(decl, body);

	let mut expr_tys = FxHashMap::default();

	for (node_id, ty_infer) in inferer.expr_type {
		// writeback
		match ty_infer.as_no_infer() {
			Ok(ty) => {
				expr_tys.insert(node_id, ty);
			}
			Err(Infer {
				tvid: mut tag,
				kind,
			}) => loop {
				let Some(ty) = inferer.infer_map.get(&tag) else {
					// set default types for expression that can be inferred via literals
					match kind {
						InferKind::Integer => {
							expr_tys.insert(node_id, TyKind::Primitive(PrimitiveKind::SignedInt));
						}
						InferKind::Float => {
							expr_tys.insert(node_id, TyKind::Primitive(PrimitiveKind::Float));
						}
						InferKind::Generic | InferKind::Explicit => {
							let span = todo!("get from expr_id");
							let report = errors::ty::report_unconstrained(span);
							tcx.dcx().emit_build(report);
						}
					}
					break;
				};
				match ty.clone().as_no_infer() {
					Ok(ty) => {
						expr_tys.insert(node_id, ty);
						break;
					}
					Err(Infer { tvid: next_tag, .. }) => tag = next_tag,
				}
			},
		}
	}

	expr_tys
}

impl Inferer<'_> {
	fn lower_ty(&self, ty: &hir::Ty) -> InferExprTy {
		match &ty.kind {
			hir::TyKind::Path(path) => match path.resolved {
				Resolution::Def(def_id) => self.type_env.get(&def_id).unwrap().as_infer(),
				Resolution::Local(id) => todo!("no generics rn"),
				Resolution::Error => todo!(),
			},
			hir::TyKind::Pointer(ty) => TyKind::Pointer(Rc::new(self.lower_ty(ty))),
			hir::TyKind::Unit => TyKind::Primitive(PrimitiveKind::Unit),
		}
	}

	fn resolve_var_ty(&self, var: &hir::Path) -> InferExprTy {
		match var.resolved {
			Resolution::Def(def_id) => {
				if let Some(ty) = self.type_env.get(&def_id) {
					ty.as_infer()
				} else {
					let report = errors::ty::variable_not_in_scope(var.span);
					self.tcx.dcx().emit_build(report);
					TyKind::Error
				}
			}
			Resolution::Local(node_id) => {
				let hir_id = self.tcx.scx.node_id_to_hir_id.borrow()[&node_id];
				if let Some(ty) = self.local_env.get(&hir_id) {
					// search in the locals defined, respecting shadowing
					ty.deref().clone()
				} else {
					todo!()
				}
			}
			Resolution::Error => todo!(),
		}
	}

	pub(crate) fn infer_fn(
		&mut self,
		ty::FnDecl { inputs, output }: &ty::FnDecl<LateTy>,
		body: &hir::Block,
	) {
		// TODO: not needed after resolution?
		for Param { name, ty, id } in inputs {
			self.local_env.insert(*id, Rc::new(ty.as_infer()));
		}

		let expected = output.clone().as_infer();
		self.return_ty = expected.clone();

		let ty = self.infer_block(body);
		self.unify(&expected, &ty);
	}

	#[expect(clippy::let_and_return)]
	fn infer_block(
		&mut self,
		hir::Block {
			stmts,
			ret,
			span,
			id,
		}: &hir::Block,
	) -> InferExprTy {
		for stmt in stmts {
			self.infer_stmt(stmt);
		}

		let expected_ret_ty = ret
			.as_ref()
			.map_or(TyKind::Primitive(PrimitiveKind::Unit), |expr| {
				self.infer_expr(expr)
			});

		expected_ret_ty
	}

	fn infer_stmt(&mut self, hir::Stmt { kind, span, id }: &hir::Stmt) {
		match kind {
			hir::StmtKind::Expr { expr } => {
				self.infer_expr(expr);
			}
			hir::StmtKind::Let {
				name,
				value,
				ty,
				mutable,
			} => {
				let explicit_ty = if let Some(ty) = ty {
					self.lower_ty(ty)
				} else {
					self.make_infer_ty(name.span, InferKind::Generic)
				};
				let expr_ty = self.infer_expr(value);
				self.unify(&explicit_ty, &expr_ty);

				self.local_env.insert(*id, Rc::new(expr_ty));
			}
		}
	}

	fn infer_expr(&mut self, expr @ hir::Expr { kind, span, id: _ }: &hir::Expr) -> InferExprTy {
		let ty = match kind {
			hir::ExprKind::Access { path } => self.resolve_var_ty(path),
			hir::ExprKind::LiteralStr { sym } => TyKind::Primitive(PrimitiveKind::Str),
			hir::ExprKind::LiteralInt { sym } => self.make_infer_ty(*span, InferKind::Integer),
			hir::ExprKind::LiteralFloat { sym } => self.make_infer_ty(*span, InferKind::Float),

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
					self.tcx.dcx().emit_build(report);
					return TyKind::Error;
				};

				if func.inputs.len() != args.bit.len() {
					let report = errors::ty::function_nb_args_mismatch(
						args.span,
						func.inputs.len(),
						args.bit.len(),
					);
					self.tcx.dcx().emit_build(report);
				}

				for (Param { ty: expected, .. }, actual) in func.inputs.iter().zip(args.bit.iter())
				{
					let actual_ty = self.infer_expr(actual);
					self.unify(expected, &actual_ty);
				}

				(*func.output).clone()
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
					.map_or(TyKind::Primitive(PrimitiveKind::Unit), |altern| {
						self.infer_block(altern)
					});

				self.unify(&conseq_ty, &altern_ty)
			}
			hir::ExprKind::Loop { block } => {
				let infer_ty = self.make_infer_ty(
					block.ret.as_ref().map_or(Span::DUMMY, |e| e.span),
					InferKind::Generic,
				);
				self.loops.push(infer_ty);

				let block_ty = self.infer_block(block);
				// enforce no ret loop
				self.unify(&TyKind::Primitive(PrimitiveKind::Unit), &block_ty);

				self.loops.pop().unwrap()
			}

			hir::ExprKind::Unit => TyKind::Primitive(PrimitiveKind::Unit),

			hir::ExprKind::Method { expr, name, params } => todo!(),
			hir::ExprKind::Field { expr, name } => todo!(),
			hir::ExprKind::Deref { expr } => todo!("ensure expr ty is pointer"),

			hir::ExprKind::Assign { target, value } => {
				let ExprKind::Access { path } = &target.kind else {
					todo!("invalid lvalue")
				};

				let target_ty = self.resolve_var_ty(path);
				let value_ty = self.infer_expr(value);
				self.unify(&target_ty, &value_ty)
			}

			hir::ExprKind::Return { expr } => {
				let ty = self.infer_expr(expr);
				self.unify(&self.return_ty.clone(), &ty);

				TyKind::Primitive(PrimitiveKind::Never)
			}
			hir::ExprKind::Break { expr, label } => {
				let ty = self.infer_expr(expr);
				let Some(expected) = self.loops.last().cloned() else {
					todo!("break not in a loop")
				};
				self.unify(&expected, &ty);

				TyKind::Primitive(PrimitiveKind::Never)
			}
			hir::ExprKind::Continue { label } => {
				if self.loops.is_empty() {
					todo!()
				}
				TyKind::Primitive(PrimitiveKind::Never)
			}
		};

		let old = self.expr_type.insert(expr.expr_id(), ty.clone());
		// TODO
		assert!(old.is_none());

		ty
	}
}

/// Unification
#[expect(clippy::match_same_arms)]
impl Inferer<'_> {
	fn unify(&mut self, expected: &InferExprTy, actual: &InferExprTy) -> InferExprTy {
		match (expected, actual) {
			(TyKind::Infer(infer), ty) | (ty, TyKind::Infer(infer)) => self.unify_infer(*infer, ty),
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
				self.tcx.dcx().emit_build(report);
				TyKind::Error
			}
		}
	}

	fn unify_infer(&mut self, infer: Infer, other: &InferExprTy) -> InferExprTy {
		let unified = match (infer.kind, other) {
			(
				InferKind::Integer,
				ty @ TyKind::Primitive(PrimitiveKind::UnsignedInt | PrimitiveKind::SignedInt),
			) => ty.clone(),
			(InferKind::Float, ty @ TyKind::Primitive(PrimitiveKind::Float)) => ty.clone(),
			(InferKind::Generic | InferKind::Explicit, ty) => ty.clone(),

			(_, TyKind::Infer(actual_infer)) => {
				if infer.kind == actual_infer.kind {
					TyKind::Infer(*actual_infer)
				} else {
					let report = errors::ty::infer_kind_unification_mismatch(
						infer.kind,
						self.ty_var_expr_map[&infer.tvid],
						actual_infer.kind,
						self.ty_var_expr_map[&actual_infer.tvid],
					);
					self.tcx.dcx().emit_build(report);
					TyKind::Error
				}
			}
			(_, ty) => {
				let report = errors::ty::infer_ty_unification_mismatch(
					infer.kind,
					self.ty_var_expr_map[&infer.tvid],
					ty,
					todo!(),
				);
				self.tcx.dcx().emit_build(report);
				TyKind::Error
			}
		};

		self.infer_map.insert(infer.tvid, unified.clone());

		unified
	}
}
