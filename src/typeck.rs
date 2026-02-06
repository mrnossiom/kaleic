use std::collections::HashMap;

use crate::{
	ast::Stmt,
	hir::{self, Enum, Expr, Function, ItemKind, NodeId, Struct, TypeAlias},
	lexer,
	session::Symbol,
	ty::{self, TyKind},
};

pub struct TypeCheck<'tcx> {
	tcx: &'tcx ty::TyCtx<'tcx>,

	expr_types: HashMap<NodeId, TyKind>,
}

impl<'tcx> TypeCheck<'tcx> {
	pub fn new(tcx: &'tcx ty::TyCtx) -> Self {
		Self {
			tcx,
			expr_types: HashMap::new(),
		}
	}

	pub fn typeck(&self, hir: &hir::Root) {
		for item in &hir.items {
			self.typeck_item(item);
		}
	}

	fn typeck_item(&self, item: &hir::Item) {
		match &item.kind {
			// TODO: validate generics and indirection
			ItemKind::Struct(Struct {
				name,
				generics,
				fields,
			}) => {}
			ItemKind::Enum(Enum {
				name,
				generics,
				variants,
			}) => {}
			ItemKind::TypeAlias(TypeAlias { name, alias }) => {}
			ItemKind::Trait {
				name,
				generics,
				members,
			} => {}
			ItemKind::TraitImpl {
				type_,
				trait_,
				members,
			} => {
				for item in members {
					// TODO
					// match item.kind {
					// 	 TraitItemKind::Type()
					// }
				}
			}

			ItemKind::Function(func) => {
				FunctionTck {}.infer_fn(func);
			}
		}
	}
}

struct FunctionTck {}

#[derive(Debug, Default)]
struct InferCx(Vec<Entry>);

type InferResult<T = (TyKind, InferCx)> = std::result::Result<T, ()>;

/// Term variable
type TmVar = ();
/// Type variable
type TyVar = ();

#[derive(Debug, Clone)]
enum Entry {
	/// Term variable bind
	TmVarBind(TmVar, TypeAlias),
	/// Type variable bind
	TyVarBind(TyVar),
	/// Existential type variable
	ExVarBind(TyVar),
	/// Solved existential type variable
	SxVarBind(TyVar, TypeAlias),

	Mark(TyVar),
}

impl FunctionTck {
	fn subst_type(&self, var: &TyVar, repl: &TypeAlias, ty: &TypeAlias) -> TypeAlias {
		todo!()
	}

	fn apply_ctx(&self, ty: &TypeAlias) -> TypeAlias {
		todo!()
	}

	fn infer_fn(&self, func: &Function) -> HashMap<NodeId, TyKind> {
		let ctx = InferCx::default();

		todo!()
	}

	fn infer_stmt(&self, stmt: &Stmt) -> InferResult {
		todo!()
	}

	fn infer_expr(&self, expr: &Expr) -> InferResult {
		match &expr.kind {
			hir::ExprKind::Access(path) => todo!(),
			hir::ExprKind::Literal(lit, _ident) => match lit {
				lexer::LiteralKind::Integer => todo!(),
				lexer::LiteralKind::Float => todo!(),
				lexer::LiteralKind::Str => todo!(),
			},

			hir::ExprKind::Unary(op, expr) => todo!(),
			hir::ExprKind::Binary(op, left, right) => todo!(),

			hir::ExprKind::FnCall { expr, args } => todo!(),
			hir::ExprKind::If {
				cond,
				conseq,
				altern,
			} => todo!(),

			hir::ExprKind::Method(_expr, _name, _args) => todo!(),
			hir::ExprKind::Field(_expr, _name) => todo!(),
			hir::ExprKind::Deref(_expr) => todo!("ensure expr ty is pointer"),

			hir::ExprKind::Assign { target: _, value } => todo!(),

			hir::ExprKind::Return(_) | hir::ExprKind::Break(_) | hir::ExprKind::Continue => {
				todo!()
			}
		}
	}

	fn infer_var() -> InferResult {
		todo!()
	}
}
