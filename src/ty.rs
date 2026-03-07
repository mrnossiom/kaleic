use std::{cell::RefCell, fmt, sync::atomic::AtomicU32};

use rustc_hash::FxHashMap;

use crate::{
	ast::{self, Ident},
	bug, errors,
	hir::{self, ExprId, Function, ItemId},
	inference::{InferTag, Inferer},
	resolve::{self, NameEnvironment},
	session::{SessionCtx, Span},
};

#[derive(Debug)]
pub struct TyCtx<'scx> {
	pub scx: &'scx SessionCtx,

	// TODO: this is going to disappear
	pub(crate) infer_tag_count: AtomicU32,

	// TODO: get rid of refcells
	pub name_env: RefCell<NameEnvironment>,
	pub ty_env: RefCell<FxHashMap<ItemId, TyKind>>,
	pub typeck_results: RefCell<FxHashMap<ExprId, TyKind>>,
}

impl<'scx> TyCtx<'scx> {
	#[must_use]
	pub fn new(scx: &'scx SessionCtx) -> Self {
		Self {
			scx,
			infer_tag_count: AtomicU32::default(),

			name_env: RefCell::default(),
			ty_env: RefCell::default(),
			typeck_results: RefCell::default(),
		}
	}
}

/// Context actions
impl TyCtx<'_> {
	/// Goes through the HIR and maps all items
	pub fn collect_items(&self, hir: &hir::Root) {
		let mut cltr = resolve::Collector::new(self);
		cltr.collect_root(hir);
		self.name_env.replace(cltr.name_env);
	}

	/// Uses the collection step to map every item to a concrete type
	pub(crate) fn compute_items_type(&self, hir: &hir::Root) {
		let mut ty_computer = resolve::TypeComputer::new(self);

		let name_env = self.name_env.borrow();
		ty_computer.compute_root(hir);

		self.ty_env.replace(ty_computer.types);
	}

	/// Computes inference for every function body and stores the result
	pub(crate) fn typeck(&self, hir: &hir::Root) {
		// let mut ty_checker = typeck::TypeCheck::new(self);
		// ty_checker.typeck(hir);
		todo!();
		// self.typeck_info.replace(Some(ty_checker.info));
	}

	/// TODO: remove old inference
	pub(crate) fn typeck_old(&self, hir: &hir::Root) {
		let mut expr_tys = FxHashMap::default();

		for item in &hir.items {
			self.typeck_item_old(item, &mut expr_tys);
		}

		*self.typeck_results.borrow_mut() = expr_tys;
	}

	fn typeck_item_old(&self, item: &hir::Item, expr_tys: &mut FxHashMap<ExprId, TyKind>) {
		match &item.kind {
			hir::ItemKind::Function(Function { name, decl, body }) => {
				let Some(body) = body.as_ref() else { return };

				let ty_env = self.ty_env.borrow();
				let TyKind::Fn(decl) = ty_env.get(&item.item_id()).unwrap() else {
					todo!()
				};

				let function_expr_tys = self.typeck_fn_old(*name, decl, body);
				expr_tys.extend(function_expr_tys);
			}
			hir::ItemKind::Extern { items } => {
				for item in items {
					self.typeck_item_old(&item.clone().into(), expr_tys);
				}
			}
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
				todo!()
			}
		}
	}

	#[must_use]
	fn typeck_fn_old(
		&self,
		name: Ident,
		decl: &FnDecl,
		body: &hir::Block,
	) -> FxHashMap<ExprId, TyKind> {
		let ty_env = self.ty_env.borrow();
		let name_env = self.name_env.borrow();

		let mut inferer = Inferer::new(self, decl, body, &name_env, &ty_env);
		inferer.infer_fn();

		let mut expr_tys = FxHashMap::default();

		for (node_id, ty_infer) in inferer.expr_type {
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
								let span = todo!("get from expr_id");
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

		expr_tys
	}

	pub fn lower_ty(&self, ty: &ast::Ty) -> TyKind<Infer> {
		match &ty.kind {
			ast::TyKind::Path(path) => self.lower_path_ty(path),
			ast::TyKind::Pointer(ty) => TyKind::Pointer(Box::new(self.lower_ty(ty))),
			ast::TyKind::Reference(ty) => todo!(),
			ast::TyKind::Unit => TyKind::Primitive(PrimitiveKind::Unit),
			ast::TyKind::ImplicitInfer => TyKind::Infer(self.next_infer_tag(), Infer::Generic),
		}
	}

	pub fn lower_path_ty(&self, path: &ast::Path) -> TyKind<Infer> {
		let path = path.simple();

		let primitive = match self.scx.symbols.resolve(path.sym).as_str() {
			"_" => Some(TyKind::Infer(self.next_infer_tag(), Infer::Explicit)),

			"void" => Some(TyKind::Primitive(PrimitiveKind::Unit)),
			"never" => Some(TyKind::Primitive(PrimitiveKind::Never)),

			"bool" => Some(TyKind::Primitive(PrimitiveKind::Bool)),
			"uint" => Some(TyKind::Primitive(PrimitiveKind::UnsignedInt)),
			"sint" => Some(TyKind::Primitive(PrimitiveKind::SignedInt)),
			"float" => Some(TyKind::Primitive(PrimitiveKind::Float)),

			"str" => Some(TyKind::Primitive(PrimitiveKind::Str)),
			_ => None,
		};

		if let Some(primitive) = primitive {
			primitive
		} else {
			let item_map = self.name_env.borrow();
			if let Some(item_id) = item_map.types.get(&path.sym) {
				// TODO: we could access the real type directly if we sorted
				// in some kind of topological order
				TyKind::Ref(*item_id)
			} else {
				eprintln!("item {:?} doesn't exist", path.sym);
				TyKind::Error
			}
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
	pub generics: ast::Generics,
	pub fields: Vec<FieldDef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldDef {
	pub name: ast::Ident,
	pub ty: TyKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Enum {
	pub generics: ast::Generics,
	pub variants: Vec<Variant>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variant {
	pub name: Ident,
	// pub kind: VariantKind,
	pub span: Span,
}

/// A concrete type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TyKind<InferKind = NoInfer> {
	// TODO: no primitive kind
	Primitive(PrimitiveKind),
	Pointer(Box<Self>),

	Fn(Box<FnDecl>),
	// TODO: merge both in an adt construct?
	Struct(Box<Struct>),
	Enum(Box<Enum>),

	// TODO: remove
	/// Refers to the type of another item
	Ref(ItemId),

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
			Self::Ref(_) => bug!("ref ty kind should be resolved before it's shown to end-user"),
			// TODO
			Self::Error => bug!("error ty kind should never be shown to end-user"),
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimitiveKind {
	Unit,
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
			Self::Unit => write!(f, "()"),
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
			Self::Ref(id) => TyKind::Ref(id),
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
			Self::Ref(id) => Ok(TyKind::Ref(id)),
			Self::Error => Ok(TyKind::Error),
		}
	}
}
