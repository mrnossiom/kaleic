use std::{
	cell::{Ref, RefCell},
	fmt,
};

use rustc_hash::FxHashMap;

use crate::{
	ast::{self, Ident},
	bug,
	hir::{self, ExprId, ItemId},
	inference::{self, TypeVarId},
	resolve::NameEnvironment,
	session::{SessionCtx, Span},
};

#[derive(Default, Debug)]
pub struct Put<T> {
	inner: RefCell<Option<T>>,
}

impl<T> Put<T> {
	#[track_caller]
	pub fn put(&self, value: T) {
		let mut inner = self.inner.borrow_mut();
		let before = inner.replace(value);

		if let Some(before) = before {
			panic!("you can only put a value once inside a `Put<T>`")
		}
	}

	#[track_caller]
	pub fn borrow(&self) -> Ref<'_, T> {
		Ref::map(self.inner.borrow(), |op| {
			if let Some(op) = op.as_ref() {
				op
			} else {
				panic!("`Put<T>` has not yet been computed")
			}
		})
	}
}

#[derive(Debug)]
pub struct PutMap<K, V> {
	inner: RefCell<FxHashMap<K, V>>,
}

impl<K, V> Default for PutMap<K, V> {
	fn default() -> Self {
		Self {
			inner: RefCell::default(),
		}
	}
}

impl<K: Eq + std::hash::Hash, V> PutMap<K, V> {
	pub fn put_key(&self, key: K, value: V) {
		let mut inner = self.inner.borrow_mut();
		let before = inner.insert(key, value);

		if let Some(before) = before {
			panic!("you can only put a value once inside a `PutMap<T>` for some key")
		}
	}

	pub fn borrow_key(&self, key: &K) -> Ref<'_, V> {
		Ref::map(self.inner.borrow(), |op| {
			if let Some(op) = op.get(key) {
				op
			} else {
				panic!("`PutMap<T>` for this key has not yet been computed")
			}
		})
	}
}

#[derive(Debug)]
pub struct TyCtx<'scx> {
	pub scx: &'scx SessionCtx,

	pub arena: (),

	pub name_env: Put<NameEnvironment>,
	pub type_env: Put<FxHashMap<ItemId, TyKind>>,
	// per function
	pub typeck_results: PutMap<ItemId, FxHashMap<ExprId, TyKind>>,
}

impl<'scx> TyCtx<'scx> {
	#[must_use]
	pub fn new(scx: &'scx SessionCtx) -> Self {
		Self {
			scx,

			arena: (),

			name_env: Put::default(),
			type_env: Put::default(),
			typeck_results: PutMap::default(),
		}
	}
}

/// Context actions
impl TyCtx<'_> {
	/// Uses the collection step to map every item to a concrete type
	pub(crate) fn compute_items_type(&self, hir: &hir::Root) {
		let mut ty_computer = TypeComputer::new(self);

		let name_env = self.name_env.borrow();
		ty_computer.compute_root(hir);

		self.type_env.put(ty_computer.types);
	}

	/// Compute inference for every function body and stores the result
	pub(crate) fn typeck(&self, hir: &hir::Root) {
		inference::infer_root(self, hir);
	}

	pub fn lower_ty(&self, ty: &ast::Ty) -> TyKind {
		match &ty.kind {
			ast::TyKind::Path(path) => self.lower_path_ty(path),
			ast::TyKind::Pointer(ty) => TyKind::Pointer(Box::new(self.lower_ty(ty))),
			ast::TyKind::Reference(ty) => todo!(),
			ast::TyKind::Unit => TyKind::Primitive(PrimitiveKind::Unit),
		}
	}

	pub fn lower_path_ty(&self, path: &ast::Path) -> TyKind {
		let path = path.simple();

		let primitive = match self.scx.symbols.resolve(path.sym).as_str() {
			// let report = errors::ty::function_cannot_infer_signature(decl.ret.span);
			// self.tcx.scx.dcx().emit_build(report);
			// TyKind::Error
			"_" => todo!(),

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
				let node_id = todo!();
				TyKind::Ref(node_id)
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

	Infer(InferKind),
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
			Self::Infer(infer) => infer.fmt(f),
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
pub struct Infer {
	pub tvid: TypeVarId,
	pub kind: InferKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferKind {
	Integer,
	Float,

	Generic,
	Explicit,
}

impl fmt::Display for Infer {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.kind.fmt(f)
	}
}

impl fmt::Display for InferKind {
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
	pub fn as_no_infer(self) -> Result<TyKind<NoInfer>, Infer> {
		match self {
			Self::Primitive(kind) => Ok(TyKind::Primitive(kind)),
			Self::Pointer(kind) => Ok(TyKind::Pointer(Box::new(kind.as_no_infer()?))),
			Self::Fn(fn_) => Ok(TyKind::Fn(fn_)),
			// Self::Adt(()) => Ok(TyKind::Adt(())),
			Self::Struct(struct_) => Ok(TyKind::Struct(struct_)),
			Self::Enum(enum_) => Ok(TyKind::Enum(enum_)),
			Self::Infer(infer) => Err(infer),
			Self::Ref(id) => Ok(TyKind::Ref(id)),
			Self::Error => Ok(TyKind::Error),
		}
	}
}

pub struct TypeComputer<'tcx> {
	tcx: &'tcx TyCtx<'tcx>,

	pub(crate) types: FxHashMap<ItemId, TyKind>,
	// trait item id -> implementors
	pub(crate) trait_impls: FxHashMap<ItemId, Vec<()>>,
}

impl<'tcx> TypeComputer<'tcx> {
	#[must_use]
	pub fn new(tcx: &'tcx TyCtx) -> Self {
		Self {
			tcx,
			types: FxHashMap::default(),
			trait_impls: FxHashMap::default(),
		}
	}
}

impl TypeComputer<'_> {
	pub fn compute_root(&mut self, root: &hir::Root) {
		let hir::Root { items } = &root;
		for item in items {
			self.compute_item(item);
		}
	}

	fn compute_item(&mut self, item: &hir::Item) {
		let ty = match &item.kind {
			hir::ItemKind::Struct(hir::Struct {
				name,
				generics,
				fields,
			}) => {
				let struct_ = Struct {
					generics: generics.clone(),
					fields: fields
						.iter()
						.map(|field| self.lower_field_def(field))
						.collect(),
				};
				TyKind::Struct(Box::new(struct_))
			}
			hir::ItemKind::Enum(hir::Enum {
				name,
				generics,
				variants,
			}) => {
				let enum_ = Enum {
					generics: generics.clone(),
					variants: variants
						.iter()
						.map(|variant| self.lower_variant(variant))
						.collect(),
				};
				TyKind::Enum(Box::new(enum_))
			}

			hir::ItemKind::Trait {
				name,
				generics,
				members,
			} => todo!(),

			hir::ItemKind::TraitImpl { type_, .. } => {
				// TODO
				// self.environment.types.insert(type_.sym, v);
				todo!()
			}

			hir::ItemKind::TypeAlias(hir::TypeAlias { name, alias }) => match &alias {
				Some(ty) => self.tcx.lower_ty(ty),
				None => todo!(
					"error about how standalone empty type aliases are not allowed, only used in traits"
				),
			},

			hir::ItemKind::Function(hir::Function { name, decl, body }) => {
				TyKind::Fn(Box::new(self.lower_fn_decl(decl)))
			}
			hir::ItemKind::ForeignMod { items } => {
				for item in items {
					match &item.kind {
						hir::ForeignItemKind::Function(hir::Function { name, decl, body }) => {
							let ty = TyKind::Fn(Box::new(self.lower_fn_decl(decl)));
							let old = self.types.insert(item.item_id(), ty);
							debug_assert!(old.is_none());
						}
					}
				}
				return;
			}
		};
		let old = self.types.insert(item.item_id(), ty);
		debug_assert!(old.is_none());
	}

	// TODO: not pub
	pub fn lower_fn_decl(&mut self, decl: &hir::FnDecl) -> FnDecl {
		// TODO: diag no infer ty in functions
		let inputs = decl
			.params
			.iter()
			.map(|ast::Param { name, ty }| {
				let ty = self.tcx.lower_ty(ty);
				Param { name: *name, ty }
			})
			.collect();

		let output = self.tcx.lower_ty(&decl.ret);

		FnDecl { inputs, output }
	}

	fn lower_field_def(&self, hir::FieldDef { name, ty }: &hir::FieldDef) -> FieldDef {
		FieldDef {
			name: *name,
			ty: self.tcx.lower_ty(ty),
		}
	}

	fn lower_variant(&self, hir::EnumVariant { name, fields, span }: &hir::EnumVariant) -> Variant {
		Variant {
			name: *name,
			span: *span,
		}
	}
}
