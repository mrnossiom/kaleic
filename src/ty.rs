use std::{
	cell::{Ref, RefCell},
	fmt,
	fmt::Write,
};

use rustc_hash::FxHashMap;

use crate::{
	ast::{self, Ident},
	bug, errors,
	hir::{self, ExprId, Visitor, visit},
	inference::{self, TypeVarId},
	resolve::{DefId, NameEnvironment, Resolution},
	session::{DcxHandle, PrintKind, ScxHandle, SessionCtx, Span},
	symbols::{kw, sym},
};

#[derive(Debug)]
pub struct Put<T> {
	inner: RefCell<Option<T>>,
}

impl<T> Default for Put<T> {
	fn default() -> Self {
		Self {
			inner: RefCell::default(),
		}
	}
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

	pub main_fn_id: Put<DefId>,
	pub type_env: Put<FxHashMap<DefId, TyKind>>,
	// per function
	pub typeck_results: PutMap<DefId, FxHashMap<ExprId, TyKind>>,
}

pub trait TcxHandle {
	fn tcx(&self) -> &TyCtx<'_>;
}

impl TcxHandle for TyCtx<'_> {
	fn tcx(&self) -> &Self {
		self
	}
}

impl<T: TcxHandle> ScxHandle for T {
	fn scx(&self) -> &SessionCtx {
		self.tcx().scx
	}
}

impl<'scx> TyCtx<'scx> {
	#[must_use]
	pub fn new(scx: &'scx SessionCtx) -> Self {
		Self {
			scx,

			arena: (),

			name_env: Put::default(),
			main_fn_id: Put::default(),
			type_env: Put::default(),
			typeck_results: PutMap::default(),
		}
	}
}

/// Context actions
impl TyCtx<'_> {
	/// Uses the collection step to map every item to a concrete type
	pub(crate) fn compute_items_type(&self, hir: &hir::Root) {
		let name_env = self.name_env.borrow();
		let mut ty_computer = TypeComputer::new(self);
		ty_computer.visit_root(hir);

		self.scx.register_artefact(
			&PrintKind::TypeEnvironment,
			"type-environment.txt",
			|artefact| {
				let env = self.type_env.borrow();
				writeln!(artefact, "{env:#?}")
			},
		);

		self.type_env.put(ty_computer.types);
	}

	/// Compute inference for every function body and stores the result
	pub(crate) fn typeck(&self, hir: &hir::Root) {
		inference::infer_root(self, hir);
	}

	pub fn lower_ty(&self, ty: &hir::Ty) -> TyKind {
		match &ty.kind {
			hir::TyKind::Path(path) => self.lower_path_ty(path),
			hir::TyKind::Pointer(ty) => TyKind::Pointer(Box::new(self.lower_ty(ty))),
			hir::TyKind::Unit => TyKind::Primitive(PrimitiveKind::Unit),
		}
	}

	pub fn lower_path_ty(&self, path: &hir::Path) -> TyKind {
		let res = match path.resolved {
			Resolution::Def(def_id) => {}
			Resolution::Local(id) => todo!("no generics rn"),
		};

		// let path = path.simple();

		let primitive = match path.sym {
			// let report = errors::ty::function_cannot_infer_signature(decl.ret.span);
			// self.tcx.dcx().emit_build(report);
			// TyKind::Error
			kw::Underscore => todo!(),

			sym::never => Some(TyKind::Primitive(PrimitiveKind::Never)),
			sym::bool => Some(TyKind::Primitive(PrimitiveKind::Bool)),
			sym::uint => Some(TyKind::Primitive(PrimitiveKind::UnsignedInt)),
			sym::sint => Some(TyKind::Primitive(PrimitiveKind::SignedInt)),
			sym::float => Some(TyKind::Primitive(PrimitiveKind::Float)),
			sym::str => Some(TyKind::Primitive(PrimitiveKind::Str)),
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
	pub name: Ident,
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
	pub name: Ident,
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
	Ref(DefId),

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

	pub(crate) types: FxHashMap<DefId, TyKind>,
	// type def id -> traits def ids
	pub(crate) trait_impls: FxHashMap<DefId, Vec<DefId>>,
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

impl visit::Visitor for TypeComputer<'_> {
	fn visit_item(&mut self, item @ hir::Item { kind, span, id }: &hir::Item) {
		match kind {
			hir::ItemKind::Function(hir::Function { name, decl, body }) => todo!(),

			hir::ItemKind::Struct(hir::Struct {
				name,
				generics,
				fields,
			}) => {
				let field_ids = Vec::new();

				let struct_ = Struct {
					generics: generics.clone(),
					fields: fields
						.iter()
						.map(|field| self.tcx.lower_ty(&field.ty))
						.collect(),
				};
			}
			hir::ItemKind::Enum(hir::Enum {
				name,
				generics,
				variants,
			}) => todo!(),

			hir::ItemKind::Trait {
				name,
				generics,
				members,
			} => todo!(),
			hir::ItemKind::TraitImpl {
				type_,
				trait_,
				members,
			} => {
				// register trait impl for the mentioned type
				let type_def_id = type_.resolved.as_def().unwrap();
				let trait_def_id = trait_.resolved.as_def().unwrap();
				self.trait_impls
					.entry(type_def_id)
					.or_default()
					.push(trait_def_id);

				for member in members {
					visit::visit_trait_item(self, member);
				}
			}

			hir::ItemKind::TypeAlias(hir::TypeAlias { name, alias }) => {
				let Some(alias) = alias else {
					let report = errors::ty::type_alias_empty(*span);
					self.tcx.dcx().emit_build(report);
					return;
				};

				let ty = self.tcx.lower_ty(alias);
			}

			hir::ItemKind::ForeignMod { items } => {
				for item in items {
					visit::visit_foreign_item(self, item)
				}
			}
		};
	}
}
