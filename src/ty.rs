use std::{
	cell::{Ref, RefCell},
	fmt,
	fmt::Write,
	rc::Rc,
};

use rustc_hash::FxHashMap;

use crate::{
	ast::Ident,
	bug,
	collect::{DefId, LangItem, ModuleId, Namespace, PerNamespace, TypeLangItem},
	hir::{self, ExprId, Visitor, visit},
	inference::TypeVarId,
	resolve::Res,
	session::{ArtefactKind, DcxHandle, ScxHandle, SessionCtx, Span},
	symbols::{Symbol, sym},
};

#[derive(Debug)]
pub(crate) struct Put<T> {
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
	pub(crate) fn put(&self, value: T) {
		let mut inner = self.inner.borrow_mut();
		let before = inner.replace(value);

		if let Some(before) = before {
			panic!("you can only put a value once inside a `Put<T>`")
		}
	}

	#[track_caller]
	pub(crate) fn borrow(&self) -> Ref<'_, T> {
		let borrow = self.inner.borrow();
		if borrow.is_some() {
			Ref::map(borrow, |op| op.as_ref().unwrap())
		} else {
			panic!("`Put<T>` has not yet been computed")
		}
	}
}

#[derive(Debug)]
pub(crate) struct PutMap<K, V> {
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
	pub(crate) fn put_key(&self, key: K, value: V) {
		let mut inner = self.inner.borrow_mut();
		let before = inner.insert(key, value);

		if let Some(before) = before {
			panic!("you can only put a value once inside a `PutMap<T>` for some key")
		}
	}

	pub(crate) fn borrow_key(&self, key: &K) -> Ref<'_, V> {
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
pub(crate) struct TyCtx<'scx> {
	scx: &'scx SessionCtx,

	pub(crate) name_env: &'scx PerNamespace<FxHashMap<(ModuleId, Symbol), DefId>>,
	lang_items: &'scx FxHashMap<LangItem, DefId>,

	pub(crate) main_fn_id: Put<DefId>,
	pub(crate) type_env: Put<FxHashMap<DefId, Rc<LateTy>>>,
	// per function
	pub(crate) typeck_results: PutMap<DefId, FxHashMap<ExprId, Rc<LateTy>>>,
}

pub(crate) trait TcxHandle {
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
	pub(crate) fn new(
		scx: &'scx SessionCtx,
		name_env: &'scx PerNamespace<FxHashMap<(ModuleId, Symbol), DefId>>,
		lang_items: &'scx FxHashMap<LangItem, DefId>,
	) -> Self {
		Self {
			scx,

			name_env,
			lang_items,

			main_fn_id: Put::default(),
			type_env: Put::default(),
			typeck_results: PutMap::default(),
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Param<Ty> {
	pub(crate) name: Ident,
	pub(crate) ty: Rc<Ty>,
	pub(crate) id: hir::NodeId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FnDecl<Ty> {
	pub(crate) inputs: Vec<Param<Ty>>,
	pub(crate) output: Rc<Ty>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Struct<Ty> {
	// pub(crate) generics: ast::Generics,
	pub(crate) fields: Vec<FieldDef<Ty>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FieldDef<Ty> {
	pub(crate) name: Ident,
	pub(crate) ty: Rc<Ty>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Enum<Ty> {
	// pub(crate) generics: ast::Generics,
	pub(crate) variants: Vec<Variant<Ty>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Variant<Ty> {
	pub(crate) name: Ident,
	pub(crate) kind: VariantKind<Ty>,
	pub(crate) span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VariantKind<Ty> {
	Unit,
	Struct(Struct<Ty>),
}

/// A concrete type
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TyKind<InferKind, RefKind> {
	// TODO: no primitive kind
	Primitive(PrimitiveKind),
	Pointer(Rc<Self>),

	Fn(FnDecl<Self>),
	// TODO: merge both in an adt construct?
	Struct(Struct<Self>),
	Enum(Enum<Self>),

	// TODO: remove, rust uses query to do recursive type resolution
	/// Refers to the type of another item
	Ref(RefKind),
	Infer(InferKind),
	Error,
}

/// Used during item type collection, gets quickly resolved to [`LateTy`]
pub(crate) type EarlyItemTy = TyKind<NoInfer, DefId>;
/// Used during type inference, gets transformed to [`LateTy`] at the end of type inference (equiv. writeback phase)
pub(crate) type InferExprTy = TyKind<Infer, NoRef>;
/// Final pure *Type* format, it contains no type variable or unresolved reference
pub(crate) type LateTy = TyKind<NoInfer, NoRef>;

impl<T: fmt::Display, U: fmt::Display> fmt::Display for TyKind<T, U> {
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
			// Self::Error => bug!("error ty kind should never be shown to end-user"),
			Self::Error => write!(f, "{{error}}"),
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PrimitiveKind {
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
pub(crate) enum NoInfer {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Infer {
	pub(crate) tvid: TypeVarId,
	pub(crate) kind: InferKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InferKind {
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

impl fmt::Display for NoInfer {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match *self {}
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

impl<Ref: Clone> TyKind<NoInfer, Ref> {
	#[must_use]
	pub(crate) fn as_infer(&self) -> TyKind<Infer, Ref> {
		match self {
			Self::Primitive(kind) => TyKind::Primitive(kind.clone()),
			Self::Pointer(ty) => TyKind::Pointer(Rc::new(ty.as_infer())),
			Self::Fn(decl) => {
				let FnDecl { inputs, output } = decl;
				TyKind::Fn(FnDecl {
					inputs: inputs
						.iter()
						.map(|Param { name, ty, id }| Param {
							name: *name,
							ty: Rc::new(ty.as_infer()),
							id: *id,
						})
						.collect(),
					output: Rc::new(output.as_infer()),
				})
			}
			Self::Struct(struct_) => todo!(),
			Self::Enum(enum_) => todo!(),
			Self::Ref(id) => TyKind::Ref(id.clone()),
			Self::Infer(no_infer) => match *no_infer {},
			Self::Error => TyKind::Error,
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NoRef {}

impl fmt::Display for NoRef {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match *self {}
	}
}

/// Uses the collection step to map every item to a concrete type
pub(crate) fn compute_items_type(tcx: &TyCtx<'_>, hir: &hir::Root) {
	let lang_items = tcx
		.lang_items
		.iter()
		.map(|(k, v)| (*v, k.clone()))
		.collect();

	let mut ty_computer = TypeCollector {
		tcx,

		lang_items,

		early_item_types: FxHashMap::default(),
		trait_impls: FxHashMap::default(),
	};
	ty_computer.visit_root(hir);

	let type_env = compute_item_types(&ty_computer.early_item_types);

	tcx.type_env.put(type_env);

	tcx.scx()
		.register_artefact(&ArtefactKind::TypeEnv(()), |artefact| {
			let env = tcx.type_env.borrow();
			writeln!(artefact, "{env:#?}")
		});
}

fn compute_item_types(
	early_ty_map: &FxHashMap<DefId, EarlyItemTy>,
) -> FxHashMap<DefId, Rc<LateTy>> {
	let mut late_ty_map = FxHashMap::default();

	let def_ids = early_ty_map.keys().copied().collect::<Vec<_>>();

	let mut ctx = visit_ty::Context {
		cycle_detection: Vec::new(),
		early_ty_map,
		late_ty_map: &mut late_ty_map,
	};

	for def_id in def_ids {
		if ctx.late_ty_map.contains_key(&def_id) {
			// already resolved
			continue;
		}

		// visit the type and return one with all references resolved
		let late_ty = visit_ty::visit_ty(&mut ctx, &TyKind::Ref(def_id));

		ctx.late_ty_map.insert(def_id, late_ty).unwrap();
	}

	late_ty_map
}

mod visit_ty {
	use std::rc::Rc;

	use rustc_hash::FxHashMap;

	use crate::{
		collect::DefId,
		ty::{EarlyItemTy, Enum, FieldDef, FnDecl, LateTy, Param, Struct, Variant, VariantKind},
	};

	pub struct Context<'a> {
		pub(crate) cycle_detection: Vec<DefId>,

		pub(crate) early_ty_map: &'a FxHashMap<DefId, EarlyItemTy>,
		pub(crate) late_ty_map: &'a mut FxHashMap<DefId, Rc<LateTy>>,
	}

	pub fn visit_ty(ctx: &mut Context, early_ty: &EarlyItemTy) -> Rc<LateTy> {
		let late_ty = match early_ty {
			EarlyItemTy::Primitive(prim) => LateTy::Primitive(prim.clone()),
			EarlyItemTy::Fn(func) => LateTy::Fn(visit_func(ctx, func)),
			EarlyItemTy::Pointer(ty) => LateTy::Pointer(visit_ty(ctx, ty)),
			EarlyItemTy::Struct(struct_) => LateTy::Struct(visit_struct(ctx, struct_)),
			EarlyItemTy::Enum(enum_) => LateTy::Enum(visit_enum(ctx, enum_)),

			EarlyItemTy::Ref(def_id) => {
				if let Some(ty) = ctx.late_ty_map.get(def_id) {
					return Rc::clone(ty);
				}

				if ctx.cycle_detection.iter().any(|id| id == def_id) {
					todo!("type cycle detected, span on {def_id:?}")
				}
				ctx.cycle_detection.push(*def_id);

				let sub_ty = &ctx.early_ty_map[def_id];
				let new_ty = visit_ty(ctx, sub_ty);

				ctx.late_ty_map.insert(*def_id, Rc::clone(&new_ty));

				return new_ty;
			}

			EarlyItemTy::Infer(infer) => LateTy::Infer(*infer),
			EarlyItemTy::Error => LateTy::Error,
		};
		Rc::new(late_ty)
	}

	fn visit_func(
		ctx: &mut Context,
		FnDecl { inputs, output }: &FnDecl<EarlyItemTy>,
	) -> FnDecl<LateTy> {
		FnDecl {
			inputs: inputs
				.iter()
				.map(|Param { name, ty, id }| Param {
					name: *name,
					ty: visit_ty(ctx, ty),
					id: *id,
				})
				.collect(),
			output: visit_ty(ctx, output),
		}
	}

	fn visit_struct(ctx: &mut Context, Struct { fields }: &Struct<EarlyItemTy>) -> Struct<LateTy> {
		Struct {
			fields: fields
				.iter()
				.map(|FieldDef { name, ty }| FieldDef {
					name: *name,
					ty: visit_ty(ctx, ty),
				})
				.collect(),
		}
	}

	fn visit_enum(ctx: &mut Context, Enum { variants }: &Enum<EarlyItemTy>) -> Enum<LateTy> {
		Enum {
			variants: variants
				.iter()
				.map(|Variant { name, kind, span }| Variant {
					name: *name,
					kind: match kind {
						VariantKind::Unit => VariantKind::Unit,
						VariantKind::Struct(struct_) => {
							VariantKind::Struct(visit_struct(ctx, struct_))
						}
					},
					span: *span,
				})
				.collect(),
		}
	}
}

pub(crate) struct TypeCollector<'tcx> {
	tcx: &'tcx TyCtx<'tcx>,

	lang_items: FxHashMap<DefId, LangItem>,

	pub(crate) early_item_types: FxHashMap<DefId, EarlyItemTy>,
	// type def id -> traits def ids
	pub(crate) trait_impls: FxHashMap<DefId, Vec<DefId>>,
}

impl TypeCollector<'_> {
	/// Lower ty at item level
	pub(crate) fn lower_ty(&self, ty: &hir::Ty) -> EarlyItemTy {
		let _ = self;
		match &ty.kind {
			hir::TyKind::Path(qpath) => match qpath {
				hir::QualifiedPath::Resolved(path) => match path.res {
					Res::Def(def_id) => TyKind::Ref(def_id),
					Res::Local(id) => todo!("no generics rn"),
					Res::SelfTy => todo!(),
					Res::Error => todo!(),
				},
				hir::QualifiedPath::TypeRelative { def_id, segment } => todo!(),
			},
			hir::TyKind::Pointer(ty) => TyKind::Pointer(Rc::new(self.lower_ty(ty))),
			hir::TyKind::Unit => TyKind::Primitive(PrimitiveKind::Unit),
		}
	}

	fn register_ty(&mut self, def_id: DefId, kind: EarlyItemTy) {
		self.early_item_types.insert(def_id, kind);
	}
}

impl visit::Visitor for TypeCollector<'_> {
	fn visit_item(&mut self, item @ hir::Item { kind, span, def_id }: &hir::Item) {
		// TODO: unify with other item kinds, like foreign and trait item
		if let Some(lang_item) = self.lang_items.get(def_id)
			&& let LangItem::Type(lang_item) = lang_item
		{
			let ty = match lang_item {
				TypeLangItem::Never => TyKind::Primitive(PrimitiveKind::Never),
				TypeLangItem::Bool => TyKind::Primitive(PrimitiveKind::Bool),
				TypeLangItem::UInt => TyKind::Primitive(PrimitiveKind::UnsignedInt),
				TypeLangItem::SInt => TyKind::Primitive(PrimitiveKind::SignedInt),
				TypeLangItem::Float => TyKind::Primitive(PrimitiveKind::Float),
			};

			self.register_ty(*def_id, ty);

			return;
		}

		match kind {
			hir::ItemKind::Function(hir::Function { name, decl, body }) => {
				let fn_decl = FnDecl {
					inputs: decl
						.params
						.iter()
						.map(|hir::Param { name, ty, id }| Param {
							name: *name,
							ty: Rc::new(self.lower_ty(ty)),
							id: *id,
						})
						.collect(),
					output: Rc::new(self.lower_ty(&decl.ret)),
				};

				self.register_ty(*def_id, TyKind::Fn(fn_decl));
			}

			hir::ItemKind::Struct(hir::Struct {
				name,
				generics,
				fields,
			}) => {
				let struct_ = Struct {
					fields: fields
						.iter()
						.map(|hir::FieldDef { name, ty }| FieldDef {
							name: *name,
							ty: Rc::new(self.lower_ty(ty)),
						})
						.collect(),
				};

				self.register_ty(*def_id, TyKind::Struct(struct_));
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
				let type_def_id = type_.res.into_def().unwrap();
				let trait_def_id = trait_.res.into_def().unwrap();
				self.trait_impls
					.entry(type_def_id)
					.or_default()
					.push(trait_def_id);

				self.visit_trait_items(members);
			}

			hir::ItemKind::TypeAlias(hir::TypeAlias { name, alias }) => {
				let Some(alias) = alias else {
					let report = errors::type_alias_empty(*span);
					self.tcx.dcx().emit_build(report);
					return;
				};

				let ty = self.lower_ty(alias);
				self.register_ty(*def_id, ty);
			}

			hir::ItemKind::ForeignMod { items } => self.visit_foreign_items(items),
		}
	}

	fn visit_trait_item(
		&mut self,
		hir::Item { kind, span, def_id }: &hir::Item<hir::TraitItemKind>,
	) {
		todo!()
	}

	fn visit_foreign_item(
		&mut self,
		hir::Item { kind, span, def_id }: &hir::Item<hir::ForeignItemKind>,
	) {
		match kind {
			hir::ForeignItemKind::Function(hir::Function { name, decl, body }) => {
				let fn_decl = FnDecl {
					inputs: decl
						.params
						.iter()
						.map(|hir::Param { name, ty, id }| Param {
							name: *name,
							ty: Rc::new(self.lower_ty(ty)),
							id: *id,
						})
						.collect(),
					output: Rc::new(self.lower_ty(&decl.ret)),
				};

				self.register_ty(*def_id, TyKind::Fn(fn_decl));
			}
		}
	}
}

pub(crate) fn check_entrypoint(tcx: &TyCtx<'_>) {
	let main_path = (ModuleId::ROOT, sym::main);
	let Some(def_id) = tcx.name_env[Namespace::Value].get(&main_path) else {
		let report = errors::no_main_function();
		tcx.dcx().emit_build(report);
		return;
	};

	let main_ty = &tcx.type_env.borrow()[def_id];

	let expected_main_ty = LateTy::Fn(FnDecl {
		inputs: vec![],
		output: LateTy::Primitive(PrimitiveKind::Unit).into(),
	});

	if **main_ty != expected_main_ty {
		let report = errors::main_function_wrong_signature(todo!());
		tcx.dcx().emit_build(report);
	}

	tcx.main_fn_id.put(*def_id);
}

pub(crate) mod errors {
	use ariadne::{Label, ReportKind};

	use crate::session::{Report, ReportBuilder, Span};

	pub fn function_cannot_infer_signature(io_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, io_span)
			.with_message("function cannot infer its signature")
			.with_label(Label::new(io_span).with_message("specify a concrete type"))
	}

	pub fn type_alias_empty(item_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, item_span)
			.with_message("type alias have to be defined outside trait definitions")
			.with_label(Label::new(item_span).with_message("define this type alias"))
	}

	pub fn no_main_function() -> ReportBuilder {
		Report::build(ReportKind::Error, Span::DUMMY).with_message("no main function")
	}

	pub fn main_function_wrong_signature(fn_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, fn_span)
			.with_message("main function doesn't match the expected signature")
			.with_label(Label::new(fn_span).with_message("here"))
	}
}
