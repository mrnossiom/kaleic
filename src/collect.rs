use std::{collections::hash_map::Entry, fmt, fmt::Write, num::NonZero, ops};

use rustc_hash::FxHashMap;

use crate::{
	ast,
	ast::{NodeId, Visitor, visit},
	attrs::{RegisterLangItem, try_parse_attr},
	session::{ArtefactKind, DcxHandle, SessionCtx},
	symbols::{Symbol, sym},
};

pub(crate) fn collect_root(scx: &SessionCtx, ast: &ast::Root) {
	let mut collector = Collector {
		scx,

		name_env: PerNamespace::default(),
		lang_items: FxHashMap::default(),
		node_id_to_def_id: FxHashMap::default(),
		modules: FxHashMap::default(),

		next_local_def_id: NonZero::new(1).unwrap(),
		next_module_id: NonZero::new(1).unwrap(),
		current_module: ModuleId::ROOT,
	};
	collector.visit_root(ast);

	let Collector {
		name_env,
		lang_items,
		node_id_to_def_id,
		modules,
		..
	} = collector;

	scx.register_artefact(&ArtefactKind::NameEnv(()), |artefact| {
		writeln!(artefact, "{name_env:#?}")
	});

	scx.name_env.put(name_env);
	scx.lang_items.put(lang_items);
	scx.node_id_to_def_id.put(node_id_to_def_id);
	scx.modules.put(modules);
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DefId(NonZero<u32>);

impl fmt::Debug for DefId {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		// def id -> did#x
		// TODO: global def id -> did{<package id>}#x
		write!(f, "did#{}", self.0)
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ModuleId(u32);

impl ModuleId {
	pub(crate) const ROOT: Self = Self(0);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Namespace {
	Type,
	Value,
}

impl fmt::Display for Namespace {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Type => write!(f, "type"),
			Self::Value => write!(f, "value"),
		}
	}
}

#[derive(Debug, Clone, Default)]
pub(crate) struct PerNamespace<T> {
	types: T,
	values: T,
}

impl<T> PerNamespace<T> {
	pub(crate) fn map_all_ns<U>(&self, f: impl Fn(&T) -> U) -> PerNamespace<U> {
		PerNamespace {
			types: f(&self.types),
			values: f(&self.values),
		}
	}
}

impl<T> ops::Index<Namespace> for PerNamespace<T> {
	type Output = T;
	fn index(&self, index: Namespace) -> &Self::Output {
		match index {
			Namespace::Type => &self.types,
			Namespace::Value => &self.values,
		}
	}
}

impl<T> ops::IndexMut<Namespace> for PerNamespace<T> {
	fn index_mut(&mut self, index: Namespace) -> &mut Self::Output {
		match index {
			Namespace::Type => &mut self.types,
			Namespace::Value => &mut self.values,
		}
	}
}

#[derive(Debug)]
struct Collector<'scx> {
	scx: &'scx SessionCtx,

	pub(crate) name_env: PerNamespace<FxHashMap<(ModuleId, Symbol), DefId>>,
	pub(crate) lang_items: FxHashMap<LangItem, DefId>,
	pub(crate) node_id_to_def_id: FxHashMap<ast::NodeId, DefId>,
	// TODO: make module resolution flat? intern entire paths?
	pub(crate) modules: FxHashMap<(ModuleId, Symbol), ModuleId>,

	next_local_def_id: NonZero<u32>,
	next_module_id: NonZero<u32>,
	current_module: ModuleId,
}

impl Collector<'_> {
	fn create_def_id(&mut self, ast_id: NodeId) -> DefId {
		let def_id = DefId(self.next_local_def_id);
		self.next_local_def_id = self.next_local_def_id.checked_add(1).unwrap();
		self.node_id_to_def_id.insert(ast_id, def_id);
		def_id
	}

	fn create_module_id(&mut self, parent: ModuleId, name: Symbol) -> ModuleId {
		let module_id = ModuleId(self.next_module_id.get());
		self.next_module_id = self.next_module_id.checked_add(1).unwrap();
		self.modules.insert((parent, name), module_id);
		module_id
	}

	fn register_lang_item(&mut self, kind: LangItem, def_id: DefId) {
		let before = self.lang_items.insert(kind, def_id);
		assert!(before.is_none());
	}

	fn register_def(&mut self, ns: Namespace, def_id: DefId, name: &ast::Ident) {
		match self.name_env[ns].entry((self.current_module, name.sym)) {
			Entry::Vacant(vacant) => _ = vacant.insert(def_id),
			Entry::Occupied(occupied) => {
				let (span1, span2) = todo!(
					"duplicate item, get spans for {:?} and {:?}",
					occupied.get(),
					def_id
				);
				let report = errors::item_name_conflict(span1, span2, ns);
				self.scx.dcx().emit_build(report);
			}
		}
	}

	fn with_module(&mut self, module_name: Symbol, f: impl FnOnce(&mut Self)) {
		let parent = self.current_module;
		self.current_module = self.create_module_id(parent, module_name);
		f(self);
		self.current_module = parent;
	}
}

impl visit::Visitor for Collector<'_> {
	fn visit_item(
		&mut self,
		item @ ast::Item {
			attrs,
			kind,
			span,
			id,
		}: &ast::Item,
	) {
		let def_id = self.create_def_id(*id);

		for attr in attrs {
			if let Some(parsed_attr) = try_parse_attr::<RegisterLangItem>(self.scx, attr) {
				match parsed_attr {
					Ok(RegisterLangItem {
						lang_item: lang_item_kind,
					}) => self.register_lang_item(lang_item_kind, def_id),
					Err(diag) => self.scx.dcx().emit(&diag),
				}
			}
		}

		match kind {
			ast::ItemKind::ExternImport { .. } => {
				todo!(
					"packages are not yet developed, use \n#path(\"path/to/tube/lib.rs\")\nmod tube_name;"
				)
			}
			ast::ItemKind::Import { tree } => {
				// TODO: register re-exports?
				// everything else is handled during import resolution
			}

			ast::ItemKind::Module {
				name,
				items,
				inline: _,
			} => {
				self.with_module(name.sym, |this| {
					this.visit_items(items);
				});
			}

			ast::ItemKind::Struct { name, .. }
			| ast::ItemKind::Enum { name, .. }
			| ast::ItemKind::TypeAlias(ast::TypeAlias { name, .. })
			| ast::ItemKind::Trait { name, .. } => {
				self.register_def(Namespace::Type, def_id, name);
			}

			ast::ItemKind::Function(ast::Function { name, .. }) => {
				self.register_def(Namespace::Value, def_id, name);
			}
			ast::ItemKind::ForeignMod { items } => self.visit_items(items),

			ast::ItemKind::TraitImpl { .. } => {
				// nothing to collect
				//
				// children items are visited during ty to ensure they match the trait definition
			}
		}
	}
}

mod errors {
	use ariadne::{Label, ReportKind};

	use crate::{
		collect::Namespace,
		session::{Report, ReportBuilder, Span},
	};

	pub fn item_name_conflict(
		original: Span,
		conflicted: Span,
		namespace: Namespace,
	) -> ReportBuilder {
		Report::build(ReportKind::Error, original)
			.with_message(format!(
				"distinct {namespace} items have a conflicting name"
			))
			.with_label(Label::new(original).with_message("this is the first item encountered"))
			.with_label(Label::new(conflicted).with_message("this item has the same name"))
	}
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum LangItem {
	Trait(TraitLangItem),
	Type(TypeLangItem),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum TraitLangItem {
	Add,
	AddAssign,
	Sub,
	SubAssign,
	Mul,
	MulAssign,
	Div,
	DivAssign,
	Not,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum TypeLangItem {
	Never,
	Bool,
	UInt,
	SInt,
	Float,
}

impl LangItem {
	pub(crate) fn parse(sym: Symbol) -> Option<Self> {
		match sym {
			sym::AddTrait => Some(Self::Trait(TraitLangItem::Add)),
			sym::AddAssignTrait => Some(Self::Trait(TraitLangItem::AddAssign)),
			sym::SubTrait => Some(Self::Trait(TraitLangItem::Sub)),
			sym::SubAssignTrait => Some(Self::Trait(TraitLangItem::SubAssign)),
			sym::MulTrait => Some(Self::Trait(TraitLangItem::Mul)),
			sym::MulAssignTrait => Some(Self::Trait(TraitLangItem::MulAssign)),
			sym::DivTrait => Some(Self::Trait(TraitLangItem::Div)),
			sym::DivAssignTrait => Some(Self::Trait(TraitLangItem::DivAssign)),
			sym::NotTrait => Some(Self::Trait(TraitLangItem::Not)),
			sym::never_ty => Some(Self::Type(TypeLangItem::Never)),
			sym::bool_ty => Some(Self::Type(TypeLangItem::Bool)),
			sym::uint_ty => Some(Self::Type(TypeLangItem::UInt)),
			sym::sint_ty => Some(Self::Type(TypeLangItem::SInt)),
			sym::float_ty => Some(Self::Type(TypeLangItem::Float)),
			_ => None,
		}
	}
}
