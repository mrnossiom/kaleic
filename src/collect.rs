use std::{
	collections::hash_map::Entry,
	fmt::{self, Write},
};

use rustc_hash::FxHashMap;

use crate::{
	ast::{self, NodeId, Visitor, visit},
	attrs::{RegisterLangItem, try_parse_attr},
	resolve::LangItem,
	session::{ArtefactKind, DcxHandle, SessionCtx},
	symbols::Symbol,
};

pub(crate) fn collect_root(scx: &SessionCtx, ast: &ast::Root) {
	let mut collector = Collector::new(scx);
	collector.visit_root(ast);

	let Collector {
		name_env,
		lang_items,
		node_id_to_def_id,
		..
	} = collector;

	scx.register_artefact(&ArtefactKind::NameEnv(()), |artefact| {
		writeln!(artefact, "{name_env:#?}")
	});

	scx.name_env.put(name_env);
	scx.lang_items.put(lang_items);
	scx.node_id_to_def_id.put(node_id_to_def_id);
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DefId(u32);

impl fmt::Debug for DefId {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		// def id -> did#x
		// TODO: global def id -> did{<package id>}#x
		write!(f, "did#{}", self.0)
	}
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
pub(crate) struct NameEnvironment {
	pub(crate) types: FxHashMap<Symbol, DefId>,
	pub(crate) values: FxHashMap<Symbol, DefId>,
}

#[derive(Debug)]
struct Collector<'scx> {
	scx: &'scx SessionCtx,

	pub(crate) name_env: NameEnvironment,
	pub(crate) lang_items: FxHashMap<LangItem, DefId>,
	pub(crate) node_id_to_def_id: FxHashMap<ast::NodeId, DefId>,

	next_local_def_id: u32,
}

impl<'scx> Collector<'scx> {
	#[must_use]
	pub(crate) fn new(scx: &'scx SessionCtx) -> Self {
		Self {
			scx,
			name_env: NameEnvironment::default(),
			lang_items: FxHashMap::default(),
			node_id_to_def_id: FxHashMap::default(),
			next_local_def_id: 0,
		}
	}
}

impl Collector<'_> {
	fn create_def(&mut self, ast_id: NodeId) -> DefId {
		let def_id = DefId(self.next_local_def_id);
		self.next_local_def_id += 1;
		self.node_id_to_def_id.insert(ast_id, def_id);
		def_id
	}

	fn register_lang_item(&mut self, kind: LangItem, def_id: DefId) {
		let before = self.lang_items.insert(kind, def_id);
		assert!(before.is_none());
	}

	fn register_def(&mut self, ns: Namespace, def_id: DefId, name: &ast::Ident) {
		let map = match ns {
			Namespace::Type => &mut self.name_env.types,
			Namespace::Value => &mut self.name_env.values,
		};

		match map.entry(name.sym) {
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
		let def_id = self.create_def(*id);

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
			ast::ItemKind::Struct { name, .. }
			| ast::ItemKind::Enum { name, .. }
			| ast::ItemKind::TypeAlias(ast::TypeAlias { name, .. })
			| ast::ItemKind::Trait { name, .. } => {
				self.register_def(Namespace::Type, def_id, name);
			}

			ast::ItemKind::Module {
				name,
				items,
				inline,
			} => {}

			ast::ItemKind::Function(ast::Function { name, .. }) => {
				self.register_def(Namespace::Value, def_id, name);
			}
			ast::ItemKind::ForeignMod { items } => self.visit_items(items),

			ast::ItemKind::ExternUse { .. } => {
				todo!(
					"packages are not yet developed, use \n#path(\"path/to/tube/lib.rs\")\nmod tube_name;"
				)
			}

			ast::ItemKind::TraitImpl { .. } => {
				// nothing to collect, type environment with collect trait
				// implementations and method resolution will select the implementation
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
