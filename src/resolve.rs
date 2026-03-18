use std::{collections::hash_map::Entry, fmt};

use rustc_hash::FxHashMap;

use crate::{
	ast::{self, Visitor},
	errors, hir,
	session::{DcxHandle, SessionCtx},
	symbols::Symbol,
};

pub(crate) fn resolve_root(scx: &SessionCtx, ast: &ast::Root) -> NameEnvironment {
	let mut cltr = Collector::new(scx);
	cltr.visit_root(ast);

	cltr.name_env
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalDefId(pub u32);

impl fmt::Debug for LocalDefId {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		// local def id -> did
		// def id -> did{<module>}
		write!(f, "did#{}", self.0)
	}
}

#[derive(Debug)]
pub enum LangItemKind {
	AddTrait,
	SubTrait,
	MulTrait,
	DivTrait,
	// and much more :)
}

#[derive(Debug)]
pub enum Namespace {
	Type,
	Value,
}

#[derive(Debug, Default)]
pub struct NameEnvironment {
	pub types: FxHashMap<Symbol, ast::NodeId>,
	pub values: FxHashMap<Symbol, ast::NodeId>,
}

#[derive(Debug)]
struct Collector<'scx> {
	scx: &'scx SessionCtx,
	pub(crate) name_env: NameEnvironment,
	pub(crate) lang_items: FxHashMap<LangItemKind, LocalDefId>,
}

impl<'scx> Collector<'scx> {
	#[must_use]
	pub fn new(scx: &'scx SessionCtx) -> Self {
		Self {
			scx,
			name_env: NameEnvironment::default(),
			lang_items: FxHashMap::default(),
		}
	}
}

impl ast::Visitor for Collector<'_> {
	fn visit_root(&mut self, ast::Root { attrs, items }: &ast::Root) {
		self.visit_attrs(attrs);
		self.visit_items(items);
	}

	fn visit_attr(
		&mut self,
		ast::Attr {
			path,
			meta,
			span,
			id,
		}: &ast::Attr,
	) {
	}

	fn visit_item(
		&mut self,
		ast::Item {
			attrs,
			kind,
			span,
			id,
		}: &ast::Item,
	) {
		match kind {
			ast::ItemKind::Trait { name, .. }
			| ast::ItemKind::Struct { name, .. }
			| ast::ItemKind::Enum { name, .. }
			| ast::ItemKind::TypeAlias(ast::TypeAlias { name, .. }) => {
				match self.name_env.types.entry(name.sym) {
					Entry::Vacant(vacant) => _ = vacant.insert(*id),
					Entry::Occupied(occupied) => {
						let item_id = occupied.get();
						let report =
							errors::ty::item_name_conflict(todo!("{item_id:?}"), *span, "type");
						self.scx.dcx().emit_build(report);
					}
				}
			}

			ast::ItemKind::Function(ast::Function { name, .. }) => {
				match self.name_env.values.entry(name.sym) {
					Entry::Vacant(vacant) => _ = vacant.insert(*id),
					Entry::Occupied(occupied) => {
						let item_id = occupied.get();
						let report =
							errors::ty::item_name_conflict(todo!("{item_id:?}"), *span, "value");
						self.scx.dcx().emit_build(report);
					}
				}
			}
			ast::ItemKind::ForeignMod { items } => {
				self.visit_items(items);
			}

			ast::ItemKind::TraitImpl { .. } => {
				// nothing to collect, type environment with collect trait
				// implementations and method resolution will select the implementation
			}
		}
	}
}

#[derive(Debug, Clone)]
pub enum Resolved {
	// Def(resolve::DefId),
	Def(hir::ItemId),
	Local(hir::ExprId),
}

impl Resolved {
	pub fn as_def(&self) -> Option<hir::ItemId> {
		match self {
			Self::Def(def) => Some(*def),
			_ => None,
		}
	}

	pub fn as_local(&self) -> Option<hir::ExprId> {
		match self {
			Self::Local(id) => Some(*id),
			_ => None,
		}
	}
}
