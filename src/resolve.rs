use std::collections::hash_map::Entry;

use rustc_hash::FxHashMap;

use crate::{
	ast, errors,
	session::{SessionCtx, Symbol},
};

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
pub struct Collector<'scx> {
	scx: &'scx SessionCtx,
	pub(crate) name_env: NameEnvironment,
}

impl<'scx> Collector<'scx> {
	#[must_use]
	pub fn new(scx: &'scx SessionCtx) -> Self {
		Self {
			scx,
			name_env: NameEnvironment::default(),
		}
	}
}

impl ast::Visitor for Collector<'_> {
	fn visit_root(&mut self, ast::Root { attrs, items }: &ast::Root) {
		self.visit_attrs(attrs);
		self.visit_items(items);
	}

	fn visit_attr(&mut self, ast::Attr { path, meta, span }: &ast::Attr) {}

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
