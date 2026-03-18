use std::{collections::hash_map::Entry, fmt};

use rustc_hash::FxHashMap;

use crate::{
	ast::{self, AttrMeta, ExprKind, Visitor},
	errors, hir,
	session::{DcxHandle, SessionCtx},
	symbols::{Symbol, sym},
};

pub(crate) fn collect_root(scx: &SessionCtx, ast: &ast::Root) -> NameEnvironment {
	let mut collector = Collector::new(scx);
	collector.visit_root(ast);

	dbg!(collector.lang_items);
	collector.name_env
}

pub(crate) fn resolve_root(scx: &SessionCtx, ast: &ast::Root) {
	let mut resolver = Resolver::new(scx);
	resolver.visit_root(ast);

	todo!()
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LangItemKind {
	AddTrait,
	AddAssignTrait,
	SubTrait,
	SubAssignTrait,

	BoolTy,
	UIntTy,
	SIntTy,
	FloatTy,
	// MulTrait,
	// DivTrait,
	// and much more :)
}

impl LangItemKind {
	fn parse(sym: Symbol) -> Option<Self> {
		match sym {
			sym::AddTrait => Some(Self::AddTrait),
			sym::AddAssignTrait => Some(Self::AddAssignTrait),
			sym::SubTrait => Some(Self::SubTrait),
			sym::SubAssignTrait => Some(Self::SubAssignTrait),
			sym::bool_ty => Some(Self::BoolTy),
			sym::uint_ty => Some(Self::UIntTy),
			sym::sint_ty => Some(Self::SIntTy),
			sym::float_ty => Some(Self::FloatTy),
			_ => None,
		}
	}
}

#[derive(Debug)]
pub enum Namespace {
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

impl Collector<'_> {
	fn register_lang_item(&mut self, kind: LangItemKind, item: &ast::Item) {
		let local_def_id = todo!("local def id from item {:?}", item.id);
		let before = self.lang_items.insert(kind, local_def_id);
		assert!(before.is_none());
	}

	fn register_def(&mut self, ns: &Namespace, item: &ast::Item, name: &ast::Ident) {
		let map = match ns {
			Namespace::Type => &mut self.name_env.types,
			Namespace::Value => &mut self.name_env.values,
		};

		match map.entry(name.sym) {
			Entry::Vacant(vacant) => _ = vacant.insert(item.id),
			Entry::Occupied(occupied) => {
				let item_id = occupied.get();
				let report = errors::ty::item_name_conflict(todo!("{item_id:?}"), item.span, ns);
				self.scx.dcx().emit_build(report);
			}
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
		item @ ast::Item {
			attrs,
			kind,
			span,
			id,
		}: &ast::Item,
	) {
		for attr in attrs {
			if attr.path.is_match(&[sym::lang_item]) {
				let AttrMeta::Tuple(exprs) = &attr.meta else {
					todo!("wrong syntax for `lang_item` attr");
				};
				let &[expr] = &exprs.as_slice() else {
					todo!("wrong syntax")
				};
				let ExprKind::LiteralStr { sym } = expr.kind else {
					todo!("wrong syntax")
				};
				let Some(kind) = LangItemKind::parse(sym) else {
					todo!("this lang item doesn't exist")
				};
				self.register_lang_item(kind, item);
			}
		}

		match kind {
			ast::ItemKind::Struct { name, .. }
			| ast::ItemKind::Enum { name, .. }
			| ast::ItemKind::TypeAlias(ast::TypeAlias { name, .. }) => {
				self.register_def(&Namespace::Type, item, name);
			}
			ast::ItemKind::Trait { name, .. } => {
				self.register_def(&Namespace::Type, item, name);
				// TODO: register trait in a separate category
			}

			ast::ItemKind::Function(ast::Function { name, .. }) => {
				self.register_def(&Namespace::Value, item, name);
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

#[derive(Debug, Clone)]
struct Resolver<'scx> {
	scx: &'scx SessionCtx,
}

impl<'scx> Resolver<'scx> {
	#[must_use]
	pub fn new(scx: &'scx SessionCtx) -> Self {
		Self { scx }
	}
}

impl Visitor for Resolver<'_> {
	fn visit_root(&mut self, ast::Root { attrs, items }: &ast::Root) {
		self.visit_attrs(attrs);
		self.visit_items(items);
	}

	fn visit_attr(&mut self, attrs: &ast::Attr) {}

	fn visit_item(
		&mut self,
		ast::Item {
			attrs,
			kind,
			span,
			id,
		}: &ast::Item,
	) {
	}
}
