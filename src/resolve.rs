use std::{collections::hash_map::Entry, fmt};

use rustc_hash::FxHashMap;

use crate::{
	ast::{self, AttrMeta, ExprKind, NodeId, visit},
	errors, hir,
	session::{DcxHandle, Diagnostic, SessionCtx},
	symbols::{Symbol, sym},
};

pub(crate) fn collect_root(scx: &SessionCtx, ast: &ast::Root) -> CollectionResult {
	let mut collector = Collector::new(scx);
	visit::Visitor::visit_root(&mut collector, ast);

	let Collector {
		name_env,
		lang_items,
		..
	} = collector;
	CollectionResult {
		name_env,
		lang_items,
	}
}

pub struct CollectionResult {
	pub name_env: NameEnvironment,
	pub lang_items: FxHashMap<LangItemKind, DefId>,
}

pub(crate) fn resolve_root(
	scx: &SessionCtx,
	ast: &ast::Root,
	name_env: &NameEnvironment,
) -> ResolutionResult {
	let mut resolver = Resolver::new(scx, name_env);
	visit::Visitor::visit_root(&mut resolver, ast);

	let Resolver { resolution_map, .. } = resolver;
	ResolutionResult { resolution_map }
}

pub struct ResolutionResult {
	pub resolution_map: FxHashMap<ast::NodeId, Resolution>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct DefId(u32);

impl fmt::Debug for DefId {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		// def id -> did#x
		// TODO: global def id -> did{<package id>}#x
		write!(f, "did#{}", self.0)
	}
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LangItemKind {
	AddTrait,
	AddAssignTrait,
	SubTrait,
	SubAssignTrait,
	//
	// MulTrait,
	// DivTrait,
	//
	BoolTy,
	UIntTy,
	SIntTy,
	FloatTy,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Default)]
pub struct NameEnvironment {
	pub types: FxHashMap<Symbol, DefId>,
	pub values: FxHashMap<Symbol, DefId>,
}

#[derive(Debug)]
struct Collector<'scx> {
	scx: &'scx SessionCtx,
	pub(crate) name_env: NameEnvironment,
	pub(crate) lang_items: FxHashMap<LangItemKind, DefId>,

	next_local_def_id: u32,
	node_id_to_def_id: FxHashMap<ast::NodeId, DefId>,
}

impl<'scx> Collector<'scx> {
	#[must_use]
	pub fn new(scx: &'scx SessionCtx) -> Self {
		Self {
			scx,
			name_env: NameEnvironment::default(),
			lang_items: FxHashMap::default(),
			next_local_def_id: 0,
			node_id_to_def_id: Default::default(),
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

	fn register_lang_item(&mut self, kind: LangItemKind, def_id: DefId) {
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
				let (span1, span2) = todo!("get spans for {:?} and {:?}", occupied.get(), def_id);
				let report = errors::ty::item_name_conflict(span1, span2, ns);
				self.scx.dcx().emit_build(report);
			}
		}
	}

	fn parse_lang_item(
		&mut self,
		item: &ast::Item,
		attr: &ast::Attr,
	) -> Result<LangItemKind, Diagnostic> {
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
		Ok(kind)
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
			if attr.path.is_match(&[sym::lang_item]) {
				match self.parse_lang_item(item, attr) {
					Ok(kind) => self.register_lang_item(kind, def_id),
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

			ast::ItemKind::Function(ast::Function { name, .. }) => {
				self.register_def(Namespace::Value, def_id, name);
			}
			ast::ItemKind::ForeignMod { .. } => {
				ast::visit::visit_item(self, item);
			}

			ast::ItemKind::TraitImpl { .. } => {
				// nothing to collect, type environment with collect trait
				// implementations and method resolution will select the implementation
			}
		}
	}
}

#[derive(Debug, Clone)]
pub enum Resolution {
	Def(DefId),
	Local(ast::NodeId),
}

impl Resolution {
	pub fn as_def(&self) -> Option<DefId> {
		match self {
			Self::Def(def) => todo!(),
			_ => None,
		}
	}

	pub fn as_local(&self) -> Option<hir::ExprId> {
		match self {
			Self::Local(id) => todo!(),
			_ => None,
		}
	}
}

#[derive(Debug, Clone)]
struct Resolver<'scx> {
	scx: &'scx SessionCtx,

	name_env: &'scx NameEnvironment,

	value_layers: Vec<(ValueLayerKind, FxHashMap<Symbol, ast::NodeId>)>,
	type_layers: Vec<(TypeLayerKind, FxHashMap<Symbol, ast::NodeId>)>,
	resolution_map: FxHashMap<ast::NodeId, Resolution>,
}

impl<'scx> Resolver<'scx> {
	#[must_use]
	pub fn new(scx: &'scx SessionCtx, name_env: &'scx NameEnvironment) -> Self {
		Self {
			scx,
			name_env,
			value_layers: Vec::default(),
			type_layers: Vec::default(),
			resolution_map: FxHashMap::default(),
		}
	}
}

impl Resolver<'_> {
	fn with_value_bindings(
		&mut self,
		layer_kind: ValueLayerKind,
		bindings: FxHashMap<Symbol, ast::NodeId>,
		f: impl FnOnce(&mut Self),
	) {
		self.value_layers.push((layer_kind, bindings));
		f(self);
		self.value_layers.pop();
	}

	fn with_type_bindings(
		&mut self,
		layer_kind: TypeLayerKind,
		bindings: FxHashMap<Symbol, ast::NodeId>,
		f: impl FnOnce(&mut Self),
	) {
		self.type_layers.push((layer_kind, bindings));
		f(self);
		self.type_layers.pop();
	}
}

#[derive(Debug, Clone)]
pub enum ValueLayerKind {
	Param,
	Local,
}

#[derive(Debug, Clone)]
pub enum TypeLayerKind {
	Generics,
}

impl Resolver<'_> {
	fn resolve_path(&mut self, ns: &Namespace, path: &ast::Path) {
		let path_simple = path.simple();

		let res = match ns {
			Namespace::Type => 'res: {
				for (layer_kind, bindings) in self.type_layers.iter().rev() {
					if let Some(id) = bindings.get(&path_simple.sym) {
						break 'res match layer_kind {
							TypeLayerKind::Generics => Resolution::Local(*id),
						};
					}
				}
				let Some(node_id) = self.name_env.types.get(&path_simple.sym) else {
					todo!("could not find type in scope: {path_simple:?}")
				};
				Resolution::Def(*node_id)
			}
			Namespace::Value => 'res: {
				for (layer_kind, bindings) in self.value_layers.iter().rev() {
					if let Some(id) = bindings.get(&path_simple.sym) {
						break 'res match layer_kind {
							ValueLayerKind::Param => Resolution::Local(*id),
							ValueLayerKind::Local => Resolution::Local(*id),
						};
					}
				}
				let Some(node_id) = self.name_env.values.get(&path_simple.sym) else {
					todo!("could not find value in scope: {path_simple:?}")
				};
				Resolution::Def(*node_id)
			}
		};

		let before = self.resolution_map.insert(path.id, res);
		assert!(before.is_none());
	}
}

impl visit::Visitor for Resolver<'_> {
	fn visit_item(&mut self, item: &ast::Item) {
		match &item.kind {
			ast::ItemKind::Function(ast::Function { decl, body, .. }) => {
				self.with_value_bindings(
					ValueLayerKind::Param,
					decl.params.iter().map(|p| (p.name.sym, p.id)).collect(),
					|this| {
						for param in &decl.params {
							this.visit_param(param);
						}
						if let Some(ret) = &decl.ret {
							this.visit_ty(ret);
						}
						if let Some(body) = body {
							this.visit_block(body);
						}
					},
				);
			}
			ast::ItemKind::Struct { generics, .. }
			| ast::ItemKind::Enum { generics, .. }
			| ast::ItemKind::Trait { generics, .. } => self.with_type_bindings(
				TypeLayerKind::Generics,
				generics.idents.iter().map(|g| (g.name.sym, g.id)).collect(),
				|this| ast::visit::visit_item(this, item),
			),
			_ => ast::visit::visit_item(self, item),
		}
	}

	fn visit_ty(&mut self, ty: &ast::Ty) {
		if let ast::TyKind::Path(path) = &ty.kind {
			self.resolve_path(&Namespace::Type, path);
		}
		ast::visit::visit_ty(self, ty);
	}

	fn visit_stmt(&mut self, stmt: &ast::Stmt) {
		if let ast::StmtKind::Let { name, .. } = &stmt.kind {
			let (_layer_kind, bindings) = self.value_layers.last_mut().unwrap();
			bindings.insert(name.sym, stmt.id);
		}
		ast::visit::visit_stmt(self, stmt);
	}

	fn visit_expr(&mut self, expr: &ast::Expr) {
		if let ast::ExprKind::Access { path } = &expr.kind {
			self.resolve_path(&Namespace::Value, path);
		}
		ast::visit::visit_expr(self, expr);
	}
}
