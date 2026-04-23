use rustc_hash::FxHashMap;

use crate::{
	ast::{self, PathSegment, Visitor, visit},
	collect::{self, DefId, ModuleId, Namespace, PerNamespace},
	hir,
	imports::ImportBundle,
	session::{DcxHandle, SessionCtx},
	symbols::{Symbol, kw},
};

pub(crate) fn resolve_root(scx: &SessionCtx, ast: &ast::Root) {
	let name_env = scx.name_env.borrow();
	let modules = scx.modules.borrow();
	let import_bundles = scx.import_bundles.borrow();

	let mut resolver = Resolver {
		scx,
		name_env: &name_env,
		modules: &modules,
		import_bundles: &import_bundles,
		layers: Vec::default(),
		resolutions: FxHashMap::default(),
		current_module: ModuleId::ROOT,
	};
	resolver.visit_root(ast);

	let Resolver { resolutions, .. } = resolver;

	// TODO: register resolution artefact
	scx.resolutions.put(resolutions);
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Res<LocalId = hir::NodeId> {
	Def(DefId),
	Local(LocalId),
	SelfTy,
	Error,
}

pub(crate) type EarlyRes = Res<ast::NodeId>;

impl<LocalId> Res<LocalId> {
	pub(crate) fn into_def(self) -> Option<DefId> {
		match self {
			Self::Def(def_id) => Some(def_id),
			_ => None,
		}
	}

	pub(crate) fn into_local(self) -> Option<LocalId> {
		match self {
			Self::Local(node_id) => Some(node_id),
			_ => None,
		}
	}

	pub(crate) fn map_local<T>(self, f: impl FnOnce(LocalId) -> T) -> Res<T> {
		match self {
			Self::Local(local) => Res::Local(f(local)),
			Self::Def(def) => Res::Def(def),
			Self::SelfTy => Res::SelfTy,
			Self::Error => Res::Error,
		}
	}
}

// used for paths in types, e.g. assoc items
#[derive(Debug, Clone)]
pub(crate) struct PartialRes {
	pub(crate) res: EarlyRes,
	pub(crate) unresolved_segments: usize,
}

impl PartialRes {
	fn new_full(res: EarlyRes) -> Self {
		Self {
			res,
			unresolved_segments: 0,
		}
	}

	fn new_self(unresolved_segments: usize) -> Self {
		Self {
			res: Res::SelfTy,
			unresolved_segments,
		}
	}
}

#[derive(Debug, Clone)]
struct Resolver<'scx> {
	scx: &'scx SessionCtx,

	name_env: &'scx PerNamespace<FxHashMap<(ModuleId, Symbol), DefId>>,
	modules: &'scx FxHashMap<(ModuleId, Symbol), ModuleId>,
	import_bundles: &'scx FxHashMap<ModuleId, ImportBundle>,

	layers: Vec<(LayerKind, FxHashMap<Symbol, ast::NodeId>)>,
	resolutions: FxHashMap<ast::NodeId, PartialRes>,

	current_module: ModuleId,
}

impl Resolver<'_> {
	fn with_layer(
		&mut self,
		layer_kind: LayerKind,
		bindings: FxHashMap<Symbol, ast::NodeId>,
		f: impl FnOnce(&mut Self),
	) {
		self.layers.push((layer_kind, bindings));
		f(self);
		self.layers.pop().unwrap();
	}

	fn with_module(&mut self, name: Symbol, f: impl FnOnce(&mut Self)) {
		let parent = self.current_module;
		self.current_module = self.modules[&(parent, name)];
		f(self);
		self.current_module = parent;
	}
}

#[derive(Debug, Clone)]
pub(crate) enum LayerKind {
	Module,

	// value ns
	Params,
	Locals,

	// type ns
	Generics,
}

impl LayerKind {
	fn is_namespace(&self, ns: Namespace) -> bool {
		match self {
			Self::Module => true,
			Self::Params | Self::Locals => ns == Namespace::Value,
			Self::Generics => ns == Namespace::Type,
		}
	}
}

impl Resolver<'_> {
	fn resolve_path(&mut self, ns: Namespace, path: &ast::Path) {
		assert!(path.segments.iter().all(|s| s.generics.params.is_empty()));

		let (base, rest) = path.segments.split_first().unwrap();

		let partial_res = 'res: {
			if base.name.sym == kw::SelfTy {
				break 'res PartialRes::new_self(rest.len());
			}

			// search local layers, if applicable
			if rest.is_empty() // only one segment path
				&& let Some(res) = self.resolve_local(ns, &base.name)
			{
				break 'res res;
			}

			// resolve all module segments then search module or relative res
			if let Some(res) = self.resolve_def(ns, path) {
				break 'res res;
			}

			let report = errors::not_in_scope(ns, path.span);
			self.scx.dcx().emit_build(report);
			PartialRes::new_full(Res::Error)
		};

		let before = self.resolutions.insert(path.id, partial_res);
		assert!(before.is_none());
	}

	fn resolve_local(&self, ns: Namespace, local: &ast::Ident) -> Option<PartialRes> {
		let relevant_layers = self
			.layers
			.iter()
			.rev()
			.filter(|(layer, _)| layer.is_namespace(ns));

		let mut local_res = None;
		for (layer_kind, bindings) in relevant_layers {
			if let Some(id) = bindings.get(&local.sym) {
				let res = match layer_kind {
					LayerKind::Module => break,
					LayerKind::Locals | LayerKind::Params | LayerKind::Generics => {
						local_res.replace(Res::Local(*id));
						break;
					}
				};
			}
		}

		local_res.map(PartialRes::new_full)
	}

	// resolve_long_path
	fn resolve_def(&mut self, ns: Namespace, path: &ast::Path) -> Option<PartialRes> {
		let bundle = self.import_bundles.get(&self.current_module).unwrap();
		let mut search_root = self.current_module;

		let mut segments_iter = path.segments.iter().enumerate().peekable();

		// resolve the maximum amount of modules
		while let Some((i, PathSegment { name, .. })) = segments_iter.peek() {
			if *i == 0
				&& let Some(module_id) = bundle.modules.get(&name.sym)
			{
				search_root = *module_id;
				segments_iter.next();
				continue;
			}

			if let Some(module_id) = self.modules.get(&(search_root, name.sym)) {
				search_root = *module_id;
				segments_iter.next();
			} else {
				break;
			}
		}

		let (i, segment) = segments_iter.next().unwrap();

		if let Some(def_res) = self.name_env[ns].get(&(search_root, segment.name.sym)) {
			return Some(PartialRes::new_full(Res::Def(*def_res)));
		}

		// if no module segments were resolved, we can search for globs and precise imports
		if i == 0 {
			if let Some(bindings) = bundle.items.get(&segment.name.sym)
				&& let Some(def_id) = bindings[ns]
			{
				return Some(PartialRes::new_full(Res::Def(def_id)));
			}

			let glob_candidates = bundle
				.globs
				.iter()
				.filter_map(|module_id| self.name_env[ns].get(&(*module_id, segment.name.sym)))
				.collect::<Vec<_>>();

			match glob_candidates.as_slice() {
				[] => {}
				[single] => return Some(PartialRes::new_full(Res::Def(**single))),
				[..] => todo!("multiple glob imports candidate  for {:?}", path),
			}
		}

		// TODO: module system is bad, revamp for better name clash check,
		//       ambig import check and uniformize the whole to no have 100 edgecases
		//       e.g. how do we handle enum constructors `my::path::Enum::Variant`

		None
	}
}

impl visit::Visitor for Resolver<'_> {
	fn visit_item(
		&mut self,
		item @ ast::Item {
			attrs,
			kind,
			span,
			id,
		}: &ast::Item,
	) {
		match kind {
			ast::ItemKind::Module {
				name,
				items,
				inline,
			} => {
				self.with_module(name.sym, |this| {
					this.with_layer(LayerKind::Module, FxHashMap::default(), |this| {
						this.visit_items(items);
					});
				});
			}
			ast::ItemKind::Function(ast::Function {
				decl,
				generics,
				body,
				..
			}) => {
				self.with_layer(
					LayerKind::Generics,
					generics.idents.iter().map(|g| (g.name.sym, g.id)).collect(),
					|this| {
						this.with_layer(
							LayerKind::Params,
							decl.params.iter().map(|p| (p.name.sym, p.id)).collect(),
							|this| visit::visit_item(this, item),
						);
					},
				);
			}
			ast::ItemKind::Struct { generics, .. }
			| ast::ItemKind::Enum { generics, .. }
			| ast::ItemKind::Trait { generics, .. } => self.with_layer(
				LayerKind::Generics,
				generics.idents.iter().map(|g| (g.name.sym, g.id)).collect(),
				|this| visit::visit_item(this, item),
			),

			ast::ItemKind::ExternImport { .. }
			| ast::ItemKind::TypeAlias(..)
			| ast::ItemKind::TraitImpl { .. }
			| ast::ItemKind::ForeignMod { .. } => visit::visit_item(self, item),

			ast::ItemKind::Import { .. } => {}
		}
	}

	fn visit_ty(&mut self, ty @ ast::Ty { kind, span }: &ast::Ty) {
		if let ast::TyKind::Path(path) = kind {
			self.resolve_path(Namespace::Type, path);
		}
		visit::visit_ty(self, ty);
	}

	fn visit_stmt(&mut self, stmt @ ast::Stmt { kind, span, id }: &ast::Stmt) {
		if let ast::StmtKind::Let { name, .. } = kind {
			// TODO
			let (_layer_kind, bindings) = self.layers.last_mut().unwrap();
			bindings.insert(name.sym, *id);
		}
		visit::visit_stmt(self, stmt);
	}

	fn visit_expr(
		&mut self,
		expr @ ast::Expr {
			attrs,
			kind,
			span,
			id,
		}: &ast::Expr,
	) {
		if let ast::ExprKind::Access { path } = kind {
			self.resolve_path(Namespace::Value, path);
		}
		visit::visit_expr(self, expr);
	}

	fn visit_attr(&mut self, attr: &ast::Attr) {}
}

mod errors {
	use ariadne::{Label, ReportKind};

	use crate::{
		collect::Namespace,
		session::{Report, ReportBuilder, Span},
	};

	pub fn not_in_scope(ns: Namespace, path_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, path_span)
			.with_message(format!("{ns} is invalid"))
			.with_label(Label::new(path_span).with_message(format!("{ns} is not in scope")))
	}
}
