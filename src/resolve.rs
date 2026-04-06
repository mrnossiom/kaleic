use rustc_hash::FxHashMap;

use crate::{
	ast::{self, Visitor, visit},
	collect::{DefId, ModuleId, NameEnvironment, Namespace},
	hir,
	session::{DcxHandle, SessionCtx},
	symbols::{Symbol, kw},
};

pub(crate) fn resolve_root(scx: &SessionCtx, ast: &ast::Root) {
	let name_env = scx.name_env.borrow();
	let modules = scx.modules.borrow();

	let mut resolver = Resolver::new(scx, &name_env, &modules);
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

	name_env: &'scx NameEnvironment,
	modules: &'scx FxHashMap<(ModuleId, Symbol), ModuleId>,

	layers: Vec<(LayerKind, FxHashMap<Symbol, ast::NodeId>)>,
	resolutions: FxHashMap<ast::NodeId, PartialRes>,

	current_module: ModuleId,
}

impl<'scx> Resolver<'scx> {
	#[must_use]
	pub(crate) fn new(
		scx: &'scx SessionCtx,
		name_env: &'scx NameEnvironment,
		modules: &'scx FxHashMap<(ModuleId, Symbol), ModuleId>,
	) -> Self {
		Self {
			scx,
			name_env,
			modules,
			layers: Vec::default(),
			resolutions: FxHashMap::default(),
			current_module: ModuleId::ROOT,
		}
	}
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

		let res = if base.name.sym == kw::SelfTy {
			PartialRes::new_self(rest.len())
		} else if rest.is_empty() {
			// search layers and module ctx
			self.resolve_local(ns, &base.name)
		} else {
			// resolve all module segments then search module or relative res
			self.resolve_long_path(path)
		};

		let before = self.resolutions.insert(path.id, res);
		assert!(before.is_none());
	}

	fn resolve_local(&self, ns: Namespace, local: &ast::Ident) -> PartialRes {
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

		if let Some(local_res) = local_res {
			PartialRes::new_full(local_res)
		} else if let Some(def_res) = self.name_env[ns].get(&(self.current_module, local.sym)) {
			PartialRes::new_full(Res::Def(*def_res))
		} else {
			let report = errors::not_in_scope(ns, local.span);
			self.scx.dcx().emit_build(report);
			PartialRes::new_full(Res::Error)
		}
	}

	fn resolve_long_path(&mut self, path: &ast::Path) -> PartialRes {
		// TODO: resolve full path taking module import into account
		todo!()

		// TODO: stop and do partial res at type
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
					this.visit_items(items);
				});
			}
			ast::ItemKind::Import { tree } => {
				todo!()
				// noop?
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
