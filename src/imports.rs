use rustc_hash::FxHashMap;

use crate::{
	ast::{self, ImportTreeKind, Visitor, visit},
	collect::{DefId, ModuleId, PerNamespace},
	session::SessionCtx,
	symbols::{Symbol, kw},
};

pub(crate) fn resolve_imports_root(scx: &SessionCtx, ast: &ast::Root) {
	let name_env = scx.name_env.borrow();
	let modules = scx.modules.borrow();

	let mut resolver = ImportResolver {
		scx,

		name_env: &name_env,
		modules: &modules,

		import_bundles: FxHashMap::default(),

		current_module: ModuleId::ROOT,
		current_imports: Vec::default(),
	};
	resolver.visit_root(ast);

	let ImportResolver { import_bundles, .. } = resolver;

	scx.import_bundles.put(import_bundles);
}

#[derive(Debug, Default)]
pub(crate) struct ImportBundle {
	pub(crate) items: FxHashMap<Symbol, PerNamespace<Option<DefId>>>,
	pub(crate) modules: FxHashMap<Symbol, ModuleId>,
	/// The list of modules to search if not found, should all be searched at once
	pub(crate) globs: Vec<ModuleId>,
}

#[derive(Debug)]
pub(crate) enum ImportKind {
	/// A direct reference to an item
	Item { parent_id: ModuleId, sym: Symbol },
	/// A direct reference to a module
	Module { module_id: ModuleId, sym: Symbol },
	/// Bring all items of the referenced module into the scope
	// TODO: also import sub-modules
	Glob { module_id: ModuleId },
}

#[derive(Debug)]
struct ImportResolver<'scx> {
	scx: &'scx SessionCtx,

	name_env: &'scx PerNamespace<FxHashMap<(ModuleId, Symbol), DefId>>,
	modules: &'scx FxHashMap<(ModuleId, Symbol), ModuleId>,

	import_bundles: FxHashMap<ModuleId, ImportBundle>,

	current_module: ModuleId,
	current_imports: Vec<ImportKind>,
}

impl ImportResolver<'_> {
	fn with_module(&mut self, name: Symbol, f: impl FnOnce(&mut Self)) {
		let parent = self.current_module;
		self.current_module = self.modules[&(parent, name)];
		f(self);
		self.current_module = parent;
	}

	fn resolve_imports(&mut self, tree: &ast::ImportTree) {
		enum ImportBase {
			Package,
			Module,
			// Parent,
			Extern(Symbol),
		}

		let (base, tree) = match &tree.kind {
			ImportTreeKind::Module(name, tree) => {
				let base = match name.sym {
					kw::Crate => ImportBase::Package,
					kw::SelfValue => ImportBase::Module,
					// kw::Super => ImportBase::Parent,
					extern_name => ImportBase::Extern(extern_name),
				};
				(base, tree)
			}
			ImportTreeKind::Branches(..) | ImportTreeKind::Item(..) | ImportTreeKind::Glob => {
				todo!("invalid base")
			}
		};

		let module_id = match base {
			ImportBase::Package => ModuleId::ROOT,
			ImportBase::Module => self.current_module,
			ImportBase::Extern(sym) => {
				todo!("packages: {:?} {:?}", sym, tree.span)
			}
		};

		self.walk_import_tree(module_id, tree);
	}

	fn walk_import_tree(&mut self, parent_id: ModuleId, tree: &ast::ImportTree) {
		match &tree.kind {
			ImportTreeKind::Branches(branches) => {
				for branch in branches {
					self.walk_import_tree(parent_id, branch);
				}
			}
			ImportTreeKind::Module(name, tree) => {
				let Some(module_id) = self.modules.get(&(parent_id, name.sym)) else {
					todo!("module doesn't exist {:?}", name)
				};
				self.walk_import_tree(*module_id, tree);
			}
			ImportTreeKind::Item(name) => {
				if let Some(module_id) = self.modules.get(&(parent_id, name.sym)) {
					self.current_imports.push(ImportKind::Module {
						module_id: *module_id,
						sym: name.sym,
					});
				} else {
					self.current_imports.push(ImportKind::Item {
						parent_id,
						sym: name.sym,
					});
				}
			}
			ImportTreeKind::Glob => {
				self.current_imports.push(ImportKind::Glob {
					module_id: parent_id,
				});
			}
		}
	}

	fn compute_import_bundle(&mut self) {
		let mut bundle = ImportBundle::default();
		for import_kind in self.current_imports.drain(..) {
			match import_kind {
				ImportKind::Item { parent_id, sym } => {
					let bindings = self
						.name_env
						.map_all_ns(|this| this.get(&(parent_id, sym)).cloned());
					let old = bundle.items.insert(sym, bindings);
					assert!(old.is_none());
				}
				ImportKind::Module { module_id, sym } => {
					let old = bundle.modules.insert(sym, module_id);
					assert!(old.is_none());
				}

				ImportKind::Glob { module_id } => bundle.globs.push(module_id),
			}
		}
		self.import_bundles.insert(self.current_module, bundle);
	}
}

impl visit::Visitor for ImportResolver<'_> {
	fn visit_root(&mut self, ast::Root { attrs: _, items }: &ast::Root) {
		self.visit_items(items);
		self.compute_import_bundle();
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
		match kind {
			ast::ItemKind::Import { tree } => self.resolve_imports(tree),

			ast::ItemKind::Module { name, .. } => {
				self.with_module(name.sym, |this| {
					visit::visit_item(this, item);
					this.compute_import_bundle();
				});
			}

			ast::ItemKind::ExternImport { .. }
			| ast::ItemKind::Function(..)
			| ast::ItemKind::TypeAlias(..)
			| ast::ItemKind::Struct { .. }
			| ast::ItemKind::Enum { .. }
			| ast::ItemKind::Trait { .. }
			| ast::ItemKind::TraitImpl { .. }
			| ast::ItemKind::ForeignMod { .. } => {}
		}
	}
}
