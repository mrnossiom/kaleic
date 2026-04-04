use rustc_hash::FxHashMap;

use crate::{
	ast::{self, Visitor, visit},
	collect::{DefId, NameEnvironment, Namespace},
	hir,
	session::{DcxHandle, SessionCtx},
	symbols::{Symbol, sym},
};

pub(crate) fn resolve_root(scx: &SessionCtx, ast: &ast::Root) {
	let name_env = scx.name_env.borrow();

	let mut resolver = Resolver::new(scx, &name_env);
	resolver.visit_root(ast);

	let Resolver { resolution_map, .. } = resolver;

	// TODO: register resolution artefact
	scx.resolution_map.put(resolution_map);
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
			sym::never_ty => Some(Self::Type(TypeLangItem::Never)),
			sym::bool_ty => Some(Self::Type(TypeLangItem::Bool)),
			sym::uint_ty => Some(Self::Type(TypeLangItem::UInt)),
			sym::sint_ty => Some(Self::Type(TypeLangItem::SInt)),
			sym::float_ty => Some(Self::Type(TypeLangItem::Float)),
			_ => None,
		}
	}
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Resolution<LocalId = hir::NodeId> {
	Def(DefId),
	Local(LocalId),
	Error,
}

pub(crate) type EarlyResolution = Resolution<ast::NodeId>;

impl<LocalId> Resolution<LocalId> {
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

	pub(crate) fn map_local<T>(self, f: impl FnOnce(LocalId) -> T) -> Resolution<T> {
		match self {
			Self::Local(local) => Resolution::Local(f(local)),
			Self::Def(def) => Resolution::Def(def),
			Self::Error => Resolution::Error,
		}
	}
}

#[derive(Debug, Clone)]
struct Resolver<'scx> {
	scx: &'scx SessionCtx,

	name_env: &'scx NameEnvironment,

	value_layers: Vec<(ValueLayerKind, FxHashMap<Symbol, ast::NodeId>)>,
	type_layers: Vec<(TypeLayerKind, FxHashMap<Symbol, ast::NodeId>)>,
	resolution_map: FxHashMap<ast::NodeId, EarlyResolution>,
}

impl<'scx> Resolver<'scx> {
	#[must_use]
	pub(crate) fn new(scx: &'scx SessionCtx, name_env: &'scx NameEnvironment) -> Self {
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
pub(crate) enum ValueLayerKind {
	Param,
	Local,
}

#[derive(Debug, Clone)]
pub(crate) enum TypeLayerKind {
	Generics,
}

impl Resolver<'_> {
	fn resolve_path(&mut self, ns: Namespace, path: &ast::Path) {
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
				if let Some(node_id) = self.name_env.types.get(&path_simple.sym) {
					Resolution::Def(*node_id)
				} else {
					let report = errors::type_not_in_scope(path.span);
					self.scx.dcx().emit_build(report);
					Resolution::Error
				}
			}
			Namespace::Value => 'res: {
				for (layer_kind, bindings) in self.value_layers.iter().rev() {
					if let Some(id) = bindings.get(&path_simple.sym) {
						break 'res match layer_kind {
							ValueLayerKind::Param | ValueLayerKind::Local => Resolution::Local(*id),
						};
					}
				}
				if let Some(node_id) = self.name_env.values.get(&path_simple.sym) {
					Resolution::Def(*node_id)
				} else {
					let report = errors::value_not_in_scope(path.span);
					self.scx.dcx().emit_build(report);
					Resolution::Error
				}
			}
		};

		let before = self.resolution_map.insert(path.id, res);
		assert!(before.is_none());
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
				todo!()
				// self.with_resolution_layer()
			}
			ast::ItemKind::Function(ast::Function {
				decl,
				generics,
				body,
				..
			}) => {
				self.with_type_bindings(
					TypeLayerKind::Generics,
					generics.idents.iter().map(|g| (g.name.sym, g.id)).collect(),
					|this| {
						this.with_value_bindings(
							ValueLayerKind::Param,
							decl.params.iter().map(|p| (p.name.sym, p.id)).collect(),
							|this| visit::visit_item(this, item),
						);
					},
				);
			}
			ast::ItemKind::Struct { generics, .. }
			| ast::ItemKind::Enum { generics, .. }
			| ast::ItemKind::Trait { generics, .. } => self.with_type_bindings(
				TypeLayerKind::Generics,
				generics.idents.iter().map(|g| (g.name.sym, g.id)).collect(),
				|this| visit::visit_item(this, item),
			),

			ast::ItemKind::ExternUse { .. }
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
			let (_layer_kind, bindings) = self.value_layers.last_mut().unwrap();
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

	use crate::session::{Report, ReportBuilder, Span};

	pub fn type_not_in_scope(path_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, path_span)
			.with_message("type is invalid")
			.with_label(Label::new(path_span).with_message("type is not in scope"))
	}

	pub fn value_not_in_scope(path_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, path_span)
			.with_message("value is invalid")
			.with_label(Label::new(path_span).with_message("value is not in scope"))
	}
}
