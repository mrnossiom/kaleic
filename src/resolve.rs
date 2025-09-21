use std::collections::{HashMap, hash_map::Entry};

use crate::{
	ast, errors,
	hir::{self, Enum, Function, Struct, Type},
	session::Symbol,
	ty::{self, Infer, PrimitiveKind, TyKind},
};

#[derive(Debug)]
pub enum Namespace {
	Type,
	Value,
}

#[derive(Debug, Default)]
pub struct Environment {
	pub types: HashMap<Symbol, TypeValueKind>,
	pub values: HashMap<Symbol, ty::TyKind>,
}

#[derive(Debug)]
pub enum TypeValueKind {
	Trait(()),
	Type(ty::TyKind),
}

#[derive(Debug)]
pub struct Collector<'tcx> {
	tcx: &'tcx ty::TyCtx<'tcx>,

	// how to recover from items with the same name
	pub(crate) item_map: HashMap<Symbol, hir::Item>,
}

impl<'tcx> Collector<'tcx> {
	#[must_use]
	pub fn new(tcx: &'tcx ty::TyCtx) -> Self {
		Self {
			tcx,
			item_map: HashMap::default(),
		}
	}
}

impl Collector<'_> {
	pub fn collect_items(&mut self, hir: &hir::Root) {
		for item in &hir.items {
			self.collect_item(item);
		}
	}

	fn collect_item(&mut self, item: &hir::Item) {
		let name = match &item.kind {
			hir::ItemKind::TraitImpl { .. } => todo!("idk"),

			hir::ItemKind::Function(Function { name, .. })
			| hir::ItemKind::Struct(Struct { name, .. })
			| hir::ItemKind::Enum(Enum { name, .. })
			| hir::ItemKind::Trait { name, .. }
			| hir::ItemKind::Type(Type(name, _)) => Some(*name),
		};

		if let Some(name) = name {
			// TODO: replace expensive clone with nodeid with quick lookup
			match self.item_map.entry(name.sym) {
				Entry::Occupied(occupied) => {
					let report = errors::ty::item_name_conflict(occupied.get().span, name.span);
					self.tcx.scx.dcx().emit_build(report);
				}
				Entry::Vacant(vacant) => {
					vacant.insert(item.clone());
				}
			}
		}
	}
}

pub struct EnvironmentComputer<'tcx> {
	tcx: &'tcx ty::TyCtx<'tcx>,

	pub(crate) environment: Environment,
}

impl<'tcx> EnvironmentComputer<'tcx> {
	#[must_use]
	pub fn new(tcx: &'tcx ty::TyCtx) -> Self {
		Self {
			tcx,

			environment: Environment::default(),
		}
	}
}

impl EnvironmentComputer<'_> {
	pub fn compute_env(&mut self, hir: &hir::Root) {
		for item in &hir.items {
			self.compute_item(item);
		}
	}

	fn compute_item(&mut self, item: &hir::Item) {
		match &item.kind {
			hir::ItemKind::Function(Function {
				name,
				decl,
				body,
				abi,
			}) => {
				let decl = self.lower_fn_decl(decl);
				self.environment
					.values
					.insert(name.sym, ty::TyKind::Fn(Box::new(decl)));
			}
			hir::ItemKind::TraitImpl { .. } => todo!(),

			hir::ItemKind::Struct(Struct {
				name,
				generics,
				fields,
			}) => {
				let struct_ = Struct {
					name: *name,
					generics: generics.clone(),
					fields: fields.clone(),
				};
				self.environment
					.types
					.insert(name.sym, TypeValueKind::Type(todo!()));
			}
			hir::ItemKind::Enum(Enum {
				name,
				generics,
				variants,
			}) => {}
			hir::ItemKind::Trait { .. } => {}

			hir::ItemKind::Type(Type(name, alias)) => {
				self.environment
					.types
					.insert(name.sym, TypeValueKind::Type(todo!()));
			}
		}
	}

	fn lower_ty(&self, ty: &ast::Ty) -> TyKind<Infer> {
		match &ty.kind {
			ast::TyKind::Path(path) => self.lower_path_ty(path),
			ast::TyKind::Pointer(ty) => TyKind::Pointer(Box::new(self.lower_ty(ty))),
			ast::TyKind::Unit => TyKind::Primitive(PrimitiveKind::Void),
			ast::TyKind::Infer => TyKind::Infer(self.tcx.next_infer_tag(), Infer::Explicit),
		}
	}

	// TODO: not pub
	pub fn lower_fn_decl(&self, decl: &hir::FnDecl) -> ty::FnDecl {
		// TODO: diag no infer ty in functions
		let inputs = decl
			.inputs
			.iter()
			.map(|ast::Param { name, ty }| {
				let ty = if let Ok(ty) = self.lower_ty(ty).as_no_infer() {
					ty
				} else {
					let report = errors::ty::function_cannot_infer_signature(name.span);
					self.tcx.scx.dcx().emit_build(report);
					TyKind::Error
				};
				ty::Param { name: *name, ty }
			})
			.collect();

		let output = if let Ok(ty) = self.lower_ty(&decl.output).as_no_infer() {
			ty
		} else {
			let report = errors::ty::function_cannot_infer_signature(decl.output.span);
			self.tcx.scx.dcx().emit_build(report);
			TyKind::Error
		};
		ty::FnDecl { inputs, output }
	}

	fn lower_path_ty(&self, path: &ast::Path) -> TyKind<Infer> {
		// TODO: remove these constraints
		assert_eq!(path.segments.len(), 1);
		assert_eq!(path.generics.len(), 0);

		let path = path.segments[0];
		match self.tcx.scx.symbols.resolve(path.sym).as_str() {
			"_" => TyKind::Infer(self.tcx.next_infer_tag(), Infer::Explicit),

			"void" => TyKind::Primitive(PrimitiveKind::Void),
			"never" => TyKind::Primitive(PrimitiveKind::Never),

			"bool" => TyKind::Primitive(PrimitiveKind::Bool),
			"uint" => TyKind::Primitive(PrimitiveKind::UnsignedInt),
			"sint" => TyKind::Primitive(PrimitiveKind::SignedInt),
			"float" => TyKind::Primitive(PrimitiveKind::Float),

			"str" => TyKind::Primitive(PrimitiveKind::Str),

			_ => {
				let report = errors::ty::type_unknown(path.span);
				self.tcx.scx.dcx().emit_build(report);
				TyKind::Error
			}
		}
	}
}
