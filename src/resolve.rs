use std::collections::{HashMap, hash_map::Entry};

use crate::{
	ast, errors,
	hir::{self, Enum, Function, Struct, TypeAlias},
	session::Symbol,
	ty::{self, TyKind},
};

#[derive(Debug)]
pub enum Namespace {
	Type,
	Value,
}

#[derive(Debug, Default)]
pub struct NameEnvironment {
	pub types: HashMap<Symbol, hir::Item>,
	pub values: HashMap<Symbol, hir::Item>,
}

#[derive(Debug)]
pub struct Collector<'tcx> {
	tcx: &'tcx ty::TyCtx<'tcx>,
	pub(crate) name_env: NameEnvironment,
}

impl<'tcx> Collector<'tcx> {
	#[must_use]
	pub fn new(tcx: &'tcx ty::TyCtx) -> Self {
		Self {
			tcx,
			name_env: NameEnvironment::default(),
		}
	}
}

impl Collector<'_> {
	pub fn collect_items(&mut self, hir: &hir::Root) {
		for item in &hir.items {
			self.collect_item(item);
		}
	}

	// TODO: replace expensive clone with nodeid with quick lookup
	fn collect_item(&mut self, item: &hir::Item) {
		match &item.kind {
			hir::ItemKind::Trait { name, .. }
			| hir::ItemKind::Struct(Struct { name, .. })
			| hir::ItemKind::Enum(Enum { name, .. })
			| hir::ItemKind::TypeAlias(TypeAlias { name, .. }) => {
				match self.name_env.types.entry(name.sym) {
					Entry::Vacant(vacant) => _ = vacant.insert(item.clone()),
					Entry::Occupied(occupied) => {
						let report =
							errors::ty::item_name_conflict(occupied.get().span, name.span, "type");
						self.tcx.scx.dcx().emit_build(report);
					}
				}
			}

			hir::ItemKind::TraitImpl { .. } => todo!("idk how to classify"),

			hir::ItemKind::Function(Function { name, .. }) => {
				match self.name_env.values.entry(name.sym) {
					Entry::Vacant(vacant) => _ = vacant.insert(item.clone()),
					Entry::Occupied(occupied) => {
						let report =
							errors::ty::item_name_conflict(occupied.get().span, name.span, "value");
						self.tcx.scx.dcx().emit_build(report);
					}
				}
			}
		}
	}
}

pub struct TypeComputer<'tcx> {
	tcx: &'tcx ty::TyCtx<'tcx>,

	pub(crate) ty_env: HashMap<hir::NodeId, ty::TyKind>,
}

impl<'tcx> TypeComputer<'tcx> {
	#[must_use]
	pub fn new(tcx: &'tcx ty::TyCtx) -> Self {
		Self {
			tcx,
			ty_env: HashMap::default(),
		}
	}
}

// approaches
//
// 1. recursive bruteforce
//
// - collect all items
// - start from first collected item, resolve recursively
//
// 2. multiple passes
//
// - collect all items with their path
// - compute type dependency and sort topologically
// - resolve in order

impl TypeComputer<'_> {
	pub fn compute_env(&mut self, name_env: &NameEnvironment) {
		for (_sym, item) in &name_env.types {
			let ty = self.compute_item(item);

			let old = self.ty_env.insert(item.id, ty);
			assert!(old.is_none());
		}
		for (sym, item) in &name_env.values {
			let ty = self.compute_item(item);

			let old = self.ty_env.insert(item.id, ty);
			assert!(old.is_none());
		}
	}

	fn compute_item(&mut self, item: &hir::Item) -> TyKind {
		match &item.kind {
			hir::ItemKind::Struct(Struct {
				name,
				generics,
				fields,
			}) => {
				let struct_ = ty::Struct {
					generics: generics.clone(),
					fields: fields
						.iter()
						.map(|field| self.lower_field_def(field))
						.collect(),
				};
				TyKind::Struct(Box::new(struct_))
			}
			hir::ItemKind::Enum(Enum {
				name,
				generics,
				variants,
			}) => {
				let enum_ = ty::Enum {
					generics: generics.clone(),
					variants: variants
						.iter()
						.map(|variant| self.lower_variant(variant))
						.collect(),
				};
				TyKind::Enum(Box::new(enum_))
			}

			hir::ItemKind::Trait { name, .. } => {
				// self.environment
				// 	.values
				// 	.insert(name.sym, ty::TyKind::Fn(Box::new(decl)));
				todo!()
			}
			hir::ItemKind::TraitImpl { type_, .. } => {
				// TODO
				// self.environment.types.insert(type_.sym, v);
				todo!()
			}

			hir::ItemKind::TypeAlias(TypeAlias { name, alias }) => match &alias {
				Some(ty) => self.tcx.lower_ty(ty).as_no_infer().unwrap(),
				None => todo!(
					"error about how standalone empty type aliases are not allowed, only used in traits"
				),
			},

			hir::ItemKind::Function(Function {
				name,
				decl,
				body,
				abi,
			}) => TyKind::Fn(Box::new(self.lower_fn_decl(decl))),
		}
	}

	// TODO: not pub
	pub fn lower_fn_decl(&mut self, decl: &hir::FnDecl) -> ty::FnDecl {
		// TODO: diag no infer ty in functions
		let inputs = decl
			.inputs
			.iter()
			.map(|ast::Param { name, ty }| {
				let ty = if let Ok(ty) = self.tcx.lower_ty(ty).as_no_infer() {
					ty
				} else {
					let report = errors::ty::function_cannot_infer_signature(name.span);
					self.tcx.scx.dcx().emit_build(report);
					TyKind::Error
				};
				ty::Param { name: *name, ty }
			})
			.collect();

		let output = if let Ok(ty) = self.tcx.lower_ty(&decl.output).as_no_infer() {
			ty
		} else {
			let report = errors::ty::function_cannot_infer_signature(decl.output.span);
			self.tcx.scx.dcx().emit_build(report);
			TyKind::Error
		};
		ty::FnDecl { inputs, output }
	}

	fn lower_field_def(&mut self, hir::FieldDef { name, ty }: &hir::FieldDef) -> ty::FieldDef {
		ty::FieldDef {
			name: *name,
			ty: self.tcx.lower_ty(ty).as_no_infer().unwrap(),
		}
	}

	fn lower_variant(
		&self,
		hir::EnumVariant { name, fields, span }: &hir::EnumVariant,
	) -> ty::Variant {
		ty::Variant {
			name: *name,
			span: *span,
		}
	}
}
