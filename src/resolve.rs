use std::collections::hash_map::Entry;

use rustc_hash::FxHashMap;

use crate::{
	ast, errors,
	hir::{self, Enum, Function, ItemId, Struct, TypeAlias},
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
	pub types: FxHashMap<Symbol, ItemId>,
	pub values: FxHashMap<Symbol, ItemId>,
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
	pub fn collect_root(&mut self, hir: &hir::Root) {
		let hir::Root { items } = &hir;
		for item in items {
			self.collect_item(item);
		}
	}

	// TODO: replace expensive clone with nodeid with quick lookup
	fn collect_item(&mut self, item: &hir::Item) {
		let hir::Item { kind, span, id } = &item;
		match &kind {
			hir::ItemKind::Trait { name, .. }
			| hir::ItemKind::Struct(Struct { name, .. })
			| hir::ItemKind::Enum(Enum { name, .. })
			| hir::ItemKind::TypeAlias(TypeAlias { name, .. }) => {
				match self.name_env.types.entry(name.sym) {
					Entry::Vacant(vacant) => _ = vacant.insert(item.item_id()),
					Entry::Occupied(occupied) => {
						let item_id = occupied.get();
						let report =
							errors::ty::item_name_conflict(todo!("{item_id:?}"), *span, "type");
						self.tcx.scx.dcx().emit_build(report);
					}
				}
			}

			hir::ItemKind::Function(Function { name, .. }) => {
				match self.name_env.values.entry(name.sym) {
					Entry::Vacant(vacant) => _ = vacant.insert(item.item_id()),
					Entry::Occupied(occupied) => {
						let item_id = occupied.get();
						let report =
							errors::ty::item_name_conflict(todo!("{item_id:?}"), *span, "value");
						self.tcx.scx.dcx().emit_build(report);
					}
				}
			}
			hir::ItemKind::Extern { items } => {
				for item in items {
					self.collect_item(&item.clone().into());
				}
			}

			hir::ItemKind::TraitImpl { .. } => {
				// nothing to collect, type environment with collect trait
				// implementations and method resolution will select the implementation
			}
		}
	}
}

pub struct TypeComputer<'tcx> {
	tcx: &'tcx ty::TyCtx<'tcx>,

	pub(crate) types: FxHashMap<ItemId, ty::TyKind>,
	// trait item id -> implementors
	pub(crate) trait_impls: FxHashMap<ItemId, Vec<()>>,
}

impl<'tcx> TypeComputer<'tcx> {
	#[must_use]
	pub fn new(tcx: &'tcx ty::TyCtx) -> Self {
		Self {
			tcx,
			types: FxHashMap::default(),
			trait_impls: FxHashMap::default(),
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
	pub fn compute_root(&mut self, root: &hir::Root) {
		let hir::Root { items } = &root;
		for item in items {
			self.compute_item(item);
		}
	}

	fn compute_item(&mut self, item: &hir::Item) {
		let ty = match &item.kind {
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

			hir::ItemKind::Function(Function { name, decl, body }) => {
				TyKind::Fn(Box::new(self.lower_fn_decl(decl)))
			}
			hir::ItemKind::Extern { items } => {
				for item in items {
					self.compute_item(&item.clone().into());
				}
				return;
			}
		};
		let old = self.types.insert(item.item_id(), ty);
		debug_assert!(old.is_none());
	}

	// TODO: not pub
	pub fn lower_fn_decl(&mut self, decl: &hir::FnDecl) -> ty::FnDecl {
		// TODO: diag no infer ty in functions
		let inputs = decl
			.params
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

		let output = if let Ok(ty) = self.tcx.lower_ty(&decl.ret).as_no_infer() {
			ty
		} else {
			let report = errors::ty::function_cannot_infer_signature(decl.ret.span);
			self.tcx.scx.dcx().emit_build(report);
			TyKind::Error
		};
		ty::FnDecl { inputs, output }
	}

	fn lower_field_def(&self, hir::FieldDef { name, ty }: &hir::FieldDef) -> ty::FieldDef {
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
