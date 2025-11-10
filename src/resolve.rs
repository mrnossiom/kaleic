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
pub struct NameEnvironment {
	pub types: HashMap<Symbol, hir::Item>,
	pub values: HashMap<Symbol, hir::Item>,
}

#[derive(Debug, Default)]
pub struct TyEnvironment {
	pub types: HashMap<Symbol, ty::TyKind>,
	pub values: HashMap<Symbol, ty::TyKind>,
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
			| hir::ItemKind::TypeAlias(Type { name, .. }) => match self.name_env.types.entry(name.sym) {
				Entry::Vacant(vacant) => _ = vacant.insert(item.clone()),
				Entry::Occupied(occupied) => {
					let report =
						errors::ty::item_name_conflict(occupied.get().span, name.span, "type");
					self.tcx.scx.dcx().emit_build(report);
				}
			},

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

pub struct TypeLayoutComputer<'tcx> {
	tcx: &'tcx ty::TyCtx<'tcx>,

	pub(crate) environment: TyEnvironment,
}

impl<'tcx> TypeLayoutComputer<'tcx> {
	#[must_use]
	pub fn new(tcx: &'tcx ty::TyCtx) -> Self {
		Self {
			tcx,

			environment: TyEnvironment::default(),
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

#[cfg(false)]
impl TypeLayoutComputer<'_> {
	pub fn compute_env(&mut self, name_env: &NameEnvironment) {
		// for item_sym in name_env.keys() {
		// 	self.compute_item(item_sym);
		// }
	}

	fn compute_item(&mut self, item_sym: &Symbol) {
		match &item.kind {
			hir::ItemKind::Struct(Struct {
				name,
				generics,
				fields,
			}) => {
				let struct_ = ty::Struct {
					name: *name,
					generics: generics.clone(),
					fields: fields
						.iter()
						.map(|field| self.lower_field_def(field))
						.collect(),
				};
				match self.environment.types.entry(name.sym) {
					Entry::Occupied(_) => todo!(),
					Entry::Vacant(entry) => {
						entry.insert_entry(TyKind::Struct(Box::new(struct_)));
					}
				}
			}
			hir::ItemKind::Enum(Enum {
				name,
				generics,
				variants,
			}) => {
				let enum_ = ty::Enum {
					name: *name,
					generics: generics.clone(),
					variants: variants
						.iter()
						.map(|variant| self.lower_variant(variant))
						.collect(),
				};
				match self.environment.types.entry(name.sym) {
					Entry::Occupied(_) => todo!("type already declared {:?}", name.sym),
					Entry::Vacant(entry) => {
						entry.insert_entry(TyKind::Enum(Box::new(enum_)));
					}
				}
			}

			hir::ItemKind::Trait { name, .. } => {
				// self.environment
				// 	.values
				// 	.insert(name.sym, ty::TyKind::Fn(Box::new(decl)));
			}
			hir::ItemKind::TraitImpl { type_, .. } => {
				// TODO
				// self.environment.types.insert(type_.sym, v);
			}

			hir::ItemKind::TypeAlias(Type(name, alias)) => match &alias {
				Some(ty) => {
					let v = self.lower_ty(ty).as_no_infer().unwrap();
					self.environment.types.insert(name.sym, v);
				}
				None => {
					todo!()
				}
			},

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
		}
	}

	fn lower_ty(&mut self, ty: &ast::Ty) -> TyKind<Infer> {
		match &ty.kind {
			ast::TyKind::Path(path) => self.lower_path_ty(path),
			ast::TyKind::Pointer(ty) => TyKind::Pointer(Box::new(self.lower_ty(ty))),
			ast::TyKind::Unit => TyKind::Primitive(PrimitiveKind::Void),
			ast::TyKind::Infer => TyKind::Infer(self.tcx.next_infer_tag(), Infer::Explicit),
		}
	}

	// TODO: not pub
	pub fn lower_fn_decl(&mut self, decl: &hir::FnDecl) -> ty::FnDecl {
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

	fn lower_path_ty(&mut self, path: &ast::Path) -> TyKind<Infer> {
		// TODO ensure path length is 1
		let primitive = path.segments.first().and_then(|seg0| {
			let primitive = match self.tcx.scx.symbols.resolve(seg0.sym).as_str() {
				"_" => TyKind::Infer(self.tcx.next_infer_tag(), Infer::Explicit),

				"void" => TyKind::Primitive(PrimitiveKind::Void),
				"never" => TyKind::Primitive(PrimitiveKind::Never),

				"bool" => TyKind::Primitive(PrimitiveKind::Bool),
				"uint" => TyKind::Primitive(PrimitiveKind::UnsignedInt),
				"sint" => TyKind::Primitive(PrimitiveKind::SignedInt),
				"float" => TyKind::Primitive(PrimitiveKind::Float),

				"str" => TyKind::Primitive(PrimitiveKind::Str),
				_ => return None,
			};
			Some(primitive)
		});

		if let Some(primitive) = primitive {
			primitive
		} else {
			let item_map = self.tcx.name_env.borrow();
			match item_map.as_ref().unwrap().get(&path.segments[0].sym) {
				Some(item) => {
					self.compute_item(item);
					self.environment.types[&path.segments[0].sym]
						.clone()
						.as_infer()
				}
				None => {
					panic!("item {:?} doesn't exist", path.segments[0].sym)
				}
			}
		}
	}

	fn lower_field_def(&mut self, hir::FieldDef { name, ty }: &hir::FieldDef) -> ty::FieldDef {
		ty::FieldDef {
			name: *name,
			ty: self.lower_ty(ty).as_no_infer().unwrap(),
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
