use crate::{
	hir::{self, Enum, ItemKind, Struct, Type},
	ty,
};

pub struct TypeCheck<'tcx> {
	tcx: &'tcx ty::TyCtx<'tcx>,
}

impl<'tcx> TypeCheck<'tcx> {
	pub fn new(tcx: &'tcx ty::TyCtx) -> Self {
		Self { tcx }
	}

	pub fn typeck(&self, hir: &hir::Root) {
		for item in &hir.items {
			self.typeck_item(item);
		}
	}

	fn typeck_item(&self, item: &hir::Item) {
		match &item.kind {
			// TODO
			ItemKind::Struct(Struct {
				name,
				generics,
				fields,
			}) => {}
			ItemKind::Enum(Enum {
				name,
				generics,
				variants,
			}) => {}
			ItemKind::TypeAlias(Type { name, alias }) => {}
			ItemKind::Trait {
				name,
				generics,
				members,
			} => {}
			ItemKind::TraitImpl {
				type_,
				trait_,
				members,
			} => {}

			ItemKind::Function(func) => {}
		}
	}
}

struct FunctionTck {}
