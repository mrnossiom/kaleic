#[cfg(feature = "backend-cranelift")]
mod cranelift;
#[cfg(feature = "backend-llvm")]
mod llvm;

use std::path::Path;

use crate::ty::TyCtx;

#[cfg(feature = "backend-cranelift")]
pub(crate) use self::cranelift::Generator as CraneliftBackend;
#[cfg(feature = "backend-llvm")]
pub(crate) use self::llvm::Generator as LlvmBackend;

#[derive(Debug)]
pub enum Backend {
	#[cfg(feature = "backend-cranelift")]
	Cranelift,
	#[cfg(feature = "backend-llvm")]
	Llvm,
	NoBackend,
}

#[allow(
	clippy::derivable_impls,
	unreachable_code,
	reason = "depends on backend feature flags"
)]
impl Default for Backend {
	fn default() -> Self {
		#[cfg(feature = "backend-cranelift")]
		return Self::Cranelift;
		#[cfg(feature = "backend-llvm")]
		return Self::Llvm;
		Self::NoBackend
	}
}

impl Backend {
	pub(crate) fn jit_backend<'tcx>(
		&self,
		tcx: &'tcx TyCtx<'tcx>,
	) -> Option<Box<dyn JitBackend + 'tcx>> {
		match self {
			#[cfg(feature = "backend-cranelift")]
			Self::Cranelift => Some(Box::new(CraneliftBackend::new_jit(tcx))),
			#[cfg(feature = "backend-llvm")]
			Self::Llvm => Some(Box::new(LlvmBackend::new_jit(tcx))),
			Self::NoBackend => None,
		}
	}

	pub(crate) fn object_backend<'tcx>(
		&self,
		tcx: &'tcx TyCtx<'tcx>,
	) -> Option<Box<dyn ObjectBackend + 'tcx>> {
		match self {
			#[cfg(feature = "backend-cranelift")]
			Self::Cranelift => Some(Box::new(CraneliftBackend::new_object(tcx))),
			#[cfg(feature = "backend-llvm")]
			Self::Llvm => Some(Box::new(LlvmBackend::new_object(tcx))),
			Self::NoBackend => None,
		}
	}
}

#[derive(Debug, Default)]
pub enum Linker {
	Ld,
	Lld,
	#[default]
	Wild,
}

pub(crate) trait CodeGenBackend {
	fn codegen_root(&mut self, hir: &crate::hir::Root);
}

pub(crate) trait JitBackend: CodeGenBackend {
	fn finalize(&mut self);

	fn call_main(&self);
}

pub(crate) trait ObjectBackend: CodeGenBackend {
	fn write_object(self: Box<Self>, path: &Path);
}
