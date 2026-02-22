#[cfg(feature = "backend-cranelift")]
mod cranelift;
#[cfg(feature = "backend-llvm")]
mod llvm;

use std::path::Path;

#[cfg(feature = "backend-cranelift")]
pub use self::cranelift::Generator as CraneliftBackend;
#[cfg(feature = "backend-llvm")]
pub use self::llvm::Generator as LlvmBackend;

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

#[derive(Debug, Default)]
pub enum Linker {
	Ld,
	Lld,
	#[default]
	Wild,
}

pub trait CodeGenBackend {
	fn codegen_root(&mut self, hir: &crate::hir::Root);
}

pub trait JitBackend: CodeGenBackend {
	fn finalize(&mut self);

	fn call_main(&self);
}

pub trait ObjectBackend: CodeGenBackend {
	fn write_object(self: Box<Self>, path: &Path);
}

pub enum BackendDispatch {
	#[cfg(feature = "backend-cranelift")]
	Cranelift,
	#[cfg(feature = "backend-llvm")]
	Llvm,
}
