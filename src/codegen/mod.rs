#[cfg(feature = "cranelift")]
mod cranelift;
#[cfg(feature = "llvm")]
mod llvm;

#[cfg(feature = "cranelift")]
pub use self::cranelift::Generator as CraneliftBackend;
#[cfg(feature = "llvm")]
pub use self::llvm::Generator as LlvmBackend;

#[derive(Debug)]
pub enum Backend {
	#[cfg(feature = "cranelift")]
	Cranelift,
	#[cfg(feature = "llvm")]
	Llvm,
	NoBackend,
}

impl Default for Backend {
	fn default() -> Self {
		#[cfg(feature = "cranelift")]
		return Self::Cranelift;
		#[cfg(feature = "llvm")]
		return Self::Llvm;
		Self::NoBackend
	}
}

pub trait CodeGenBackend {
	fn codegen_root(&mut self, hir: &crate::hir::Root);
}

pub trait JitBackend: CodeGenBackend {
	fn call_main(&mut self);
}

pub trait ObjectBackend: CodeGenBackend {
	// TODO: change to common object
	#[cfg(feature = "cranelift")]
	fn get_object(self) -> cranelift_object::ObjectProduct;
}
