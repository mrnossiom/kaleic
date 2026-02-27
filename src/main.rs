use std::path::PathBuf;

use clap::Parser;
use kaleic::{driver, session::SessionCtx};

mod options {
	use clap::ValueEnum;
	use kaleic::{codegen, session};

	#[derive(Debug, Clone, ValueEnum)]
	pub enum Backend {
		#[cfg(feature = "backend-cranelift")]
		Cranelift,
		#[cfg(feature = "backend-llvm")]
		Llvm,
	}

	impl From<&Backend> for codegen::Backend {
		fn from(val: &Backend) -> Self {
			match val {
				#[cfg(feature = "backend-cranelift")]
				Backend::Cranelift => Self::Cranelift,
				#[cfg(feature = "backend-llvm")]
				Backend::Llvm => Self::Llvm,
			}
		}
	}

	#[derive(Debug, Clone, ValueEnum)]
	pub enum Linker {
		Ld,
		Lld,
		Wild,
	}

	impl From<&Linker> for codegen::Linker {
		fn from(val: &Linker) -> Self {
			match val {
				Linker::Ld => Self::Ld,
				Linker::Lld => Self::Lld,
				Linker::Wild => Self::Wild,
			}
		}
	}

	#[derive(Debug, Clone, ValueEnum)]
	pub enum PrintKind {
		Ast,
		AstPretty,
		Hir,
		HirPretty,
		BackendIr,
		Items,
		Env,
	}

	impl From<PrintKind> for session::PrintKind {
		fn from(val: PrintKind) -> Self {
			match val {
				PrintKind::Ast => Self::Ast,
				PrintKind::AstPretty => Self::AstPretty,
				PrintKind::Hir => Self::HigherIr,
				PrintKind::HirPretty => Self::HigherIrPretty,
				PrintKind::BackendIr => Self::BackendIr,
				PrintKind::Items => Self::CollectedItems,
				PrintKind::Env => Self::TypeEnvironment,
			}
		}
	}
}

// this has no default option, default options are in the options struct
#[derive(clap::Parser)]
struct Args {
	pub inputs: Vec<PathBuf>,
	#[clap(long)]
	pub no_std: bool,

	#[clap(long)]
	pub jit: bool,

	#[clap(long)]
	pub backend: Option<options::Backend>,
	#[clap(long)]
	pub linker: Option<options::Linker>,

	#[clap(long)]
	pub output: Option<PathBuf>,
	#[clap(long)]
	pub print: Vec<options::PrintKind>,
}

fn main() {
	let args = Args::parse();

	let mut scx = SessionCtx::default();

	let SessionCtx { options, .. } = &mut scx;
	options.inputs = args.inputs;
	if !args.no_std {
		options.inputs.extend([
			"std/rt.kl".into(),
			"std/libc.kl".into(),
			// "std/arith.kl".into(),
		]);
	}

	options.jit = args.jit;
	args.backend.inspect(|value| options.backend = value.into());
	args.linker.inspect(|value| options.linker = value.into());
	args.output.inspect(|value| options.output = value.into());
	options.print.extend(args.print.into_iter().map(Into::into));

	driver::pipeline(&scx);
}
