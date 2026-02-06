use std::path::PathBuf;

use clap::Parser;
use kaleic::{driver, session::SessionCtx};
use tracing_subscriber::{EnvFilter, FmtSubscriber, fmt::time};

mod options {
	use clap::ValueEnum;
	use kaleic::{codegen, session};

	#[derive(Debug, Clone, ValueEnum)]
	pub enum Backend {
		#[cfg(feature = "cranelift")]
		Cranelift,
		#[cfg(feature = "llvm")]
		Llvm,
	}

	impl From<Backend> for codegen::Backend {
		fn from(val: Backend) -> Self {
			match val {
				#[cfg(feature = "cranelift")]
				Backend::Cranelift => Self::Cranelift,
				#[cfg(feature = "llvm")]
				Backend::Llvm => Self::Llvm,
			}
		}
	}

	#[derive(Debug, Clone, ValueEnum)]
	pub enum PrintKind {
		Ast,
		AstPretty,
		Hir,
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
				PrintKind::BackendIr => Self::BackendIr,
				PrintKind::Items => Self::CollectedItems,
				PrintKind::Env => Self::TypeEnvironment,
			}
		}
	}
}

#[derive(clap::Parser)]
struct Args {
	pub input: Option<PathBuf>,

	#[clap(long)]
	pub jit: bool,

	#[clap(long)]
	pub backend: Option<options::Backend>,

	#[clap(long, default_value = ".cache/kaleic")]
	pub output: PathBuf,
	#[clap(long)]
	pub print: Vec<options::PrintKind>,
}

fn main() {
	FmtSubscriber::builder()
		.with_env_filter(EnvFilter::from_default_env())
		.with_timer(time::Uptime::default())
		.with_writer(std::io::stderr)
		.init();

	let args = Args::parse();

	let mut scx = SessionCtx::default();

	scx.options.input = args.input;
	scx.options.jit = args.jit;
	if let Some(val) = args.backend {
		scx.options.backend = val.into();
	}
	scx.options.output = args.output;
	scx.options
		.print
		.extend(args.print.into_iter().map(Into::into));

	driver::pipeline(&scx);
}
