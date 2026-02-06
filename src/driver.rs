use std::{fmt::Write as _, fs, io::Write as _};

use ariadne::ReportKind;

use crate::{
	codegen::{self, Backend, CodeGenBackend, JitBackend, ObjectBackend},
	lowerer, parser, pretty_print,
	session::{Diagnostic, OutputKind, PrintKind, Report, SessionCtx, Span},
	ty,
};

pub fn pipeline(scx: &SessionCtx) {
	let debug_output = scx.options.output.join("debug");
	_ = fs::remove_dir_all(&debug_output);
	fs::create_dir_all(&debug_output).unwrap();

	let filename = scx.options.input.as_ref().unwrap_or_else(|| {
		let report = Report::build(ReportKind::Error, Span::DUMMY)
			.with_message("expected an input filename");
		scx.dcx().emit_fatal(&Diagnostic::new(report))
	});

	let source = scx
		.source_map
		.write()
		.load_source_from_file(filename)
		.unwrap();

	// parsing source
	let ast = parser::parse_root(scx, &source);
	if scx.options.print.contains(&PrintKind::Ast) {
		let mut file = fs::File::create(debug_output.join("ast.txt")).unwrap();
		write!(file, "{ast:#?}").unwrap();
	}
	if scx.options.print.contains(&PrintKind::AstPretty) {
		let mut file = fs::File::create(debug_output.join("ast-pretty.txt")).unwrap();
		pretty_print::pretty_print_root(&ast, &mut file).unwrap();
	}

	scx.dcx().check_sane_or_exit();

	// lowering to HIR
	let hir = lowerer::lower_root(scx, &ast);
	if scx.options.print.contains(&PrintKind::HigherIr) {
		let mut file = fs::File::create(debug_output.join("hir.txt")).unwrap();
		write!(file, "{hir:#?}").unwrap();
	}

	scx.dcx().check_sane_or_exit();

	// type collection, inference and analysis
	let tcx = ty::TyCtx::new(scx);

	tcx.collect_items(&hir);
	if scx.options.print.contains(&PrintKind::CollectedItems) {
		let item_map = tcx.name_env.borrow();
		let name_environment = item_map.as_ref().unwrap();
		let mut file = fs::File::create(debug_output.join("name-environment.txt")).unwrap();
		writeln!(file, "> Type items:").unwrap();
		for (name, item) in &name_environment.types {
			writeln!(file, "{name:#?}: {item:?}").unwrap();
		}
		writeln!(file, "> Value items:").unwrap();
		for (name, item) in &name_environment.values {
			writeln!(file, "{name:#?}: {item:?}").unwrap();
		}
	}

	tcx.compute_items_type(&hir);
	if scx.options.print.contains(&PrintKind::TypeEnvironment) {
		let env = tcx.ty_env.borrow();
		let mut file = fs::File::create(debug_output.join("type-environment.txt")).unwrap();
		for (name, ty) in env.as_ref().unwrap().iter() {
			writeln!(file, "{name:?}: {ty:?}").unwrap();
		}
		writeln!(file).unwrap();
	}

	// tcx.typeck(&hir);
	tcx.typeck_old(&hir);

	scx.dcx().check_sane_or_exit();

	// lower HIR bodies to TBIR
	// codegen TBIR bodies
	if scx.options.jit {
		let backend: &mut dyn JitBackend = match scx.options.backend {
			#[cfg(feature = "cranelift")]
			Backend::Cranelift => &mut codegen::CraneliftBackend::new_jit(&tcx),
			#[cfg(feature = "llvm")]
			Backend::Llvm => &mut codegen::LlvmBackend::new_jit(&tcx),
			Backend::NoBackend => panic!("cannot codegen without a backend"),
		};

		backend.codegen_root(&hir);
		backend.call_main();
	} else {
		// let mut backend = match scx.options.backend {
		// 	#[cfg(feature = "cranelift")]
		// 	Backend::Cranelift => codegen::CraneliftBackend::new_object(&tcx),
		// 	#[cfg(feature = "llvm")]
		// 	Backend::Llvm => todo!("no object backend for llvm"),
		// };

		// backend.codegen_root(&hir);

		// let object = backend.get_object();
		// let bytes = object.emit().unwrap();
		// std::fs::write(path, bytes).unwrap();
	}

	tracing::info!("Reached pipeline end successfully!");
}
