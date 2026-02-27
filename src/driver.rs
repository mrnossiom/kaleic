use std::{fmt::Write as _, fs};

use ariadne::ReportKind;

use crate::{
	lowerer, parser,
	pretty_print::pretty_print,
	session::{Diagnostic, PrintKind, Report, SessionCtx, Span},
	ty,
};

pub fn pipeline(scx: &SessionCtx) {
	_ = fs::remove_dir_all(&scx.options.debug_output);
	fs::create_dir_all(&scx.options.debug_output).unwrap();

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
		let mut artefact = scx.register_artefact("ast.txt");
		write!(artefact, "{ast:#?}").unwrap();
	}
	if scx.options.print.contains(&PrintKind::AstPretty) {
		let mut artefact = scx.register_artefact("ast-pretty.txt");
		pretty_print(&ast, &mut artefact).unwrap();
	}

	scx.dcx().check_sane_or_exit();

	// lowering to HIR
	let hir = lowerer::lower_root(scx, &ast);
	if scx.options.print.contains(&PrintKind::HigherIr) {
		let mut artefact = scx.register_artefact("hir.txt");
		write!(artefact, "{hir:#?}").unwrap();
	}
	if scx.options.print.contains(&PrintKind::HigherIrPretty) {
		let mut artefact = scx.register_artefact("hir-pretty.txt");
		pretty_print(&hir, &mut artefact).unwrap();
	}

	scx.dcx().check_sane_or_exit();

	// type collection, inference and analysis
	let tcx = ty::TyCtx::new(scx);

	tcx.collect_items(&hir);
	if scx.options.print.contains(&PrintKind::CollectedItems) {
		let item_map = tcx.name_env.borrow();
		let name_environment = item_map.as_ref().unwrap();
		let mut artefact = scx.register_artefact("name-environment.txt");
		writeln!(artefact, "> Type items:").unwrap();
		for (name, item) in &name_environment.types {
			writeln!(artefact, "{name:#?}: {item:?}").unwrap();
		}
		writeln!(artefact, "> Value items:").unwrap();
		for (name, item) in &name_environment.values {
			writeln!(artefact, "{name:#?}: {item:?}").unwrap();
		}
	}

	tcx.compute_items_type(&hir);
	if scx.options.print.contains(&PrintKind::TypeEnvironment) {
		let env = tcx.ty_env.borrow();
		let mut artefact = scx.register_artefact("type-environment.txt");
		for (name, ty) in env.as_ref().unwrap().iter() {
			writeln!(artefact, "{name:?}: {ty:?}").unwrap();
		}
		writeln!(artefact).unwrap();
	}

	// tcx.typeck(&hir);
	tcx.typeck_old(&hir);

	scx.dcx().check_sane_or_exit();

	// codegen hir bodies
	if scx.options.jit {
		let Some(mut backend) = scx.options.backend.jit_backend(&tcx) else {
			panic!("cannot jit for backend {:?}", scx.options.backend)
		};

		backend.codegen_root(&hir);
		backend.call_main();

		eprintln!("Finished execution!");
	} else {
		let Some(mut backend) = scx.options.backend.object_backend(&tcx) else {
			panic!("cannot codegen for backend {:?}", scx.options.backend)
		};

		backend.codegen_root(&hir);

		let main_object = scx.options.output.join("main.o");
		let object = backend.write_object(&main_object);

		// link to binary
		let mut cmd = std::process::Command::new("wild");

		cmd.arg(&main_object);

		// link libc
		cmd.args(["-l", "c"]);
		// no `_start` symbol
		cmd.args(["-e", "main"]);

		cmd.arg("--output");
		let binary = scx.options.output.join("binary.elf");
		cmd.arg(&binary);

		cmd.status().unwrap();

		eprintln!("Successfully linked binary to `{}`!", binary.display());
	}
}
