use std::{fmt::Write as _, fs, process::Command};

use ariadne::ReportKind;

use crate::{
	ast, codegen, lowerer, parser,
	pretty_print::pretty_print,
	resolve,
	session::{DcxHandle, Diagnostic, PrintKind, Report, SessionCtx, Span},
	ty,
};

pub fn pipeline(scx: &SessionCtx) {
	_ = fs::remove_dir_all(&scx.options.debug_output);
	fs::create_dir_all(&scx.options.debug_output).unwrap();

	let ast = parse_files(scx);

	scx.dcx().check_sane_or_exit();

	let name_env = resolve::collect_root(scx, &ast);
	() = resolve::resolve_root(scx, &ast);

	scx.register_artefact(
		&PrintKind::CollectedItems,
		"name-environment.txt",
		|artefact| {
			writeln!(artefact, "> Type items:").unwrap();
			for (name, item) in &name_env.types {
				writeln!(artefact, "{name:#?}: {item:?}").unwrap();
			}
			writeln!(artefact, "> Value items:").unwrap();
			for (name, item) in &name_env.values {
				writeln!(artefact, "{name:#?}: {item:?}").unwrap();
			}
		},
	);

	// lowering to HIR
	let hir = lowerer::lower_root(scx, &ast);

	scx.register_artefact(&PrintKind::HigherIr, "hir.txt", |artefact| {
		write!(artefact, "{hir:#?}").unwrap();
	});
	scx.register_artefact(&PrintKind::HigherIrPretty, "hir-pretty.txt", |artefact| {
		pretty_print(&hir, artefact).unwrap();
	});

	scx.dcx().check_sane_or_exit();

	// type collection, inference and analysis
	let tcx = ty::TyCtx::new(scx);
	tcx.name_env.put(name_env);

	tcx.compute_items_type(&hir);
	scx.register_artefact(
		&PrintKind::TypeEnvironment,
		"type-environment.txt",
		|artefact| {
			let env = tcx.type_env.borrow();
			for (name, ty) in env.iter() {
				writeln!(artefact, "{name:?}: {ty:?}").unwrap();
			}
			writeln!(artefact).unwrap();
		},
	);

	tcx.typeck(&hir);

	scx.dcx().check_sane_or_exit();

	// codegen hir bodies
	if scx.options.jit {
		let Some(mut backend) = scx.options.backend.jit_backend(&tcx) else {
			panic!("cannot jit for backend {:?}", scx.options.backend)
		};

		backend.codegen_root(&hir);
		backend.finalize();
		backend.call_main();
	} else {
		let Some(mut backend) = scx.options.backend.object_backend(&tcx) else {
			panic!("cannot codegen for backend {:?}", scx.options.backend)
		};

		backend.codegen_root(&hir);

		let main_object = scx.options.output.join("main.o");
		backend.write_object(&main_object);

		// link to binary

		let mut cmd = match scx.options.linker {
			codegen::Linker::Ld => Command::new("ld"),
			codegen::Linker::Lld => Command::new("lld"),
			codegen::Linker::Wild => Command::new("wild"),
		};

		cmd.arg(&main_object);

		// link libc
		cmd.args(["-l", "c"]);

		cmd.arg("--output");
		let binary = scx.options.output.join("binary.elf");
		cmd.arg(&binary);

		cmd.status().unwrap();
	}
}

fn parse_files(scx: &SessionCtx) -> ast::Root {
	if scx.options.inputs.is_empty() {
		let report = Report::build(ReportKind::Error, Span::DUMMY)
			.with_message("expected at least one input file");
		scx.dcx().emit_fatal(&Diagnostic::new(report))
	}

	scx.options
		.inputs
		.iter()
		.map(|filename| {
			let source = scx
				.source_map
				.write()
				.load_source_from_file(filename)
				.unwrap();

			// parsing source
			let ast = parser::parse_root(scx, &source);

			scx.register_artefact(
				&PrintKind::Ast,
				&format!("ast.{}.txt", filename.file_stem().unwrap().display()),
				|artefact| write!(artefact, "{ast:#?}").unwrap(),
			);
			scx.register_artefact(
				&PrintKind::AstPretty,
				&format!("ast-pretty.{}.txt", filename.file_stem().unwrap().display()),
				|artefact| pretty_print(&ast, artefact).unwrap(),
			);

			ast
		})
		.fold(
			ast::Root {
				attrs: Vec::new(),
				items: Vec::new(),
			},
			|mut final_, cur| {
				final_.attrs.extend(cur.attrs);
				final_.items.extend(cur.items);
				final_
			},
		)
}
