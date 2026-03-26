use std::{fmt::Write as _, fs, process::Command};

use ariadne::ReportKind;

use crate::{
	ast, codegen, inference, lowerer, parser,
	pretty_print::pretty_print,
	resolve,
	session::{DcxHandle, Diagnostic, PrintKind, Report, SessionCtx, Span},
	ty,
};

pub fn pipeline(scx: &SessionCtx) {
	if fs::exists(&scx.options.debug_output).unwrap() {
		fs::remove_dir_all(&scx.options.debug_output).unwrap();
	}
	fs::create_dir_all(&scx.options.debug_output).unwrap();

	let ast = parse_files(scx);
	scx.dcx().check_sane_or_exit();

	resolve::collect_root(scx, &ast);
	resolve::resolve_root(scx, &ast);
	let hir = lowerer::lower_root(scx, &ast);
	scx.dcx().check_sane_or_exit();

	let name_env = scx.name_env.borrow();
	let lang_items = scx.lang_items.borrow();
	let tcx = ty::TyCtx::new(scx, &name_env, &lang_items);

	ty::compute_items_type(&tcx, &hir);
	ty::check_entrypoint(&tcx);
	inference::infer_root(&tcx, &hir);
	scx.dcx().check_sane_or_exit();

	// TODO: document pass outputs markdown
	// if scx.options.document { }

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
				|artefact| write!(artefact, "{ast:#?}"),
			);
			scx.register_artefact(
				&PrintKind::AstPretty,
				&format!("ast-pretty.{}.txt", filename.file_stem().unwrap().display()),
				|artefact| pretty_print(&ast, artefact),
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
