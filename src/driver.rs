use ariadne::ReportKind;

use crate::{
	codegen::{self, Backend, CodeGenBackend, JitBackend, ObjectBackend},
	lowerer, parser, pretty_print,
	session::{Diagnostic, OutputKind, PrintKind, Report, SessionCtx, Span},
	ty,
};

pub fn pipeline(scx: &SessionCtx) {
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
		println!("{ast:#?}");
	}
	if scx.options.print.contains(&PrintKind::AstPretty) {
		pretty_print::pretty_print_root(&ast).unwrap();
	}

	scx.dcx().check_sane_or_exit();

	// lowering to HIR
	let hir = lowerer::lower_root(scx, &ast);
	if scx.options.print.contains(&PrintKind::HigherIr) {
		println!("{hir:#?}");
	}

	scx.dcx().check_sane_or_exit();

	// type collection, inference and analysis
	let tcx = ty::TyCtx::new(scx);

	tcx.collect_root(&hir);
	if scx.options.print.contains(&PrintKind::CollectedItems) {
		let item_map = tcx.item_map.borrow();
		for (name, item) in item_map.as_ref().unwrap() {
			let mut item_info = format!("{:?}", item.kind);
			item_info.truncate(60);
			println!("{name:?}: {item_info}");
		}
		println!();
	}

	// TODO: move in collect_root?
	tcx.compute_env(&hir);
	if scx.options.print.contains(&PrintKind::Environment) {
		let env = tcx.environment.borrow();
		for (name, ty) in &env.as_ref().unwrap().values {
			println!("{name:?}: {ty:?}");
		}
		println!();
	}

	tcx.typeck();

	scx.dcx().check_sane_or_exit();

	// lower HIR bodies to TBIR
	// codegen TBIR bodies
	match &scx.options.output {
		OutputKind::Jit => {
			let backend: &mut dyn JitBackend = match scx.options.backend {
				#[cfg(feature = "cranelift")]
				Backend::Cranelift => &mut codegen::CraneliftBackend::new_jit(&tcx),
				#[cfg(feature = "llvm")]
				Backend::Llvm => &mut codegen::LlvmBackend::new_jit(&tcx),
				Backend::NoBackend => panic!("cannot codegen without a backend"),
			};

			backend.codegen_root(&hir);
			backend.call_main();
		}
		OutputKind::Object(path) => {
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
	}

	tracing::info!("Reached pipeline end successfully!");
}
