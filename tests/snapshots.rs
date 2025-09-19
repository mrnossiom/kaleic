use kaleic::{lowerer::lower_root, parser::parse_root, session};

#[test]
fn fibo() {
	let source = include_str!("../samples/fibonacci.kl");

	let scx = session::SessionCtx::default();

	let source = scx.source_map.write().load_source("entry", source.into());

	let ast = parse_root(&scx, &source);
	insta::assert_debug_snapshot!(ast);

	// lowering to HIR
	let hir = lower_root(&scx, &ast);
	insta::assert_debug_snapshot!(hir);
}
