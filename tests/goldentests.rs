use goldentests::TestConfig;

#[test]
fn ui() -> Result<(), Box<dyn std::error::Error>> {
	let bless = std::env::var("BLESS").is_ok();

	TestConfig::with_custom_keywords(
		"target/x86_64-unknown-linux-gnu/debug/kaleic",
		"tests/ui",
		"//@",
		"args ",
		"extra_args ",
		"stdout",
		"stderr",
		"exit ",
		bless,
	)
	.run_tests()
	.map_err(|()| "golden tests failed")?;

	Ok(())
}
