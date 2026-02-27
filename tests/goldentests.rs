use goldentests::TestConfig;

#[test]
fn ui() -> Result<(), Box<dyn std::error::Error>> {
	let bless = std::env::var("BLESS").is_ok();

	TestConfig {
		test_path: "tests/ui".into(),

		// TODO: make it dynamic
		binary_path: "target/x86_64-unknown-linux-gnu/debug/kaleic".into(),
		base_args: "".into(),
		base_args_after: "".into(),

		test_args_prefix: "//@args".into(),
		test_args_after_prefix: "//@extra_args".into(),

		test_exit_status_prefix: "//@exit".into(),
		test_stdout_prefix: "//@stdout".into(),
		test_stderr_prefix: "//@stderr".into(),
		test_line_prefix: "//~ ".into(),

		overwrite_tests: bless,
	}
	.run_tests()
	.map_err(|_| "golden tests failed")?;

	Ok(())
}
