_default:
	@just --list --unsorted --list-heading '' --list-prefix '—— '

# Build kaleic in debug mode
build *args:
	cargo build -F debug {{args}}

# Compile the program with args given to kaleic
compile *args:
	cargo run -F debug -- {{args}}

# Compile the given program and execute with the given args
exec program *args:
	cargo run -F debug -- {{program}}
	.cache/kaleic/binary.elf {{args}}
