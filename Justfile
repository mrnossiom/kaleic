_default:
	@just --list --unsorted --list-heading '' --list-prefix '— '

# Build kaleic in debug mode
build *args:
	cargo build -F debug {{args}}

# Compile the program with args given to kaleic
compile *args:
	cargo run -F debug -- {{args}}

# Compile the given program and execute with the given args
exec program *args:
	#!/usr/bin/env bash
	set -euo pipefail

	compiler_flags=$(echo "{{args}}" | sed 's/ -- .*//; s/ --//')
	program_args=$(echo "{{args}}" | sed -n 's/.* -- //p')

	just compile {{program}} $compiler_flags
	.cache/kaleic/binary.elf $program_args

serve-docs *args:
	mdbook serve docs {{args}}
