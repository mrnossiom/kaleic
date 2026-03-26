{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs?ref=nixos-unstable";

    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";

    mdbook-treesitter.url = "github:mrnossiom/mdbook-treesitter";
    mdbook-treesitter.inputs.nixpkgs.follows = "nixpkgs";

    tree-sitter-kalei.url = "github:mrnossiom/tree-sitter-kalei";
    tree-sitter-kalei.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
      mdbook-treesitter,
      tree-sitter-kalei,
    }:
    let
      inherit (nixpkgs.lib) genAttrs;

      forAllSystems = genAttrs [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forAllPkgs =
        function:
        forAllSystems (
          system:
          function {
            pkgs = pkgs.${system};
            lpkgs = lpkgs.${system};
          }
        );

      pkgs = forAllSystems (
        system:
        import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        }
      );
      lpkgs = forAllSystems (system: {
        inherit (mdbook-treesitter.packages.${system}) mdbook-treesitter;
        inherit (tree-sitter-kalei.packages.${system}) tree-sitter-kalei;
      });
    in
    {
      formatter = forAllPkgs ({ pkgs, ... }: pkgs.nixfmt-tree);

      packages = forAllPkgs (
        { pkgs, ... }:
        {
          docs = pkgs.stdenv.mkDerivation {
            name = "kaleic-docs";
            src = ./docs;
            nativeBuildInputs = [ pkgs.mdbook ];
            buildPhase = "mdbook build";
            installPhase = "cp -r book $out";
          };
        }
      );

      devShells = forAllPkgs (
        { pkgs, lpkgs }:
        let
          file-rust-toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
          rust-toolchain = file-rust-toolchain.override { extensions = [ "rust-analyzer" ]; };
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              pkg-config
              rust-toolchain
              lldb
              typos

              mdbook
              lpkgs.mdbook-treesitter

              # linker
              wild

              # inkwell/llvm deps
              llvmPackages_21.llvm.dev
              libffi
              libxml2
            ];

            shellHook = ''
              echo "[flake] copying tree-sitter parser and queries for kalei and rust"
              mkdir -p docs/treesitter/kalei docs/treesitter/rust
              cp -f ${lpkgs.tree-sitter-kalei}/parser docs/treesitter/kalei.so
              cp -f ${lpkgs.tree-sitter-kalei}/queries/* docs/treesitter/kalei/
              cp -f ${pkgs.tree-sitter-grammars.tree-sitter-rust}/parser docs/treesitter/rust.so
              cp -f ${pkgs.tree-sitter-grammars.tree-sitter-rust}/queries/* docs/treesitter/rust/
            '';

            RUST_SRC_PATH = pkgs.rustPlatform.rustLibSrc;

            LLVM_SYS_211_PREFIX = pkgs.llvmPackages_21.llvm.dev;

            RUST_BACKTRACE = "1";
            RUST_LOG = "info,kaleic=debug,cranelift_jit=warn,cranelift_object=warn";
          };
        }
      );
    };
}
