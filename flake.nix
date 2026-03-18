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
      inherit (nixpkgs.lib) genAttrs cleanSource;

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
        inherit (self.packages.${system}) mdbook-treesitter-grammars;
      });
    in
    {
      formatter = forAllPkgs ({ pkgs, ... }: pkgs.nixfmt-tree);

      packages = forAllPkgs (
        { pkgs, lpkgs }:
        let
          bundleMdBookTreeSitterGrammars =
            grammars:
            pkgs.stdenv.mkDerivation {
              name = "mdbook-treesitter-grammars";
              unpackPhase = "true";
              installPhase = ''
                mkdir $out
              ''
              + pkgs.lib.concatStringsSep "\n" (
                pkgs.lib.mapAttrsToList (name: grammar: ''
                  mkdir $out/${name}
                  cp ${grammar}/parser $out/${name}.so
                  cp ${grammar}/queries/* $out/${name}/
                '') grammars
              );
            };
        in
        {
          book = pkgs.stdenv.mkDerivation {
            name = "kaleic-book";
            src = cleanSource ./docs;
            nativeBuildInputs = [
              pkgs.mdbook
              lpkgs.mdbook-treesitter
            ];
            postPatch = ''
              ln -s ${lpkgs.mdbook-treesitter-grammars} treesitter
            '';
            buildPhase = "mdbook build";
            installPhase = "cp -r book $out";
          };

          rust-docs =
            let
              rust-toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
              rustPlatform = pkgs.makeRustPlatform {
                cargo = rust-toolchain;
                rustc = rust-toolchain;
              };
            in
            rustPlatform.buildRustPackage {
              pname = "kaleic-rust-docs";
              version = "0.0.0";
              src = cleanSource ./.;

              cargoLock = {
                lockFile = ./Cargo.lock;
                outputHashes = {
                  "ariadne-0.6.0" = "sha256-G13rZlJB+qJ+wLvXICct/rlqfEyZw2kzY7I0aYd0czA=";
                };
              };

              nativeBuildInputs = [
                pkgs.pkg-config
                pkgs.llvmPackages_21.llvm.dev
              ];
              buildInputs = [
                pkgs.libffi
                pkgs.libxml2
              ];

              LLVM_SYS_211_PREFIX = pkgs.llvmPackages_21.llvm.dev;

              buildPhase = "cargo doc --all --no-deps --document-private-items";
              installPhase = "cp -r target/doc $out";

              doCheck = false;
            };

          mdbook-treesitter-grammars = bundleMdBookTreeSitterGrammars {
            kalei = lpkgs.tree-sitter-kalei;
            rust = pkgs.tree-sitter-grammars.tree-sitter-rust;
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
              TARGET="docs/treesitter"
              CURRENT_STORE_PATH="${lpkgs.mdbook-treesitter-grammars}"
              if [ -L "$TARGET" ]; then
                EXISTING_PATH=$(readlink -f "$TARGET")

                # relink only if the target path is different
                if [ "$EXISTING_PATH" != "$CURRENT_STORE_PATH" ]; then
                  echo "[flake] relinking '$TARGET' to the new tree-sitter parsers and queries"
                  rm "$TARGET"
                  ln -s "$CURRENT_STORE_PATH" "$TARGET"
                fi
              elif [ -e "$TARGET" ]; then
                echo "[flake] WARNING: '$TARGET' is not a symlink. skipping."
              else
                echo "[flake] linking '$TARGET' to the tree-sitter parsers and queries"
                ln -s "$CURRENT_STORE_PATH" "$TARGET"
              fi
            '';

            RUST_SRC_PATH = pkgs.rustPlatform.rustLibSrc;

            LLVM_SYS_211_PREFIX = pkgs.llvmPackages_21.llvm.dev;

            RUST_BACKTRACE = "1";
          };
        }
      );
    };
}
