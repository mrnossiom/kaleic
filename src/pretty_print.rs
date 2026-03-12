//! Pretty print a source file.
//!
//! This essentially prints the AST taking line length into account. It also
//! reads content that not in the AST like comments to not lose any data.

#![expect(unused_variables, clippy::todo)]

use std::fmt::{self, Write as _};

use crate::{ast, session};

pub struct PrettyFormatter<'fmt> {
	inner: &'fmt mut dyn fmt::Write,

	indent: u32,
}

impl<'fmt> PrettyFormatter<'fmt> {
	fn new(inner: &'fmt mut dyn fmt::Write) -> Self {
		Self { inner, indent: 0 }
	}

	fn with_indent(&mut self, f: impl FnOnce(&mut PrettyFormatter) -> fmt::Result) -> fmt::Result {
		self.indent += 1;
		f(self)?;
		self.indent -= 1;
		Ok(())
	}

	fn newline(&mut self) -> fmt::Result {
		writeln!(self.inner)?;
		for _ in 0..self.indent {
			write!(self.inner, "\t")?;
		}
		Ok(())
	}

	fn write_seq<T>(
		&mut self,
		elements: &[T],
		mut print: impl FnMut(&mut Self, &T) -> fmt::Result,
		sep: &str,
	) -> fmt::Result {
		for (i, elem) in elements.iter().enumerate() {
			self.newline()?;
			print(self, elem)?;
			write!(self.inner, "{sep}")?;
		}
		Ok(())
	}

	fn write_seq_oneline<T>(
		&mut self,
		elements: &[T],
		mut print: impl FnMut(&mut Self, &T) -> fmt::Result,
		sep: &str,
	) -> fmt::Result {
		for (i, elem) in elements.iter().enumerate() {
			if i != 0 {
				write!(self.inner, "{sep} ")?;
			}
			print(self, elem)?;
		}
		Ok(())
	}
}

impl fmt::Write for PrettyFormatter<'_> {
	fn write_str(&mut self, s: &str) -> fmt::Result {
		self.inner.write_str(s)
	}
}

pub fn pretty_print(node: &dyn PrettyPrint, mut output: &mut dyn fmt::Write) -> fmt::Result {
	let mut f = PrettyFormatter::new(&mut output);
	node.pprint(&mut f)?;
	Ok(())
}

pub trait PrettyPrint {
	fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result;
}

macro_rules! pp {
    ($f:expr, $($item:tt),* $(,)?) => {{
        $(
            pp!(@single $f, $item);
        )*
        Ok(())
    }};

    // Handle literal strings
    (@single $f:expr, '\n') => { $f.newline()? };
    (@single $f:expr, $lit:literal) => { $f.write_str($lit)? };
    // Handle expressions to be pretty-printed
    (@single $f:expr, ($e:expr)) => { $e.pprint($f)? };
    // Handle optionals
    (@single $f:expr, (? $opt:expr)) => {
        if let Some(ref val) = $opt {
            val.pprint($f)?;
        }
    };
    (@single $f:expr, (? $before:literal $opt:expr)) => {
        if let Some(val) = &$opt {
        	$f.write_str($before)?;
            val.pprint($f)?;
        }
    };
    // Handle lists
    (@single $f:expr, [$list:expr, '\n']) => {
		for item in $list {
			item.pprint($f)?;
			$f.newline()?;
		}
    };
    (@single $f:expr, ['\n', $list:expr]) => {
		for item in $list {
			$f.newline()?;
			item.pprint($f)?;
		}
    };
    (@single $f:expr, [$list:expr, $delim:literal]) => {
		for (i, item) in $list.iter().enumerate() {
			if i > 0 {
				$f.write_str($delim)?;
			}
			item.pprint($f)?;
		}
    };
    // Handle indent block
    (@single $f:expr, { $($more:tt),* }) => {
		$f.with_indent(|f| {
			pp!(f, $($more),*)
		})?;
    };
}

impl<T: PrettyPrint> PrettyPrint for ast::Spanned<T> {
	fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
		self.bit.pprint(f)
	}
}

impl PrettyPrint for ast::Ident {
	fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
		self.sym.pprint(f)
	}
}

impl PrettyPrint for session::Symbol {
	fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
		// TODO
		write!(f, "{self:#?}")
	}
}

mod ast_pp {
	use super::*;

	use crate::ast;

	impl PrettyPrint for ast::Root {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { attrs, items } = &self;
			pp!(f, [attrs, '\n'], '\n', [items, '\n'])
		}
	}

	impl PrettyPrint for ast::Item {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self {
				kind,
				attrs,
				span,
				id: _,
			} = &self;

			pp!(f, [attrs, '\n'], (kind))
		}
	}

	impl PrettyPrint for ast::ItemKind {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match self {
				Self::Function(func) => func.pprint(f),
				Self::TypeAlias(ty) => ty.pprint(f),

				Self::Struct {
					name,
					generics,
					fields,
				} => pp!(
					f,
					"struct ",
					(name),
					(generics),
					" {",
					{ [fields, ","] },
					'\n',
					"}"
				),

				Self::Enum {
					name,
					generics,
					variants,
				} => pp!(
					f,
					"struct ",
					(name),
					(generics),
					" {",
					{ [variants, ","] },
					'\n',
					"}"
				),

				Self::Trait {
					name,
					generics,
					members,
				} => pp!(f, "trait ", (name), " {", { ['\n', members] }, '\n', "}"),

				Self::TraitImpl {
					type_,
					trait_,
					members,
				} => pp!(
					f,
					"for ",
					(type_),
					" impl ",
					(trait_),
					" {",
					{ ['\n', members] },
					'\n',
					"}"
				),

				Self::ForeignMod { items } => {
					pp!(f, "unsafe extern {", { ['\n', items] }, '\n', "}")
				}
			}
		}
	}

	impl PrettyPrint for ast::Variant {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self {
				name,
				kind,
				span: _,
			} = &self;

			match kind {
				ast::VariantKind::Bare => pp!(f, (name)),
				ast::VariantKind::Tuple(fields) => pp!(f, (name), "(", [fields, ","], ")"),
				ast::VariantKind::Struct(fields) => pp!(f, (name), "{", [fields, ","], "}"),
			}
		}
	}

	impl PrettyPrint for ast::Generics {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self(inner) = &self;
			if !inner.is_empty() {
				pp!(f, "<", [inner, ","], ">")?;
			}
			Ok(())
		}
	}

	impl PrettyPrint for ast::FieldDef {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { name, ty, span: _ } = &self;
			pp!(f, (name), ": ", (ty))
		}
	}

	impl PrettyPrint for ast::Function {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { name, decl, body } = &self;

			pp!(f, "fn ", (name), "(", [decl.params, ","], ")", (? " " decl.ret))?;

			if let Some(body) = &body {
				write!(f, " ")?;
				body.pprint(f)?;
			} else {
				write!(f, ";")?;
			}

			Ok(())
		}
	}

	impl PrettyPrint for ast::TypeAlias {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { name, alias } = &self;
			pp!(f, "type ", (name), (? " = " alias), ";")
		}
	}

	impl PrettyPrint for ast::Expr {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { kind, span: _ } = &self;
			kind.pprint(f)
		}
	}

	impl PrettyPrint for ast::ExprKind {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match &self {
				Self::Access { path } => path.pprint(f),
				Self::LiteralStr { sym } => pp!(f, "\"", (sym), "\""),
				Self::LiteralInt { sym } | Self::LiteralFloat { sym } => sym.pprint(f),
				Self::Paren { expr } => pp!(f, "(", (expr), ")"),

				Self::Unary { op, expr } => pp!(f, (op.bit), (expr)),
				Self::Binary { op, left, right } => pp!(f, (left), " ", (op), " ", (right)),
				Self::ShortCircuit { op, left, right } => pp!(f, (left), " ", (op), " ", (right)),

				Self::FnCall { expr, args } => pp!(f, (expr), "(", [args.bit, ", "], ")"),

				Self::If {
					cond,
					conseq,
					altern,
				} => pp!(f, "if ", (cond), " ", (conseq), (? " else " altern)),
				Self::Match { expr, arms } => todo!(),

				Self::Loop { body } => pp!(f, "loop ", (body)),
				Self::WhileLoop { check, body } => pp!(f, "while ", (check), " ", (body)),

				Self::Method { expr, name, params } => {
					pp!(f, (expr), ".", (name), "(", [params, ", "], ")")
				}

				Self::Field { expr, name } => pp!(f, (expr), ".", (name)),

				Self::Deref { expr } => pp!(f, (expr), ".*"),
				Self::Assign { target, value } => pp!(f, (target), " = ", (value)),

				Self::Return { expr } => pp!(f, "return", (? " " expr)),

				Self::Break { expr, label } => pp!(f, "break", (? " '" label), (? " " expr)),
				Self::Continue { label } => pp!(f, "continue", (? " '" label)),
			}
		}
	}

	impl PrettyPrint for ast::Param {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { name, ty } = &self;
			pp!(f, (name), ": ", (ty))
		}
	}

	impl PrettyPrint for ast::Ty {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { kind, span } = &self;
			kind.pprint(f)
		}
	}

	impl PrettyPrint for ast::TyKind {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match &self {
				Self::Path(path) => path.pprint(f),
				Self::Pointer(ty) => pp!(f, "*", (ty)),
				Self::Reference(ty) => pp!(f, "&", (ty)),
				Self::Unit => write!(f, "()"),
			}
		}
	}

	impl PrettyPrint for ast::Path {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { segments, generics } = &self;
			pp!(f, [segments, "::"])?;
			if !generics.is_empty() {
				pp!(f, "<", [generics, ", "], ">")?;
			}
			Ok(())
		}
	}

	impl PrettyPrint for ast::Block {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { stmts, span: _ } = &self;
			pp!(f, "{", { ['\n', stmts] }, '\n', "}")
		}
	}

	impl PrettyPrint for ast::Stmt {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { kind, span: _ } = &self;
			kind.pprint(f)
		}
	}

	impl PrettyPrint for ast::StmtKind {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match &self {
				Self::Let {
					name,
					ty,
					value,
					mutable,
				} => {
					write!(f, "let ")?;
					if *mutable {
						write!(f, "mut ")?;
					}
					pp!(f, (name), (? ": " ty), (? " = " value), ";")
				}
				Self::Empty => write!(f, "; empty stmt"),
				Self::Expr(expr) => pp!(f, (expr), ";"),
				Self::ExprRet(expr) => expr.pprint(f),
			}
		}
	}

	impl PrettyPrint for ast::UnaryOp {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match self {
				Self::Not => write!(f, "!"),
				Self::Minus => write!(f, "-"),
			}
		}
	}

	impl PrettyPrint for ast::BinaryOp {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match self {
				Self::Plus => write!(f, "+"),
				Self::Minus => write!(f, "-"),
				Self::Mul => write!(f, "*"),
				Self::Div => write!(f, "/"),
				Self::Mod => write!(f, "%"),

				Self::And => write!(f, "&"),
				Self::Or => write!(f, "|"),
				Self::Xor => write!(f, "^"),

				Self::Shl => write!(f, "<<"),
				Self::Shr => write!(f, ">>"),

				Self::Gt => write!(f, ">"),
				Self::Ge => write!(f, ">="),
				Self::Lt => write!(f, "<"),
				Self::Le => write!(f, "<="),

				Self::Ne => write!(f, "!="),
				Self::EqEq => write!(f, "=="),
			}
		}
	}

	impl PrettyPrint for ast::ShortCircuitOp {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match self {
				Self::And => write!(f, "and"),
				Self::Or => write!(f, "or"),
			}
		}
	}

	impl PrettyPrint for ast::Attr {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { path, meta, span } = self;

			pp!(f, "#", (path))?;

			match meta {
				ast::AttrMeta::None => {}
				ast::AttrMeta::Tuple(_) | ast::AttrMeta::Map(_) | ast::AttrMeta::List(_) => todo!(),
			}

			Ok(())
		}
	}
}

mod hir_pp {
	use super::*;

	use crate::hir;

	impl PrettyPrint for hir::Root {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { items } = &self;
			pp!(f, [items, '\n'])
		}
	}

	impl<T: PrettyPrint> PrettyPrint for hir::Item<T> {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			self.kind.pprint(f)
		}
	}

	impl PrettyPrint for hir::ItemKind {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match &self {
				Self::Function(func) => func.pprint(f),
				Self::ForeignMod { items } => {
					pp!(f, "unsafe extern {", { ['\n', items] }, '\n', "}")
				}

				Self::TypeAlias(ty) => ty.pprint(f),
				Self::Struct(hir::Struct {
					name,
					generics,
					fields,
				}) => pp!(
					f,
					"struct ",
					(name),
					(generics),
					" {",
					{ ['\n', fields] },
					'\n',
					"}"
				),

				Self::Enum(hir::Enum {
					name,
					generics,
					variants,
				}) => {
					pp!(f, "enum ", (name), (generics), " {")?;
					f.with_indent(|f| {
						f.write_seq(
							variants,
							|f, variant| {
								variant.name.sym.pprint(f)?;
								write!(f, " {{")?;
								f.write_seq_oneline(
									&variant.fields,
									|f, field| field.pprint(f),
									",",
								)?;
								write!(f, "}}")?;
								Ok(())
							},
							",",
						)
					})?;
					pp!(f, '\n', "}}")
				}

				Self::Trait {
					name,
					generics,
					members,
				} => {
					pp!(f, "trait ", (name), (generics), " {")?;
					f.with_indent(|f| {
						for item in members {
							item.pprint(f)?;
						}
						Ok(())
					})?;
					pp!(f, '\n', "}}")
				}
				Self::TraitImpl {
					type_,
					trait_,
					members,
				} => pp!(
					f,
					"impl ",
					(trait_),
					" for ",
					(type_),
					" {",
					{ ['\n', members] },
					'\n',
					"}"
				),
			}
		}
	}

	impl PrettyPrint for hir::TraitItemKind {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match self {
				Self::Function(func) => func.pprint(f),
				Self::TypeAlias(alias) => alias.pprint(f),
			}
		}
	}

	impl PrettyPrint for hir::ForeignItemKind {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match self {
				Self::Function(func) => func.pprint(f),
			}
		}
	}

	impl PrettyPrint for hir::Function {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { name, decl, body } = &self;

			pp!(f, "fn ", (name), "(", [decl.params, ", "], ") ", (decl.ret))?;
			if let Some(body) = &body {
				write!(f, " ")?;
				body.pprint(f)?;
			} else {
				write!(f, ";")?;
			}

			Ok(())
		}
	}

	impl PrettyPrint for hir::Abi {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match self {
				Self::Kalei => write!(f, "kalei"),
				Self::C => write!(f, "c"),
			}
		}
	}

	impl PrettyPrint for hir::TypeAlias {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { name, alias } = &self;
			pp!(f, "type ", (name), (? " = " alias), ";")
		}
	}

	impl PrettyPrint for hir::FieldDef {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { name, ty } = &self;
			pp!(f, (name), ": ", (ty))
		}
	}

	impl PrettyPrint for hir::Block {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self {
				stmts,
				ret,
				span: _,
				id: _,
			} = &self;
			write!(f, "{{")?;
			f.with_indent(|f| {
				for stmt in stmts {
					f.newline()?;
					stmt.pprint(f)?;
				}
				if let Some(expr) = ret {
					f.newline()?;
					expr.pprint(f)?;
				}
				Ok(())
			})?;
			pp!(f, '\n', "}}")
		}
	}

	impl PrettyPrint for hir::Stmt {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self {
				kind,
				span: _,
				id: _,
			} = &self;
			kind.pprint(f)
		}
	}

	impl PrettyPrint for hir::StmtKind {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match &self {
				Self::Let {
					name,
					ty,
					value,
					mutable,
				} => {
					write!(f, "let ")?;
					if *mutable {
						write!(f, "mut ")?;
					}
					pp!(f, (name), (? ": " ty), " = ", (value), ";")
				}
				Self::Expr { expr } => pp!(f, (expr), ";"),
			}
		}
	}

	impl PrettyPrint for hir::Expr {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self {
				kind,
				span: _,
				id: _,
			} = &self;
			kind.pprint(f)
		}
	}

	impl PrettyPrint for hir::ExprKind {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			// TODO: parenthesize ambiguous expressions
			match &self {
				Self::Access { path } => path.pprint(f),
				Self::LiteralStr { sym } => pp!(f, "\"", (sym), "\""),
				Self::LiteralInt { sym } | Self::LiteralFloat { sym } => sym.pprint(f),
				Self::Unit => write!(f, "()"),

				Self::Unary { op, expr } => pp!(f, (op), (expr)),
				Self::Binary { op, left, right } => pp!(f, (left), " ", (op), " ", (right)),

				Self::If {
					cond,
					conseq,
					altern,
				} => pp!(f, "if ", (cond), " ", (conseq), (? " else " altern)),
				Self::Loop { block } => pp!(f, "loop ", (block)),

				Self::FnCall { expr, args } => pp!(f, (expr), "(", [args.bit, ", "], ")"),
				Self::Method { expr, name, params } => {
					pp!(f, (expr), ".", (name), "(", [params, ", "], ")")
				}

				Self::Field { expr, name } => pp!(f, (expr), ".", (name)),
				Self::Deref { expr } => pp!(f, (expr), ".*"),
				Self::Assign { target, value } => pp!(f, (target), " = ", (value)),

				Self::Return { expr } => pp!(f, "return ", (expr)),
				Self::Break { expr, label } => pp!(f, "break ", (? " '" label), (expr)),
				Self::Continue { label } => pp!(f, "continue ", (? " '" label)),
			}
		}
	}
}
