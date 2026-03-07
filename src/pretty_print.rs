//! Pretty print a source file.
//!
//! This essentially prints the AST taking line length into account. It also
//! reads content that not in the AST like comments to not lose any data.

#![expect(unused_variables, clippy::todo)]

use std::fmt::{self, Write as _};

use crate::session::Symbol;

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

mod ast_pp {
	use super::*;

	use crate::ast;

	impl PrettyPrint for ast::Root {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { attrs, items } = &self;

			for attr in attrs {
				attr.pprint(f)?;
				f.newline()?;
			}
			f.newline()?;

			for item in items {
				item.pprint(f)?;
				f.newline()?;

				f.newline()?;
			}

			Ok(())
		}
	}

	impl PrettyPrint for ast::Item {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { kind, attrs, span } = &self;

			for attr in attrs {
				attr.pprint(f)?;
				f.newline()?;
			}
			kind.pprint(f)?;
			Ok(())
		}
	}

	impl PrettyPrint for ast::ItemKind {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match self {
				Self::Function(func) => func.pprint(f)?,
				Self::TypeAlias(ty) => ty.pprint(f)?,

				Self::Struct {
					name,
					generics,
					fields,
				} => {
					write!(f, "struct ")?;
					name.sym.pprint(f)?;
					generics.pprint(f)?;
					write!(f, " {{")?;
					f.with_indent(|f| f.write_seq(fields, |f, variant| variant.pprint(f), ","))?;
					f.newline()?;
					write!(f, "}}")?;
				}
				Self::Enum {
					name,
					generics,
					variants,
				} => {
					write!(f, "enum ")?;
					name.sym.pprint(f)?;
					generics.pprint(f)?;
					write!(f, " {{")?;
					f.with_indent(|f| {
						f.write_seq(
							variants,
							|f, variant| {
								variant.name.sym.pprint(f)?;
								match &variant.kind {
									ast::VariantKind::Bare => {}
									ast::VariantKind::Tuple(fields) => {
										write!(f, "(")?;
										f.write_seq_oneline(
											fields,
											|f, field| field.pprint(f),
											",",
										)?;
										write!(f, ")")?;
									}
									ast::VariantKind::Struct(fields) => {
										write!(f, " {{")?;
										f.write_seq_oneline(
											fields,
											|f, field| field.pprint(f),
											",",
										)?;
										write!(f, "}}")?;
									}
								}
								Ok(())
							},
							",",
						)
					})?;
					f.newline()?;
					write!(f, "}}")?;
				}
				Self::Trait {
					name,
					generics,
					members,
				} => todo!(),
				Self::TraitImpl {
					type_,
					trait_,
					members,
				} => todo!(),
				Self::Extern { items } => {
					write!(f, "unsafe extern {{")?;
					f.with_indent(|f| {
						for item in items {
							f.newline()?;
							item.pprint(f)?;
						}
						Ok(())
					})?;
					f.newline()?;
					write!(f, "}}")?;
				}
			}
			Ok(())
		}
	}

	impl PrettyPrint for ast::Generics {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self(inner) = &self;
			if !inner.is_empty() {
				write!(f, "<")?;
				f.write_seq_oneline(inner, |f, generic| generic.sym.pprint(f), ",")?;
				write!(f, ">")?;
			}
			Ok(())
		}
	}

	impl PrettyPrint for ast::FieldDef {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { name, ty, span: _ } = &self;
			name.sym.pprint(f)?;
			write!(f, ": ")?;
			ty.pprint(f)?;
			Ok(())
		}
	}

	impl PrettyPrint for ast::Function {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { name, decl, body } = &self;

			write!(f, "fn ")?;
			name.sym.pprint(f)?;
			write!(f, "(")?;
			f.write_seq_oneline(&decl.params, |f, param| param.pprint(f), ", ")?;
			write!(f, ")")?;

			if let Some(ret) = &decl.ret {
				write!(f, " ")?;
				ret.pprint(f)?;
			}

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
			write!(f, "type ")?;
			name.sym.pprint(f)?;
			if let Some(alias) = alias {
				write!(f, " = ")?;
				alias.pprint(f)?;
			}
			write!(f, ";")?;
			Ok(())
		}
	}

	impl PrettyPrint for ast::Expr {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { kind, span } = &self;

			Ok(())
		}
	}

	impl PrettyPrint for ast::ExprKind {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match &self {
				Self::Access { path } => path.pprint(f)?,
				Self::LiteralStr { sym } => {
					write!(f, "\"")?;
					sym.pprint(f)?;
					write!(f, "\"")?;
				}
				Self::LiteralInt { sym } | Self::LiteralFloat { sym } => {
					sym.pprint(f)?;
				}

				Self::Paren { expr } => {
					write!(f, "(")?;
					expr.pprint(f)?;
					write!(f, ")")?;
				}
				Self::Unary { op, expr } => {
					op.bit.pprint(f)?;
					expr.pprint(f)?;
				}
				Self::Binary { op, left, right } => {
					left.pprint(f)?;
					write!(f, " ")?;
					op.bit.pprint(f)?;
					write!(f, " ")?;
					right.pprint(f)?;
				}
				Self::ShortCircuit { op, left, right } => {
					left.pprint(f)?;
					write!(f, " ")?;
					op.bit.pprint(f)?;
					write!(f, " ")?;
					right.pprint(f)?;
				}

				Self::FnCall { expr, args } => {
					expr.pprint(f)?;
					write!(f, "(")?;
					f.write_seq_oneline(&args.bit, |f, arg| arg.pprint(f), ",")?;
					write!(f, ")")?;
				}
				Self::If {
					cond,
					conseq,
					altern,
				} => {
					write!(f, "if ")?;
					cond.pprint(f)?;
					write!(f, " ")?;
					conseq.pprint(f)?;
					if let Some(altern) = altern {
						write!(f, " else ")?;
						altern.pprint(f)?;
					}
				}
				Self::Match { expr, arms } => todo!(),
				Self::Loop { body } => {
					write!(f, "loop ")?;
					body.pprint(f)?;
				}
				Self::WhileLoop { check, body } => {
					write!(f, "while ")?;
					check.pprint(f)?;
					write!(f, " ")?;
					body.pprint(f)?;
				}

				Self::Method { expr, name, params } => {
					expr.pprint(f)?;
					write!(f, ".")?;
					name.sym.pprint(f)?;
					write!(f, "(")?;
					f.write_seq_oneline(params, |f, param| param.pprint(f), ",")?;
					write!(f, ")")?;
				}
				Self::Field { expr, name } => {
					expr.pprint(f)?;
					write!(f, ".")?;
					name.sym.pprint(f)?;
				}
				Self::Deref { expr } => {
					expr.pprint(f)?;
					write!(f, ".*")?;
				}
				Self::Assign { target, value } => {
					target.pprint(f)?;
					write!(f, " = ")?;
					value.pprint(f)?;
				}
				Self::Return { expr } => {
					write!(f, "return")?;
					if let Some(expr) = expr {
						write!(f, " ")?;
						expr.pprint(f)?;
					}
				}
				Self::Break { expr, label } => {
					write!(f, "break ")?;
					if let Some(expr) = expr {
						write!(f, " ")?;
						expr.pprint(f)?;
					}
				}
				Self::Continue { label } => write!(f, "continue")?,
			}
			Ok(())
		}
	}

	impl PrettyPrint for ast::Param {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { name, ty } = &self;
			name.sym.pprint(f)?;
			write!(f, ": ")?;
			ty.pprint(f)?;
			Ok(())
		}
	}

	impl PrettyPrint for ast::Ty {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { kind, span } = &self;
			kind.pprint(f)?;
			Ok(())
		}
	}

	impl PrettyPrint for ast::TyKind {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match &self {
				Self::Path(path) => path.pprint(f)?,

				Self::Pointer(ty) => {
					write!(f, "*")?;
					ty.pprint(f)?;
				}
				Self::Reference(ty) => {
					write!(f, "&")?;
					ty.pprint(f)?;
				}
				Self::Unit => write!(f, "()")?,
				Self::ImplicitInfer => write!(f, "_")?,
			}
			Ok(())
		}
	}

	impl PrettyPrint for Symbol {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			// TODO
			write!(f, "{self:#?}")
		}
	}

	impl PrettyPrint for ast::Path {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { segments, generics } = &self;
			f.write_seq_oneline(segments, |f, segment| segment.sym.pprint(f), "::")?;
			if !self.generics.is_empty() {
				write!(f, "<")?;
				f.write_seq_oneline(generics, |f, generic| generic.pprint(f), ", ")?;
				write!(f, ">")?;
			}
			Ok(())
		}
	}

	impl PrettyPrint for ast::Block {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { stmts, span: _ } = &self;
			write!(f, "{{")?;
			f.with_indent(|f| {
				for stmt in stmts {
					f.newline()?;
					stmt.pprint(f)?;
				}
				Ok(())
			})?;
			f.newline()?;
			write!(f, "}}")?;
			Ok(())
		}
	}

	impl PrettyPrint for ast::Stmt {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { kind, span: _ } = &self;
			kind.pprint(f)?;
			Ok(())
		}
	}

	impl PrettyPrint for ast::StmtKind {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match &self {
				Self::Let {
					ident: name,
					ty,
					value,
					mutable,
				} => {
					write!(f, "let ")?;
					if *mutable {
						write!(f, "mut ")?;
					}
					name.sym.pprint(f)?;
					if let Some(ty) = &ty {
						write!(f, ": ")?;
						ty.pprint(f)?;
					}
					if let Some(value) = &value {
						write!(f, " = ")?;
						value.pprint(f)?;
					}
					write!(f, ";")?;
				}

				Self::Empty => write!(f, "; empty stmt")?,
				Self::Expr(expr) => {
					expr.pprint(f)?;
					write!(f, ";")?;
				}
				Self::ExprRet(expr) => expr.pprint(f)?,
			}
			Ok(())
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

			write!(f, "#")?;
			path.pprint(f)?;

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
			for item in items {
				item.kind.pprint(f)?;
				f.newline()?;

				f.newline()?;
			}
			Ok(())
		}
	}

	impl PrettyPrint for hir::ItemKind {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match &self {
				Self::Function(func) => func.pprint(f)?,
				Self::Extern { items } => {
					write!(f, "unsafe extern {{")?;
					f.with_indent(|f| {
						for item in items {
							Self::from(item.kind.clone()).pprint(f)?;
						}
						Ok(())
					})?;
					f.newline()?;
					write!(f, "}}")?;
				}

				Self::TypeAlias(ty) => ty.pprint(f)?,
				Self::Struct(hir::Struct {
					name,
					generics,
					fields,
				}) => {
					write!(f, "struct ")?;
					name.sym.pprint(f)?;
					generics.pprint(f)?;
					write!(f, " {{")?;
					f.with_indent(|f| f.write_seq(fields, |f, variant| variant.pprint(f), ","))?;
					f.newline()?;
					write!(f, "}}")?;
				}
				Self::Enum(hir::Enum {
					name,
					generics,
					variants,
				}) => {
					write!(f, "enum ")?;
					name.sym.pprint(f)?;
					generics.pprint(f)?;
					write!(f, " {{")?;
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
					f.newline()?;
					write!(f, "}}")?;
				}

				Self::Trait {
					name,
					generics,
					members,
				} => {
					write!(f, "trait ")?;
					name.sym.pprint(f)?;
					// TODO: generics
					write!(f, " {{")?;
					f.with_indent(|f| {
						for item in members {
							Self::from(item.kind.clone()).pprint(f)?;
						}
						Ok(())
					})?;
					f.newline()?;
					write!(f, "}}")?;
				}
				Self::TraitImpl {
					type_,
					trait_,
					members,
				} => {
					write!(f, "impl ")?;
					trait_.pprint(f)?;
					write!(f, " for ")?;
					type_.pprint(f)?;
					write!(f, " {{")?;
					f.with_indent(|f| {
						for item in members {
							Self::from(item.kind.clone()).pprint(f)?;
						}
						Ok(())
					})?;
					f.newline()?;
					write!(f, "}}")?;
				}
			}
			Ok(())
		}
	}

	impl PrettyPrint for hir::Function {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { name, decl, body } = &self;

			write!(f, "fn ")?;
			name.sym.pprint(f)?;
			write!(f, "(")?;
			f.write_seq_oneline(&decl.params, |f, param| param.pprint(f), ", ")?;
			write!(f, ")")?;

			write!(f, " ")?;
			decl.ret.pprint(f)?;

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
			write!(f, "type ")?;
			name.sym.pprint(f)?;
			if let Some(alias) = alias {
				write!(f, " = ")?;
				alias.pprint(f)?;
			}
			write!(f, ";")?;
			Ok(())
		}
	}

	impl PrettyPrint for hir::FieldDef {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self { name, ty } = &self;
			name.sym.pprint(f)?;
			write!(f, ": ")?;
			ty.pprint(f)?;
			Ok(())
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
			f.newline()?;
			write!(f, "}}")?;
			Ok(())
		}
	}

	impl PrettyPrint for hir::Stmt {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self {
				kind,
				span: _,
				id: _,
			} = &self;
			kind.pprint(f)?;
			Ok(())
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
					name.sym.pprint(f)?;
					write!(f, ": ")?;
					ty.pprint(f)?;
					write!(f, " = ")?;
					value.pprint(f)?;
					write!(f, ";")?;
				}
				Self::Expr { expr } => {
					expr.pprint(f)?;
					write!(f, ";")?;
				}
			}
			Ok(())
		}
	}

	impl PrettyPrint for hir::Expr {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self {
				kind,
				span: _,
				id: _,
			} = &self;
			kind.pprint(f)?;
			Ok(())
		}
	}

	impl PrettyPrint for hir::ExprKind {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			// TODO: parenthesize ambiguous expressions
			match &self {
				Self::Access { path } => path.pprint(f)?,
				Self::LiteralStr { sym } => {
					write!(f, "\"")?;
					sym.pprint(f)?;
					write!(f, "\"")?;
				}
				Self::LiteralInt { sym } | Self::LiteralFloat { sym } => {
					sym.pprint(f)?;
				}

				Self::Unit => {
					write!(f, "()")?;
				}

				Self::Unary { op, expr } => {
					op.bit.pprint(f)?;
					expr.pprint(f)?;
				}
				Self::Binary { op, left, right } => {
					left.pprint(f)?;
					write!(f, " ")?;
					op.bit.pprint(f)?;
					write!(f, " ")?;
					right.pprint(f)?;
				}

				Self::FnCall { expr, args } => {
					expr.pprint(f)?;
					write!(f, "(")?;
					f.write_seq_oneline(&args.bit, |f, arg| arg.pprint(f), ",")?;
					write!(f, ")")?;
				}
				Self::If {
					cond,
					conseq,
					altern,
				} => {
					write!(f, "if ")?;
					cond.pprint(f)?;
					write!(f, " ")?;
					conseq.pprint(f)?;
					if let Some(altern) = altern {
						write!(f, " else ")?;
						altern.pprint(f)?;
					}
				}
				Self::Loop { block } => {
					write!(f, "loop ")?;
					block.pprint(f)?;
				}

				Self::Method { expr, name, params } => {
					expr.pprint(f)?;
					write!(f, ".")?;
					name.sym.pprint(f)?;
					write!(f, "(")?;
					f.write_seq_oneline(params, |f, param| param.pprint(f), ",")?;
					write!(f, ")")?;
				}
				Self::Field { expr, name } => {
					expr.pprint(f)?;
					write!(f, ".")?;
					name.sym.pprint(f)?;
				}
				Self::Deref { expr } => {
					expr.pprint(f)?;
					write!(f, ".*")?;
				}
				Self::Assign { target, value } => {
					target.pprint(f)?;
					write!(f, " = ")?;
					value.pprint(f)?;
				}
				Self::Return { expr } => {
					write!(f, "return ")?;
					expr.pprint(f)?;
				}
				Self::Break { expr, label } => {
					write!(f, "break ")?;
					expr.pprint(f)?;
				}
				Self::Continue { label } => write!(f, "continue")?,
			}
			Ok(())
		}
	}
}
