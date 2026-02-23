//! Pretty print a source file.
//!
//! This essentially prints the AST taking line length into account. It also
//! reads content that not in the AST like comments to not lose any data.

#![expect(unused_variables, clippy::todo)]

use std::fmt::{self, Write as _};

use crate::{lexer::LiteralKind, session::Symbol};

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
		write!(self.inner, "\n")?;
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

// annoying to seal but should be
pub trait PrettyPrint {
	fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result;
}

mod ast_pp {
	use super::*;

	use crate::ast::{
		BinaryOp, Block, Expr, ExprKind, FieldDef, Function, Item, ItemKind, Param, Path, Root,
		ShortCircuitOp, Stmt, StmtKind, Ty, TyKind, TypeAlias, UnaryOp, VariantKind,
	};

	impl PrettyPrint for Root {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			for item in &self.items {
				item.pprint(f)?;
				f.newline()?;

				f.newline()?;
			}
			Ok(())
		}
	}

	impl PrettyPrint for Item {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match &self.kind {
				ItemKind::Function(func) => func.pprint(f)?,
				ItemKind::TypeAlias(ty) => ty.pprint(f)?,

				ItemKind::Struct {
					name,
					generics,
					fields,
				} => {
					write!(f, "struct ")?;
					name.sym.pprint(f)?;
					if !generics.is_empty() {
						write!(f, "<")?;
						f.write_seq_oneline(generics, |f, generic| generic.sym.pprint(f), ",")?;
						write!(f, ">")?;
					}
					write!(f, " {{")?;
					f.with_indent(|f| f.write_seq(fields, |f, variant| variant.pprint(f), ","))?;
					f.newline()?;
					write!(f, "}}")?;
				}
				ItemKind::Enum {
					name,
					generics,
					variants,
				} => {
					write!(f, "enum ")?;
					name.sym.pprint(f)?;
					if !generics.is_empty() {
						write!(f, "<")?;
						f.write_seq_oneline(generics, |f, generic| generic.sym.pprint(f), ",")?;
						write!(f, ">")?;
					}
					write!(f, " {{")?;
					f.with_indent(|f| {
						f.write_seq(
							variants,
							|f, variant| {
								variant.name.sym.pprint(f)?;
								match &variant.kind {
									VariantKind::Bare => {}
									VariantKind::Tuple(fields) => {
										write!(f, "(")?;
										f.write_seq_oneline(
											fields,
											|f, field| field.pprint(f),
											",",
										)?;
										write!(f, ")")?;
									}
									VariantKind::Struct(fields) => {
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
				ItemKind::Trait { .. } => todo!(),
				ItemKind::TraitImpl { .. } => todo!(),
			}
			Ok(())
		}
	}

	impl PrettyPrint for FieldDef {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			self.name.sym.pprint(f)?;
			write!(f, ": ")?;
			self.ty.pprint(f)?;
			Ok(())
		}
	}

	impl PrettyPrint for Function {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self {
				name,
				decl,
				body,
				abi,
			} = &self;

			if let Some(abi) = &abi {
				write!(f, "extern ")?;
				abi.pprint(f)?;
				write!(f, " ")?;
			}

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

	impl PrettyPrint for TypeAlias {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			write!(f, "type ")?;
			self.name.sym.pprint(f)?;
			if let Some(alias) = &self.alias {
				write!(f, " = ")?;
				alias.pprint(f)?;
			}
			write!(f, ";")?;
			Ok(())
		}
	}

	impl PrettyPrint for Expr {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match &self.kind {
				ExprKind::Access { path } => path.pprint(f),
				ExprKind::Literal { lit, sym } => match lit {
					LiteralKind::Integer | LiteralKind::Float => sym.pprint(f),
					LiteralKind::Str => {
						write!(f, "\"")?;
						sym.pprint(f)?;
						write!(f, "\"")?;
						Ok(())
					}
				},

				ExprKind::Paren { expr } => {
					write!(f, "(")?;
					expr.pprint(f)?;
					write!(f, ")")?;
					Ok(())
				}
				ExprKind::Unary { op, expr } => {
					op.bit.pprint(f)?;
					expr.pprint(f)?;
					Ok(())
				}
				ExprKind::Binary { op, left, right } => {
					left.pprint(f)?;
					write!(f, " ")?;
					op.bit.pprint(f)?;
					write!(f, " ")?;
					right.pprint(f)?;
					Ok(())
				}
				ExprKind::ShortCircuit { op, left, right } => {
					left.pprint(f)?;
					write!(f, " ")?;
					op.bit.pprint(f)?;
					write!(f, " ")?;
					right.pprint(f)?;
					Ok(())
				}

				ExprKind::FnCall { expr, args } => {
					expr.pprint(f)?;
					write!(f, "(")?;
					f.write_seq_oneline(&args.bit, |f, arg| arg.pprint(f), ",")?;
					write!(f, ")")?;
					Ok(())
				}
				ExprKind::If {
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
					Ok(())
				}
				ExprKind::Match { expr, arms } => todo!(),

				ExprKind::Method { expr, name, params } => {
					expr.pprint(f)?;
					write!(f, ".")?;
					name.sym.pprint(f)?;
					write!(f, "(")?;
					f.write_seq_oneline(params, |f, param| param.pprint(f), ",")?;
					write!(f, ")")?;
					Ok(())
				}
				ExprKind::Field { expr, name } => {
					expr.pprint(f)?;
					write!(f, ".")?;
					name.sym.pprint(f)?;
					Ok(())
				}
				ExprKind::Deref { expr } => {
					expr.pprint(f)?;
					write!(f, ".*")?;
					Ok(())
				}
				ExprKind::Assign { target, value } => {
					target.pprint(f)?;
					write!(f, " = ")?;
					value.pprint(f)?;
					Ok(())
				}
				ExprKind::Return { expr } => {
					write!(f, "return")?;
					if let Some(expr) = expr {
						expr.pprint(f)?;
					}
					Ok(())
				}
				ExprKind::Break { expr, label } => {
					write!(f, "break")?;
					if let Some(expr) = expr {
						expr.pprint(f)?;
					}
					Ok(())
				}
				ExprKind::Continue { label } => write!(f, "continue"),
			}
		}
	}

	impl PrettyPrint for Param {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			self.name.sym.pprint(f)?;
			write!(f, ": ")?;
			self.ty.pprint(f)?;
			Ok(())
		}
	}

	impl PrettyPrint for Ty {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match &self.kind {
				TyKind::Path(path) => path.pprint(f)?,

				TyKind::Pointer(ty) => {
					write!(f, "&")?;
					ty.pprint(f)?;
				}
				TyKind::Unit => write!(f, "()")?,
				TyKind::ImplicitInfer => write!(f, "_")?,
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

	impl PrettyPrint for Path {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			f.write_seq_oneline(&self.segments, |f, segment| segment.sym.pprint(f), "::")?;
			if !self.generics.is_empty() {
				write!(f, "<")?;
				f.write_seq_oneline(&self.generics, |f, generic| generic.pprint(f), ", ")?;
				write!(f, ">")?;
			}
			Ok(())
		}
	}

	impl PrettyPrint for Block {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			write!(f, "{{")?;
			f.with_indent(|f| {
				for stmt in &self.stmts {
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

	impl PrettyPrint for Stmt {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match &self.kind {
				StmtKind::Loop { body } => {
					write!(f, "loop ")?;
					body.pprint(f)?;
				}
				StmtKind::WhileLoop { check, body } => {
					write!(f, "while ")?;
					check.pprint(f)?;
					write!(f, " ")?;
					body.pprint(f)?;
				}

				StmtKind::Let {
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

				StmtKind::Empty => write!(f, "; empty stmt")?,
				StmtKind::Expr(expr) => {
					expr.pprint(f)?;
					write!(f, ";")?;
				}
				StmtKind::ExprRet(expr) => expr.pprint(f)?,
			}
			Ok(())
		}
	}

	impl PrettyPrint for UnaryOp {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match self {
				Self::Not => write!(f, "!"),
				Self::Minus => write!(f, "-"),
			}
		}
	}

	impl PrettyPrint for BinaryOp {
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

	impl PrettyPrint for ShortCircuitOp {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match self {
				Self::And => write!(f, "and"),
				Self::Or => write!(f, "or"),
			}
		}
	}
}

mod hir_pp {
	use super::*;

	use crate::hir::{
		Abi, Block, Enum, Expr, ExprKind, FieldDef, Function, Item, ItemKind, Root, Stmt, StmtKind,
		Struct, TypeAlias,
	};

	impl PrettyPrint for Root {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			for item in &self.items {
				item.pprint(f)?;
				f.newline()?;

				f.newline()?;
			}
			Ok(())
		}
	}

	impl PrettyPrint for Item {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match &self.kind {
				ItemKind::Function(func) => func.pprint(f)?,
				ItemKind::TypeAlias(ty) => ty.pprint(f)?,

				ItemKind::Struct(Struct {
					name,
					generics,
					fields,
				}) => {
					write!(f, "struct ")?;
					name.sym.pprint(f)?;
					if !generics.is_empty() {
						write!(f, "<")?;
						f.write_seq_oneline(generics, |f, generic| generic.sym.pprint(f), ",")?;
						write!(f, ">")?;
					}
					write!(f, " {{")?;
					f.with_indent(|f| f.write_seq(fields, |f, variant| variant.pprint(f), ","))?;
					f.newline()?;
					write!(f, "}}")?;
				}
				ItemKind::Enum(Enum {
					name,
					generics,
					variants,
				}) => {
					write!(f, "enum ")?;
					name.sym.pprint(f)?;
					if !generics.is_empty() {
						write!(f, "<")?;
						f.write_seq_oneline(generics, |f, generic| generic.sym.pprint(f), ",")?;
						write!(f, ">")?;
					}
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
				ItemKind::Trait { .. } => todo!(),
				ItemKind::TraitImpl { .. } => todo!(),
			}
			Ok(())
		}
	}

	impl PrettyPrint for Function {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			let Self {
				name,
				decl,
				body,
				abi,
			} = &self;

			write!(f, "fn ")?;
			abi.pprint(f)?;
			write!(f, " ")?;
			name.sym.pprint(f)?;
			write!(f, "(")?;
			f.write_seq_oneline(&decl.inputs, |f, param| param.pprint(f), ", ")?;
			write!(f, ")")?;

			write!(f, " ")?;
			decl.output.pprint(f)?;

			if let Some(body) = &body {
				write!(f, " ")?;
				body.pprint(f)?;
			} else {
				write!(f, ";")?;
			}

			Ok(())
		}
	}

	impl PrettyPrint for Abi {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match self {
				Self::Kalei => write!(f, "kalei"),
				Self::C => write!(f, "c"),
			}
		}
	}

	impl PrettyPrint for TypeAlias {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			write!(f, "type ")?;
			self.name.sym.pprint(f)?;
			if let Some(alias) = &self.alias {
				write!(f, " = ")?;
				alias.pprint(f)?;
			}
			write!(f, ";")?;
			Ok(())
		}
	}

	impl PrettyPrint for FieldDef {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			self.name.sym.pprint(f)?;
			write!(f, ": ")?;
			self.ty.pprint(f)?;
			Ok(())
		}
	}

	impl PrettyPrint for Block {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			write!(f, "{{")?;
			f.with_indent(|f| {
				for stmt in &self.stmts {
					f.newline()?;
					stmt.pprint(f)?;
				}
				if let Some(expr) = &self.ret {
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

	impl PrettyPrint for Stmt {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			match &self.kind {
				StmtKind::Loop { block } => {
					write!(f, "loop ")?;
					block.pprint(f)?;
				}
				StmtKind::Let {
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
				StmtKind::Expr { expr } => {
					expr.pprint(f)?;
					write!(f, ";")?;
				}
			}
			Ok(())
		}
	}

	impl PrettyPrint for Expr {
		fn pprint(&self, f: &mut PrettyFormatter) -> fmt::Result {
			// TODO: parenthesize ambiguous expressions
			match &self.kind {
				ExprKind::Access { path } => path.pprint(f),
				ExprKind::Literal { lit, sym } => match lit {
					LiteralKind::Integer | LiteralKind::Float => sym.pprint(f),
					LiteralKind::Str => {
						write!(f, "\"")?;
						sym.pprint(f)?;
						write!(f, "\"")?;
						Ok(())
					}
				},

				ExprKind::Unary { op, expr } => {
					op.bit.pprint(f)?;
					expr.pprint(f)?;
					Ok(())
				}
				ExprKind::Binary { op, left, right } => {
					left.pprint(f)?;
					write!(f, " ")?;
					op.bit.pprint(f)?;
					write!(f, " ")?;
					right.pprint(f)?;
					Ok(())
				}

				ExprKind::FnCall { expr, args } => {
					expr.pprint(f)?;
					write!(f, "(")?;
					f.write_seq_oneline(&args.bit, |f, arg| arg.pprint(f), ",")?;
					write!(f, ")")?;
					Ok(())
				}
				ExprKind::If {
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
					Ok(())
				}
				ExprKind::Method { expr, name, params } => {
					expr.pprint(f)?;
					write!(f, ".")?;
					name.sym.pprint(f)?;
					write!(f, "(")?;
					f.write_seq_oneline(params, |f, param| param.pprint(f), ",")?;
					write!(f, ")")?;
					Ok(())
				}
				ExprKind::Field { expr, name } => {
					expr.pprint(f)?;
					write!(f, ".")?;
					name.sym.pprint(f)?;
					Ok(())
				}
				ExprKind::Deref { expr } => {
					expr.pprint(f)?;
					write!(f, ".*")?;
					Ok(())
				}
				ExprKind::Assign { target, value } => {
					target.pprint(f)?;
					write!(f, " = ")?;
					value.pprint(f)?;
					Ok(())
				}
				ExprKind::Return { expr } => {
					write!(f, "return")?;
					if let Some(expr) = expr {
						expr.pprint(f)?;
					}
					Ok(())
				}
				ExprKind::Break { expr, label } => {
					write!(f, "break")?;
					if let Some(expr) = expr {
						expr.pprint(f)?;
					}
					Ok(())
				}
				ExprKind::Continue { label } => write!(f, "continue"),
			}
		}
	}
}
