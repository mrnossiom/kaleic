//! Pretty print a source file.
//!
//! This essentially prints the AST taking line length into account. It also
//! reads content that not in the AST like comments to not lose any data.

#![expect(unused_variables, clippy::todo)]

use std::io::{self, Write, stdout};

use crate::{
	ast::{
		BinaryOp, Block, Expr, ExprKind, FieldDef, Function, Item, ItemKind, Param, Path, Root,
		Stmt, StmtKind, Ty, TyKind, TypeAlias, UnaryOp, VariantKind,
	},
	lexer::LiteralKind,
	session::Symbol,
};

type Result<T> = std::result::Result<T, io::Error>;

pub struct PrettyFormatter<'fmt> {
	inner: &'fmt mut dyn Write,

	ident: u32,
}

impl<'fmt> PrettyFormatter<'fmt> {
	fn new(inner: &'fmt mut dyn Write) -> Self {
		Self { inner, ident: 0 }
	}

	fn with_ident(&mut self, f: impl FnOnce(&mut PrettyFormatter) -> Result<()>) -> Result<()> {
		self.ident += 1;
		f(self)?;
		self.ident -= 1;
		Ok(())
	}

	fn newline(&mut self) -> io::Result<()> {
		self.inner.write_all(b"\n")?;
		for _ in 0..self.ident {
			self.inner.write_all(b"\t")?;
		}

		Ok(())
	}

	fn write(&mut self, s: &str) -> io::Result<()> {
		self.inner.write_all(s.as_bytes())?;
		Ok(())
	}

	fn write_seq<T>(
		&mut self,
		elements: &[T],
		mut print: impl FnMut(&mut Self, &T) -> Result<()>,
		sep: &str,
	) -> Result<()> {
		for (i, elem) in elements.iter().enumerate() {
			self.newline()?;
			print(self, elem)?;
			self.write(sep)?;
		}
		Ok(())
	}

	fn write_seq_oneline<T>(
		&mut self,
		elements: &[T],
		mut print: impl FnMut(&mut Self, &T) -> Result<()>,
		sep: &str,
	) -> Result<()> {
		for (i, elem) in elements.iter().enumerate() {
			if i != 0 {
				self.write(sep)?;
				self.write(" ")?;
			}
			print(self, elem)?;
		}
		Ok(())
	}
}

// impl<'fmt> fmt::Write for PrettyFormatter<'fmt> {}

pub fn pretty_print_root(root: &Root, mut output: &mut dyn Write) -> Result<()> {
	let mut f = PrettyFormatter::new(&mut output);
	root.pprint(&mut f)?;
	Ok(())
}

pub fn pretty_print_item(item: &Item) -> Result<()> {
	let mut output = &stdout();
	let mut f = PrettyFormatter::new(&mut output);
	item.pprint(&mut f)?;
	Ok(())
}

trait PrettyPrint {
	fn pprint(&self, f: &mut PrettyFormatter) -> Result<()>;
}

impl PrettyPrint for Root {
	fn pprint(&self, f: &mut PrettyFormatter) -> Result<()> {
		for item in &self.items {
			item.pprint(f)?;
			f.newline()?;

			f.newline()?;
		}
		Ok(())
	}
}

impl PrettyPrint for Item {
	fn pprint(&self, f: &mut PrettyFormatter) -> Result<()> {
		match &self.kind {
			ItemKind::Function(func) => func.pprint(f)?,
			ItemKind::TypeAlias(ty) => ty.pprint(f)?,

			ItemKind::Struct {
				name,
				generics,
				fields,
			} => {
				f.write("struct ")?;
				name.sym.pprint(f)?;
				if !generics.is_empty() {
					f.write("<")?;
					f.write_seq_oneline(generics, |f, generic| generic.sym.pprint(f), ",")?;
					f.write(">")?;
				}
				f.write(" {")?;
				f.with_ident(|f| f.write_seq(fields, |f, variant| variant.pprint(f), ","))?;
				f.newline()?;
				f.write("}")?;
			}
			ItemKind::Enum {
				name,
				generics,
				variants,
			} => {
				f.write("enum ")?;
				name.sym.pprint(f)?;
				if !generics.is_empty() {
					f.write("<")?;
					f.write_seq_oneline(generics, |f, generic| generic.sym.pprint(f), ",")?;
					f.write(">")?;
				}
				f.write(" {")?;
				f.with_ident(|f| {
					f.write_seq(
						variants,
						|f, variant| {
							variant.name.sym.pprint(f)?;
							match &variant.kind {
								VariantKind::Bare => {}
								VariantKind::Tuple(fields) => {
									f.write("(")?;
									f.write_seq_oneline(fields, |f, field| field.pprint(f), ",")?;
									f.write(")")?;
								}
								VariantKind::Struct(fields) => {
									f.write(" { ")?;
									f.write_seq_oneline(fields, |f, field| field.pprint(f), ",")?;
									f.write(" }")?;
								}
							}
							Ok(())
						},
						",",
					)
				})?;
				f.newline()?;
				f.write("}")?;
			}
			ItemKind::Trait { .. } => todo!(),
			ItemKind::TraitImpl { .. } => todo!(),
		}
		Ok(())
	}
}

impl PrettyPrint for FieldDef {
	fn pprint(&self, f: &mut PrettyFormatter) -> Result<()> {
		self.name.sym.pprint(f)?;
		f.write(": ")?;
		self.ty.pprint(f)?;
		Ok(())
	}
}

impl PrettyPrint for Function {
	fn pprint(&self, f: &mut PrettyFormatter) -> Result<()> {
		let Self {
			name,
			decl,
			body,
			abi,
		} = &self;

		if let Some(abi) = &abi {
			f.write("extern ")?;
			abi.pprint(f)?;
			f.write(" ")?;
		}

		f.write("fn ")?;
		name.sym.pprint(f)?;
		f.write("(")?;
		f.write_seq_oneline(&decl.params, |f, param| param.pprint(f), ", ")?;
		f.write(")")?;

		if let Some(ret) = &decl.ret {
			f.write(" ")?;
			ret.pprint(f)?;
		}

		if let Some(body) = &body {
			f.write(" ")?;
			body.pprint(f)?;
		} else {
			f.write(";")?;
		}

		Ok(())
	}
}

impl PrettyPrint for TypeAlias {
	fn pprint(&self, f: &mut PrettyFormatter) -> Result<()> {
		f.write("type ")?;
		self.name.sym.pprint(f)?;
		if let Some(alias) = &self.alias {
			f.write(" = ")?;
			alias.pprint(f)?;
		}
		f.write(";")?;
		Ok(())
	}
}

impl PrettyPrint for Expr {
	fn pprint(&self, f: &mut PrettyFormatter) -> Result<()> {
		match &self.kind {
			ExprKind::Access(path) => path.pprint(f),
			ExprKind::Literal(kind, sym) => match kind {
				LiteralKind::Integer | LiteralKind::Float => sym.pprint(f),
				LiteralKind::Str => {
					f.write("\"")?;
					sym.pprint(f)?;
					f.write("\"")?;
					Ok(())
				}
			},

			ExprKind::Paren(expr) => {
				f.write("(")?;
				expr.pprint(f)?;
				f.write(")")?;
				Ok(())
			}
			ExprKind::Unary { op, expr } => {
				op.bit.pprint(f)?;
				expr.pprint(f)?;
				Ok(())
			}
			ExprKind::Binary { op, left, right } => {
				left.pprint(f)?;
				f.write(" ")?;
				op.bit.pprint(f)?;
				f.write(" ")?;
				right.pprint(f)?;
				Ok(())
			}

			ExprKind::FnCall { expr, args } => {
				expr.pprint(f)?;
				f.write("(")?;
				f.write_seq_oneline(&args.bit, |f, arg| arg.pprint(f), ",")?;
				f.write(")")?;
				Ok(())
			}
			ExprKind::If {
				cond,
				conseq,
				altern,
			} => {
				f.write("if ")?;
				cond.pprint(f)?;
				f.write(" ")?;
				conseq.pprint(f)?;
				if let Some(altern) = altern {
					f.write(" else ")?;
					altern.pprint(f)?;
				}
				Ok(())
			}
			ExprKind::Method(expr, name, args) => {
				expr.pprint(f)?;
				f.write(".")?;
				name.sym.pprint(f)?;
				f.write("(")?;
				f.write_seq_oneline(args, |f, arg| arg.pprint(f), ",")?;
				f.write(")")?;
				Ok(())
			}
			ExprKind::Field(expr, name) => {
				expr.pprint(f)?;
				f.write(".")?;
				name.sym.pprint(f)?;
				Ok(())
			}
			ExprKind::Deref(expr) => {
				expr.pprint(f)?;
				f.write(".*")?;
				Ok(())
			}
			ExprKind::Assign { target, value } => {
				target.pprint(f)?;
				f.write(" = ")?;
				value.pprint(f)?;
				Ok(())
			}
			ExprKind::Return(expr) => {
				f.write("return")?;
				if let Some(expr) = expr {
					expr.pprint(f)?;
				}
				Ok(())
			}
			ExprKind::Break(expr) => {
				f.write("break")?;
				if let Some(expr) = expr {
					expr.pprint(f)?;
				}
				Ok(())
			}
			ExprKind::Continue => f.write("continue"),
		}
	}
}

impl PrettyPrint for Param {
	fn pprint(&self, f: &mut PrettyFormatter) -> Result<()> {
		self.name.sym.pprint(f)?;
		f.write(": ")?;
		self.ty.pprint(f)?;
		Ok(())
	}
}

impl PrettyPrint for Ty {
	fn pprint(&self, f: &mut PrettyFormatter) -> Result<()> {
		match &self.kind {
			TyKind::Path(path) => path.pprint(f)?,

			TyKind::Pointer(ty) => {
				f.write("&")?;
				ty.pprint(f)?;
			}
			TyKind::Unit => f.write("()")?,
			TyKind::ImplicitInfer => f.write("_")?,
		}
		Ok(())
	}
}

impl PrettyPrint for Symbol {
	fn pprint(&self, f: &mut PrettyFormatter) -> Result<()> {
		// TODO
		f.write(&format!("{self:#?}"))
	}
}

impl PrettyPrint for Path {
	fn pprint(&self, f: &mut PrettyFormatter) -> Result<()> {
		f.write_seq_oneline(&self.segments, |f, segment| segment.sym.pprint(f), "::")?;
		if !self.generics.is_empty() {
			f.write("<")?;
			f.write_seq_oneline(&self.generics, |f, generic| generic.pprint(f), ", ")?;
			f.write(">")?;
		}
		Ok(())
	}
}

impl PrettyPrint for Block {
	fn pprint(&self, f: &mut PrettyFormatter) -> Result<()> {
		f.write("{")?;
		f.with_ident(|f| {
			for stmt in &self.stmts {
				f.newline()?;
				stmt.pprint(f)?;
			}

			Ok(())
		})?;
		f.newline()?;
		f.write("}")?;
		Ok(())
	}
}

impl PrettyPrint for Stmt {
	fn pprint(&self, f: &mut PrettyFormatter) -> Result<()> {
		match &self.kind {
			StmtKind::Loop { body } => {
				f.write("loop ")?;
				body.pprint(f)?;
			}
			StmtKind::WhileLoop { check, body } => {
				f.write("while ")?;
				check.pprint(f)?;
				f.write(" ")?;
				body.pprint(f)?;
			}

			StmtKind::Let { name, ty, value } => {
				f.write("let ")?;
				name.sym.pprint(f)?;
				if let Some(ty) = &ty {
					f.write(": ")?;
					ty.pprint(f)?;
				}
				f.write(" = ")?;
				value.pprint(f)?;
				f.write(";")?;
			}

			StmtKind::Empty => f.write("; empty stmt")?,
			StmtKind::Expr(expr) => {
				expr.pprint(f)?;
				f.write(";")?;
			}
			StmtKind::ExprRet(expr) => expr.pprint(f)?,
		}
		Ok(())
	}
}

impl PrettyPrint for UnaryOp {
	fn pprint(&self, f: &mut PrettyFormatter) -> Result<()> {
		match self {
			Self::Not => f.write("!"),
			Self::Minus => f.write("-"),
		}
	}
}

impl PrettyPrint for BinaryOp {
	fn pprint(&self, f: &mut PrettyFormatter) -> Result<()> {
		match self {
			Self::Plus => f.write("+"),
			Self::Minus => f.write("-"),
			Self::Mul => f.write("*"),
			Self::Div => f.write("/"),
			Self::Mod => f.write("%"),

			Self::And => f.write("&"),
			Self::Or => f.write("|"),
			Self::Xor => f.write("^"),

			Self::Shl => f.write("<<"),
			Self::Shr => f.write(">>"),

			Self::Gt => f.write(">"),
			Self::Ge => f.write(">="),
			Self::Lt => f.write("<"),
			Self::Le => f.write("<="),

			Self::Ne => f.write("!="),
			Self::EqEq => f.write("=="),
		}
	}
}
