//! Tokens to AST parsing logic
//!
//! Contains the recursive decent parser of the language.
//!
//! Entrypoint to parsing is [`parse_root`].

use std::{
	fmt::{self, Write},
	mem,
	ops::ControlFlow,
	path::{self, PathBuf},
	sync::atomic::{AtomicU32, Ordering},
};

use ariadne::{Label, Report, ReportKind};

#[allow(
	clippy::enum_glob_use,
	reason = "single glob usage here, reduces code size"
)]
use crate::lexer::TokenKind::*;
use crate::{
	ast::{
		self, Attr, AttrKind, AttrMeta, AttrPath, BinaryOp, Block, Expr, ExprKind, FieldDef,
		FnDecl, Function, GenericParams, Generics, Ident as Id, Item, ItemKind, Param, Path,
		PathSegment, Root, ShortCircuitOp, Spanned, Stmt, StmtKind, Ty, TyKind, TypeAlias, UnaryOp,
		Variant, VariantKind,
	},
	attrs::{ModPath, NoCore, NoStd, try_parse_attr},
	lexer::{Lexer, Token, TokenKind},
	pretty_print,
	session::{ArtefactKind, DcxHandle, Diagnostic, SessionCtx, SourceFile, Span},
	symbols::{Symbol, kw, sym},
};

pub(crate) fn parse_root(scx: &SessionCtx, source: &SourceFile) -> Root {
	let module = ModuleData {
		mod_path: Vec::new(),
		filepath_stack: Vec::new(),
		current_dir: source.path.parent().unwrap().to_path_buf(),
	};
	let mut p = Parser::new(scx, source, module);
	let ast = match p.parse_root() {
		Ok(ast) => ast,
		Err(diag) => scx.dcx().emit_fatal(&diag),
	};

	scx.register_artefact(&ArtefactKind::Ast(()), |artefact| {
		write!(artefact, "{ast:#?}")
	});
	scx.register_artefact(&ArtefactKind::AstPretty(()), |artefact| {
		pretty_print::pretty_print(&ast, artefact)
	});

	ast
}

type Result<T> = std::result::Result<T, Diagnostic>;

#[derive(Debug)]
enum AssocOp {
	Binary(BinaryOp),
	ShortCircuit(ShortCircuitOp),
	Assign,
}

impl AssocOp {
	fn from_token_kind(kind: TokenKind) -> Option<Self> {
		let kind = match kind {
			Plus => Self::Binary(BinaryOp::Plus),
			Dash => Self::Binary(BinaryOp::Minus),
			Star => Self::Binary(BinaryOp::Mul),
			Div => Self::Binary(BinaryOp::Div),
			Mod => Self::Binary(BinaryOp::Mod),
			BitwiseAnd => Self::Binary(BinaryOp::And),
			BitwiseOr => Self::Binary(BinaryOp::Or),
			Xor => Self::Binary(BinaryOp::Xor),
			Shl => Self::Binary(BinaryOp::Shl),
			Shr => Self::Binary(BinaryOp::Shr),
			Gt => Self::Binary(BinaryOp::Gt),
			Ge => Self::Binary(BinaryOp::Ge),
			Lt => Self::Binary(BinaryOp::Lt),
			Le => Self::Binary(BinaryOp::Le),
			EqEq => Self::Binary(BinaryOp::EqEq),
			Ne => Self::Binary(BinaryOp::Ne),
			Eq => Self::Assign,
			Kw(kw::And) => Self::ShortCircuit(ShortCircuitOp::And),
			Kw(kw::Or) => Self::ShortCircuit(ShortCircuitOp::Or),
			_ => return None,
		};
		Some(kind)
	}

	fn precedence(&self) -> u32 {
		match self {
			Self::Binary(op) => match op {
				BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => 48,
				BinaryOp::Minus | BinaryOp::Plus => 40,
				BinaryOp::Shl | BinaryOp::Shr => 32,
				BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => 24,
				BinaryOp::Gt | BinaryOp::Ge | BinaryOp::Lt | BinaryOp::Le => 16,
				BinaryOp::Ne | BinaryOp::EqEq => 8,
			},
			Self::ShortCircuit(op) => match op {
				ShortCircuitOp::And | ShortCircuitOp::Or => 4,
			},
			Self::Assign => 0,
		}
	}
}

struct Parser<'scx> {
	scx: &'scx SessionCtx,

	lexer: Lexer<'scx, 'scx>,

	token: Token,
	last_token: Token,

	module: ModuleData,
}

#[derive(Debug, Clone)]
struct ModuleData {
	mod_path: Vec<Symbol>,
	/// Used to detect import cycles, usually when used with `#path`
	filepath_stack: Vec<PathBuf>,
	current_dir: PathBuf,
}

impl ModuleData {
	fn with_file(&self, name: Symbol, filepath: PathBuf) -> Self {
		let Self {
			mut mod_path,
			mut filepath_stack,
			..
		} = (*self).clone();

		if filepath_stack.contains(&filepath) {
			todo!("cycle detected")
		}

		mod_path.push(name);
		let current_dir = filepath.parent().unwrap().to_path_buf();
		filepath_stack.push(filepath);

		Self {
			mod_path,
			filepath_stack,
			current_dir,
		}
	}
}

impl<'scx> Parser<'scx> {
	fn new(scx: &'scx SessionCtx, file: &'scx SourceFile, module: ModuleData) -> Self {
		let SourceFile {
			path,
			content,
			offset,
		} = &file;

		let mut parser = Self {
			scx,

			lexer: Lexer::new(scx, content, *offset),

			token: Token::DUMMY,
			last_token: Token::DUMMY,

			module,
		};

		// init the first token
		parser.bump();

		parser
	}

	// TODO: find more elegany way to expand multiple files
	fn make_node_id(&self) -> ast::NodeId {
		static NEXT_NODE_ID: AtomicU32 = AtomicU32::new(0);
		let _ = self;
		ast::NodeId::new(NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed))
	}
}

/// Helper methods
impl Parser<'_> {
	fn bump(&mut self) {
		self.last_token = mem::replace(&mut self.token, self.lexer.next().unwrap_or(Token::DUMMY));
	}

	fn check(&self, token: TokenKind) -> bool {
		self.token.kind == token
	}

	fn eat(&mut self, token: TokenKind) -> bool {
		if self.check(token) {
			self.bump();
			true
		} else {
			false
		}
	}

	fn eat_kw(&mut self, kw: Symbol) -> bool {
		debug_assert!(kw.is_keyword());
		self.eat(Kw(kw))
	}

	#[track_caller]
	fn expect(&mut self, expected_kind: TokenKind) -> Result<Token> {
		if self.check(expected_kind) {
			self.bump();
			Ok(self.token)
		} else {
			let report = errors::expected_token_kind(expected_kind, self.token);
			Err(Diagnostic::new(report))
		}
	}

	fn eat_ident(&mut self) -> Option<Id> {
		self.token.as_ident().inspect(|_| {
			self.bump();
		})
	}

	fn expect_ident(&mut self) -> Result<Id> {
		self.eat_ident().ok_or_else(|| {
			let placeholder = self.scx.symbols.intern("_");
			let report = errors::expected_token_kind(Ident(placeholder), self.token);
			Diagnostic::new(report)
		})
	}

	fn close_span(&self, start: Span) -> Span {
		start.to(self.last_token.span)
	}

	/// Parse a separated sequence with a starting and ending token
	///
	/// e.g. `(foo, bar,)`, `(bar, baz)`
	fn parse_seq_rest<T: fmt::Debug>(
		&mut self,
		start: TokenKind,
		end: TokenKind,
		separator: TokenKind,
		mut parse: impl FnMut(&mut Self) -> Result<T>,
	) -> Result<Vec<T>> {
		debug_assert_eq!(self.last_token.kind, start);

		let mut finished = false;
		let mut seq = Vec::new();

		while !self.eat(end) && !finished {
			seq.push(parse(self)?);

			// no comma means no item left
			finished = !self.eat(separator);
		}

		Ok(seq)
	}

	fn parse_until<T>(
		&mut self,
		end: TokenKind,
		mut parse: impl FnMut(&mut Self) -> Result<T>,
	) -> Result<Vec<T>> {
		let mut many = Vec::new();
		while !self.eat(end) {
			many.push(parse(self)?);
		}
		Ok(many)
	}

	/// Keep parsing using `parse` while we can eat `start`
	fn parse_while<T>(
		&mut self,
		start: TokenKind,
		mut parse: impl FnMut(&mut Self) -> Result<T>,
	) -> Result<Vec<T>> {
		let mut many = Vec::new();
		while self.eat(start) {
			many.push((parse(self))?);
		}
		Ok(many)
	}

	/// Looks one token ahead
	#[expect(dead_code)]
	fn look_ahead(&self) -> TokenKind {
		self.lexer.clone().next().map_or(Eof, |tkn| tkn.kind)
	}
}

impl Parser<'_> {
	fn parse_root(&mut self) -> Result<Root> {
		let (attrs, items) = self.parse_module_inner()?;

		let mut expanded_items = Vec::new();
		expanded_items.extend(self.core_libs_import(&attrs));
		expanded_items.extend(items);

		Ok(Root {
			attrs,
			items: expanded_items,
		})
	}

	/// Make an `extern use` item for `std`, `core` or none
	/// with respect to `no_std` and `no_core` attributes
	fn core_libs_import(&self, root_attrs: &[Attr]) -> Option<Item> {
		enum AutoImportState {
			None,
			NoStd(Span),
			NoCore(Span),
		}

		let mut state = AutoImportState::None;

		for attr in root_attrs {
			if let Some(parsed_attr) = try_parse_attr::<NoStd>(self.scx, attr) {
				let NoStd {} = match parsed_attr {
					Ok(attr) => attr,
					Err(diag) => {
						self.scx.dcx().emit(&diag);
						continue;
					}
				};
				match state {
					AutoImportState::None => state = AutoImportState::NoStd(attr.span),
					AutoImportState::NoStd(_span) => todo!("lint duplicate"),
					AutoImportState::NoCore(_span) => {
						todo!("lint about stricter import restriction")
					}
				}
			}
			if let Some(parsed_attr) = try_parse_attr::<NoCore>(self.scx, attr) {
				let NoCore {} = match parsed_attr {
					Ok(attr) => attr,
					Err(diag) => {
						self.scx.dcx().emit(&diag);
						continue;
					}
				};
				match state {
					AutoImportState::None | AutoImportState::NoStd(..) => {
						state = AutoImportState::NoCore(attr.span);
					}
					AutoImportState::NoCore(span) => todo!("lint duplicate"),
				}
			}
		}

		let kind = match state {
			AutoImportState::None => Some(ItemKind::ExternUse {
				name: Id::new(sym::std, Span::DUMMY),
			}),
			AutoImportState::NoStd(..) => Some(ItemKind::ExternUse {
				name: Id::new(sym::core, Span::DUMMY),
			}),
			AutoImportState::NoCore(..) => None,
		};

		kind.map(|kind| Item {
			attrs: vec![],
			kind,
			span: Span::DUMMY,
			id: self.make_node_id(),
		})
	}
}

/// Expressions
impl Parser<'_> {
	/// Parse an expression
	fn parse_expr(&mut self) -> Result<Expr> {
		let lhs = self.parse_expr_single_and_postfix()?;
		self.parse_expr_assoc_rest(None, lhs)
	}

	/// Parse an expression right-hand side by eating association operators
	/// (e.g. binary operators or assignment equal) while their precedence is higher.
	fn parse_expr_assoc_rest(&mut self, precedence: Option<u32>, mut lhs: Expr) -> Result<Expr> {
		let lo = self.token.span;

		while let Some(assoc_op) = self.eat_assoc_token_with_precedence(precedence) {
			let left = Box::new(lhs);
			let right = Box::new(self.parse_expr()?);

			let new_kind = match assoc_op.bit {
				AssocOp::Binary(bin_op) => {
					let op = Spanned::new(bin_op, assoc_op.span);
					ExprKind::Binary { op, left, right }
				}
				AssocOp::ShortCircuit(sc_op) => {
					let op = Spanned::new(sc_op, assoc_op.span);
					ExprKind::ShortCircuit { op, left, right }
				}
				AssocOp::Assign => ExprKind::Assign {
					target: left,
					value: right,
				},
			};

			lhs = Expr {
				// attributes on binary operation needs paren
				attrs: Vec::default(),
				kind: new_kind,
				span: self.close_span(lo),
				id: self.make_node_id(),
			};
		}
		Ok(lhs)
	}

	fn eat_assoc_token_with_precedence(
		&mut self,
		prev_prec: Option<u32>,
	) -> Option<Spanned<AssocOp>> {
		if let Some(op) = AssocOp::from_token_kind(self.token.kind)
			// only continue if next op precedence higher
			&& prev_prec.is_none_or(|prev| prev <= op.precedence())
		{
			// eat assoc op
			self.bump();

			Some(Spanned::new(op, self.last_token.span))
		} else {
			None
		}
	}

	/// Parse a single expression with postfix constructs
	fn parse_expr_single_and_postfix(&mut self) -> Result<Expr> {
		let mut expr = self.parse_expr_single()?;
		loop {
			match self.parse_expr_postfix(expr)? {
				ControlFlow::Continue(next_expr) => expr = next_expr,
				ControlFlow::Break(next_expr) => break Ok(next_expr),
			}
		}
	}

	// check for postfix constructs
	fn parse_expr_postfix(&mut self, mut expr: Expr) -> Result<ControlFlow<Expr, Expr>> {
		let lo = expr.span.start();

		let (kind, attrs) = if self.eat(Dot) {
			let attrs = mem::take(&mut expr.attrs);

			let kind = if matches!(self.token.kind, Ident(_)) {
				// `<expr> . foo` or `<expr> . bar ( <args> )`
				let field = self.expect_ident()?;

				if self.eat(OpenParen) {
					let params =
						self.parse_seq_rest(OpenParen, CloseParen, Comma, Parser::parse_expr)?;
					ExprKind::Method {
						expr: Box::new(expr),
						name: field,
						params,
					}
				} else {
					ExprKind::Field {
						expr: Box::new(expr),
						name: field,
					}
				}
			} else if self.eat(Star) {
				// `<expr> . *`
				ExprKind::Deref {
					expr: Box::new(expr),
				}
			} else if self.eat_kw(kw::Match) {
				// `<expr> . match { <arms> }`
				let arms = self.parse_expr_match();
				ExprKind::Match {
					expr: Box::new(expr),
					arms: todo!("parse match expression"),
				}
			} else {
				let report = errors::expected_construct_no_match("a postfix construct", self.token);
				return Err(Diagnostic::new(report));
			};

			(kind, attrs)
		} else if self.check(OpenParen) {
			let attrs = mem::take(&mut expr.attrs);

			// `<expr> ()`
			let kind = self.parse_fn_call(expr)?;

			(kind, attrs)
		} else {
			return Ok(ControlFlow::Break(expr));
		};

		Ok(ControlFlow::Continue(Expr {
			attrs,
			kind,
			span: self.close_span(lo),
			id: self.make_node_id(),
		}))
	}

	/// Parse a single expression without eating binary operators
	///
	/// See [`Self::parse_expr`] for full expression parsing including binary operations
	fn parse_expr_single(&mut self) -> Result<Expr> {
		let lo = self.token.span;

		let attrs = self.parse_attrs(AttrKind::Next)?;

		let kind = if self.eat_kw(kw::Not) {
			self.parse_expr_not()?
		} else if self.eat(Dash) {
			self.parse_expr_neg()?
		} else if matches!(self.token.kind, Ident(sym)) {
			self.parse_expr_access()?
		} else if let LiteralStr(sym) = self.token.kind {
			self.bump();
			// TODO: handle prefixed strings (e.g. c"content")
			ExprKind::LiteralStr { sym }
		} else if let LiteralInt(sym) = self.token.kind {
			self.bump();
			ExprKind::LiteralInt { sym }
		} else if let LiteralFloat(sym) = self.token.kind {
			self.bump();
			ExprKind::LiteralFloat { sym }
		} else if self.eat(OpenParen) {
			self.parse_expr_paren()?
		} else if self.eat_kw(kw::If) {
			self.parse_expr_if()?
		} else if self.eat_kw(kw::While) {
			self.parse_expr_while()?
		} else if self.eat_kw(kw::Loop) {
			self.parse_expr_loop()?
		} else if self.eat_kw(kw::Return) {
			self.parse_expr_return()?
		} else if self.eat_kw(kw::Break) {
			self.parse_expr_break()?
		} else if self.eat_kw(kw::Continue) {
			self.parse_expr_continue()?
		} else {
			let report = errors::expected_construct_no_match("an expression", self.token);
			return Err(Diagnostic::new(report));
		};

		Ok(Expr {
			attrs,
			kind,
			span: self.close_span(lo),
			id: self.make_node_id(),
		})
	}

	/// Parse [`ExprKind::Unary`] for [`UnaryOp::Not`]
	fn parse_expr_not(&mut self) -> Result<ExprKind> {
		debug_assert_eq!(self.last_token.kind, Bang);

		let expr = Box::new(self.parse_expr()?);

		let op = Spanned::new(UnaryOp::Not, self.last_token.span);
		Ok(ExprKind::Unary { op, expr })
	}

	/// Parse [`ExprKind::Unary`] for [`UnaryOp::Minus`]
	fn parse_expr_neg(&mut self) -> Result<ExprKind> {
		debug_assert_eq!(self.last_token.kind, Dash);

		let expr = Box::new(self.parse_expr()?);

		let op = Spanned::new(UnaryOp::Minus, self.last_token.span);
		Ok(ExprKind::Unary { op, expr })
	}

	/// Parse [`ExprKind::Access`]
	fn parse_expr_access(&mut self) -> Result<ExprKind> {
		let path = self.parse_path()?;

		Ok(ExprKind::Access { path })
	}

	/// Parse [`ExprKind::Paren`]
	fn parse_expr_paren(&mut self) -> Result<ExprKind> {
		debug_assert_eq!(self.last_token.kind, OpenParen);

		let expr = Box::new(self.parse_expr()?);
		self.expect(CloseParen)?;

		Ok(ExprKind::Paren { expr })
	}

	/// Parse [`ExprKind::If`]
	fn parse_expr_if(&mut self) -> Result<ExprKind> {
		debug_assert_eq!(self.last_token.kind, Kw(kw::If));

		let cond = Box::new(self.parse_expr()?);
		let conseq = Box::new(self.parse_block()?);
		let altern = if self.eat_kw(kw::Else) {
			Some(Box::new(self.parse_block()?))
		} else {
			None
		};

		Ok(ExprKind::If {
			cond,
			conseq,
			altern,
		})
	}

	/// Parse [`ExprKind::WhileLoop`]
	fn parse_expr_while(&mut self) -> Result<ExprKind> {
		debug_assert_eq!(self.last_token.kind, Kw(kw::While));

		let check = Box::new(self.parse_expr()?);
		let body = Box::new(self.parse_block()?);

		Ok(ExprKind::WhileLoop { check, body })
	}

	/// Parse [`ExprKind::Return`]
	fn parse_expr_return(&mut self) -> Result<ExprKind> {
		debug_assert_eq!(self.last_token.kind, Kw(kw::Return));

		// TODO: bad for recovery
		let expr = self.parse_expr().ok().map(Box::new);

		Ok(ExprKind::Return { expr })
	}

	/// Parse [`ExprKind::Break`]
	fn parse_expr_break(&mut self) -> Result<ExprKind> {
		debug_assert_eq!(self.last_token.kind, Kw(kw::Break));

		let label_span = self.token.span;
		let label = if self.eat(Apostrophe) {
			let label = self.expect_ident()?;
			Some(Spanned::new(label, self.close_span(label_span)))
		} else {
			None
		};

		let expr = self.parse_expr().ok().map(Box::new);

		Ok(ExprKind::Break { expr, label })
	}

	/// Parse [`ExprKind::Continue`]
	fn parse_expr_continue(&mut self) -> Result<ExprKind> {
		debug_assert_eq!(self.last_token.kind, Kw(kw::Continue));

		let label_span = self.token.span;
		let label = if self.eat(Apostrophe) {
			let label = self.expect_ident()?;
			Some(Spanned::new(label, self.close_span(label_span)))
		} else {
			None
		};

		Ok(ExprKind::Continue { label })
	}
}

/// Items
impl Parser<'_> {
	fn parse_item(&mut self) -> Result<Item> {
		let lo = self.token.span;

		let mut attrs = self.parse_attrs(AttrKind::Next)?;

		let kind = if self.eat_kw(kw::Fn) {
			self.parse_function()?
		} else if self.eat_kw(kw::Unsafe) {
			self.parse_item_unsafe_extern()?
		} else if self.eat_kw(kw::Struct) {
			self.parse_item_struct()?
		} else if self.eat_kw(kw::Enum) {
			self.parse_item_enum()?
		} else if self.eat_kw(kw::Trait) {
			self.parse_item_trait()?
		} else if self.eat_kw(kw::For) {
			self.parse_item_trait_impl()?
		} else if self.eat_kw(kw::Type) {
			self.parse_item_type_alias()?
		} else if self.eat_kw(kw::Mod) {
			self.parse_item_mod(&mut attrs)?
		} else if self.eat_kw(kw::Extern) {
			// TODO: recover to unsafe extern block
			self.parse_item_extern_use()?
		} else {
			let report = errors::expected_construct_no_match("an item", self.token);
			return Err(Diagnostic::new(report));
		};

		Ok(Item {
			kind,
			attrs,
			span: self.close_span(lo),
			id: self.make_node_id(),
		})
	}

	fn parse_module_inner(&mut self) -> Result<(Vec<Attr>, Vec<Item>)> {
		let attrs = self.parse_attrs(AttrKind::Parent)?;
		let items = self.parse_until(Eof, Parser::parse_item)?;
		Ok((attrs, items))
	}

	fn parse_function(&mut self) -> Result<ItemKind> {
		debug_assert_eq!(self.last_token.kind, Kw(kw::Fn));

		let name = self.expect_ident()?;
		let generics = self.parse_generics()?;
		let decl = self.parse_fn_decl()?;

		let body = if self.check(OpenBrace) {
			Some(Box::new(self.parse_block()?))
		} else if self.eat(Semi) {
			None
		} else {
			let report =
				errors::expected_construct_no_match("a function body or a semicolon", self.token);
			return Err(Diagnostic::new(report));
		};

		Ok(ItemKind::Function(Function {
			name,
			generics,
			decl,
			body,
		}))
	}

	fn parse_item_type_alias(&mut self) -> Result<ItemKind> {
		debug_assert_eq!(self.last_token.kind, Kw(kw::Type));

		let name = self.expect_ident()?;
		let alias = if self.eat(Eq) {
			let ty = Some(Box::new(self.parse_ty()?));
			self.expect(Semi)?;
			ty
		} else if self.eat(Semi) {
			None
		} else {
			let report = errors::expected_construct_no_match("a type alias body", self.token);
			return Err(Diagnostic::new(report));
		};

		Ok(ItemKind::TypeAlias(TypeAlias { name, alias }))
	}

	fn parse_item_mod(&mut self, attrs: &mut Vec<Attr>) -> Result<ItemKind> {
		debug_assert_eq!(self.last_token.kind, Kw(kw::Mod));

		let mut custom_path = None;

		for attr in &*attrs {
			if let Some(parsed_attr) = try_parse_attr::<ModPath>(self.scx, attr) {
				let path = match parsed_attr {
					Ok(ModPath { path }) => path,
					Err(diag) => {
						self.scx.dcx().emit(&diag);
						continue;
					}
				};
				let before = custom_path.replace(path);
				assert!(before.is_none());
			}
		}

		let name = self.expect_ident()?;
		let (inner_attrs, items, inline) = if self.eat(Semi) {
			let filepath = match custom_path {
				Some(path) => self.module.current_dir.join(path),
				None => self.submodule_path(name)?,
			};
			let value = self.scx.source_map.write().load_source_from_file(&filepath);
			let file = match value {
				Ok(file) => file,
				Err(err) => todo!("could not read file {}: {err}", filepath.display()),
			};

			let submodule = self.module.with_file(name.sym, filepath);
			let mut parser = Parser::new(self.scx, &file, submodule);
			let (attrs, items) = parser.parse_module_inner()?;
			(attrs, items, false)
		} else {
			self.expect(OpenBrace)?;
			let (attrs, items) = self.parse_module_inner()?;
			self.expect(CloseBrace)?;
			(attrs, items, true)
		};

		attrs.extend(inner_attrs);

		Ok(ItemKind::Module {
			name,
			items,
			inline,
		})
	}

	fn submodule_path(&self, name: Id) -> Result<PathBuf> {
		// `<ident>.kl`
		let sibling_path = format!("{}.kl", self.scx.symbols.resolve(name.sym));
		let sibling_path = self.module.current_dir.join(sibling_path);
		let sibling_path_exists = match sibling_path.try_exists() {
			Ok(exists) => exists,
			Err(err) => todo!(),
		};
		// `<ident>/mod.kl`
		let child_path = format!(
			"{}{}mod.kl",
			self.scx.symbols.resolve(name.sym),
			path::MAIN_SEPARATOR
		);
		let child_path = self.module.current_dir.join(child_path);
		let child_path_exists = match child_path.try_exists() {
			Ok(exists) => exists,
			Err(err) => todo!(),
		};

		let file_path = match (sibling_path_exists, child_path_exists) {
			(true, false) => sibling_path,
			(false, true) => child_path,
			// Both files exist, so we can't load the scope
			(true, true) => {
				let report =
					errors::module_multiple_candidates(name.span, &child_path, &sibling_path);
				return Err(Diagnostic::new(report));
			}
			// Neither file exists, so we can't load the scope
			(false, false) => {
				let report = errors::module_no_candidates(name.span, &child_path, &sibling_path);
				return Err(Diagnostic::new(report));
			}
		};

		Ok(file_path)
	}

	/// Parse [`ItemKind::ForeignMod`] block of items
	fn parse_item_unsafe_extern(&mut self) -> Result<ItemKind> {
		debug_assert_eq!(self.last_token.kind, Kw(kw::Unsafe));

		self.expect(Kw(kw::Extern))?;

		self.expect(OpenBrace)?;
		let items = self.parse_until(CloseBrace, Parser::parse_item)?;

		Ok(ItemKind::ForeignMod { items })
	}

	/// Parse [`ItemKind::Struct`]
	fn parse_item_struct(&mut self) -> Result<ItemKind> {
		debug_assert_eq!(self.last_token.kind, Kw(kw::Struct));

		let name = self.expect_ident()?;
		let generics = self.parse_generics()?;
		let fields = if self.eat(OpenBrace) {
			self.parse_seq_rest(OpenBrace, CloseBrace, Comma, Self::parse_field_def)?
		} else if self.eat(OpenParen) {
			let fields = self.parse_seq_rest(OpenParen, CloseParen, Comma, Self::parse_ty)?;

			fields
				.into_iter()
				.enumerate()
				.map(|(i, ty)| FieldDef {
					name: Id::new(self.scx.symbols.intern(&i.to_string()), ty.span),
					span: ty.span,
					ty,
				})
				.collect()
		} else if self.eat(Semi) {
			Vec::new()
		} else {
			let report = errors::expected_construct_no_match("a struct definition", self.token);
			return Err(Diagnostic::new(report));
		};

		Ok(ItemKind::Struct {
			name,
			generics,
			fields,
		})
	}

	/// Parse [`ItemKind::Enum`]
	fn parse_item_enum(&mut self) -> Result<ItemKind> {
		debug_assert_eq!(self.last_token.kind, Kw(kw::Enum));

		let name = self.expect_ident()?;
		let generics = self.parse_generics()?;
		self.expect(OpenBrace)?;
		let variants =
			self.parse_seq_rest(OpenBrace, CloseBrace, Comma, Self::parse_variant_def)?;

		Ok(ItemKind::Enum {
			name,
			generics,
			variants,
		})
	}

	/// Parse [`ItemKind::Trait`]
	fn parse_item_trait(&mut self) -> Result<ItemKind> {
		debug_assert_eq!(self.last_token.kind, Kw(kw::Trait));

		let name = self.expect_ident()?;
		let generics = self.parse_generics()?;

		self.expect(OpenBrace)?;
		let members = self.parse_until(CloseBrace, Parser::parse_item)?;

		Ok(ItemKind::Trait {
			name,
			generics,
			members,
		})
	}

	/// Parse [`ItemKind::TraitImpl`]
	fn parse_item_trait_impl(&mut self) -> Result<ItemKind> {
		debug_assert_eq!(self.last_token.kind, Kw(kw::For));

		let type_ = self.parse_path()?;
		self.expect(Kw(kw::Impl))?;
		let trait_ = self.parse_path()?;
		self.expect(OpenBrace)?;
		let members = self.parse_until(CloseBrace, Parser::parse_item)?;

		Ok(ItemKind::TraitImpl {
			type_,
			trait_,
			members,
		})
	}

	/// Parse [`FieldDef`]
	fn parse_field_def(&mut self) -> Result<FieldDef> {
		let lo = self.token.span;

		let name = self.expect_ident()?;
		self.expect(Colon)?;
		let ty = self.parse_ty()?;

		Ok(FieldDef {
			name,
			ty,
			span: self.close_span(lo),
		})
	}

	/// Parse [`Variant`] and [`VariantKind`]
	fn parse_variant_def(&mut self) -> Result<Variant> {
		let lo = self.token.span;

		let name = self.expect_ident()?;

		let fields = if self.eat(OpenBrace) {
			let fields =
				self.parse_seq_rest(OpenBrace, CloseBrace, Comma, Self::parse_field_def)?;
			VariantKind::Struct(fields)
		} else if self.eat(OpenParen) {
			let fields = self.parse_seq_rest(OpenParen, CloseParen, Comma, Self::parse_ty)?;
			VariantKind::Tuple(fields)
		} else {
			VariantKind::Bare
		};

		Ok(Variant {
			name,
			kind: fields,
			span: self.close_span(lo),
		})
	}

	fn parse_fn_decl(&mut self) -> Result<FnDecl> {
		let lo = self.token.span;
		let generics = self.parse_generics()?;
		self.expect(OpenParen)?;
		let params = self.parse_seq_rest(OpenParen, CloseParen, Comma, Parser::parse_param)?;
		let ret = if !self.check(OpenBrace) && !self.check(Semi) {
			Some(self.parse_ty()?)
		} else {
			None
		};

		Ok(FnDecl {
			params,
			ret,
			span: self.close_span(lo),
		})
	}

	fn parse_param(&mut self) -> Result<Param> {
		let name = self.expect_ident()?;
		self.expect(Colon)?;
		let ty = self.parse_ty()?;
		Ok(Param {
			name,
			ty,
			id: self.make_node_id(),
		})
	}

	fn parse_path(&mut self) -> Result<Path> {
		let lo = self.token.span;

		let mut segments = Vec::new();
		segments.push(self.parse_path_segment()?);
		segments.extend(self.parse_while(ColonColon, Self::parse_path_segment)?);

		Ok(Path {
			segments,
			span: self.close_span(lo),
			id: self.make_node_id(),
		})
	}

	fn parse_attr_path(&mut self) -> Result<AttrPath> {
		let lo = self.token.span;

		let mut segments = Vec::new();
		segments.push(self.expect_ident()?);
		segments.extend(self.parse_while(ColonColon, Self::expect_ident)?);

		Ok(AttrPath {
			segments,
			span: self.close_span(lo),
			id: self.make_node_id(),
		})
	}

	fn parse_path_segment(&mut self) -> Result<PathSegment> {
		let lo = self.token.span;

		let name = self.expect_ident()?;
		let generics = self.parse_generic_params()?;

		Ok(PathSegment {
			name,
			generics,
			span: self.close_span(lo),
		})
	}

	fn parse_generic_params(&mut self) -> Result<GenericParams> {
		let lo = self.token.span;

		let mut params = Vec::new();

		if self.check(Lt) {
			let mut finished = false;

			self.expect(Lt)?;
			while !self.eat(Gt) && !finished {
				params.push(self.parse_ty()?);

				// no comma means no item left
				finished = !self.eat(Comma);
			}
		}

		Ok(GenericParams {
			params,
			span: self.close_span(lo),
		})
	}

	fn parse_generics(&mut self) -> Result<Generics> {
		if !self.check(Lt) {
			return Ok(Generics {
				idents: vec![],
				span: Span::DUMMY,
			});
		}

		let lo = self.token.span;

		// TODO: this is a modified expansion of
		// let (generics, span) = self.parse_seq(Angled, Comma, Self::expect_ident)?;

		let mut finished = false;
		let mut generics = Vec::new();

		self.expect(Lt)?;
		while !self.eat(Gt) && !finished {
			let name = self.expect_ident()?;
			let default = if self.eat(Eq) {
				Some(self.parse_ty()?)
			} else {
				None
			};

			generics.push(ast::Generic {
				name,
				default,
				id: self.make_node_id(),
			});

			// no comma means no item left
			finished = !self.eat(Comma);
		}

		Ok(Generics {
			idents: generics,
			span: self.close_span(lo),
		})
	}

	/// Parse [`ExprKind::FnCall`]
	fn parse_fn_call(&mut self, expr: Expr) -> Result<ExprKind> {
		let args_lo = self.token.span;
		self.expect(OpenParen)?;
		let args = self.parse_seq_rest(OpenParen, CloseParen, Comma, Parser::parse_expr)?;

		Ok(ExprKind::FnCall {
			expr: Box::new(expr),
			args: Spanned::new(args, self.close_span(args_lo)),
		})
	}

	fn parse_expr_match(&self) -> Vec<()> {
		todo!()
	}

	fn parse_item_extern_use(&mut self) -> Result<ItemKind> {
		debug_assert_eq!(self.last_token.kind, Kw(kw::Extern));

		self.expect(Kw(kw::Use))?;
		let name = self.expect_ident()?;
		self.expect(Semi)?;

		Ok(ItemKind::ExternUse { name })
	}
}

/// Types
impl Parser<'_> {
	fn parse_ty(&mut self) -> Result<Ty> {
		let lo = self.token.span;

		let kind = if matches!(self.token.kind, Ident(_)) {
			self.parse_ty_path()?
		} else if self.eat(Star) {
			self.parse_ty_pointer()?
		// } else if self.eat(Ampersand) {
		// 	self.parse_ty_reference()?
		} else {
			let report = errors::expected_construct_no_match("a type", self.token);
			return Err(Diagnostic::new(report));
		};

		Ok(Ty {
			kind,
			span: self.close_span(lo),
		})
	}

	fn parse_ty_path(&mut self) -> Result<TyKind> {
		debug_assert!(matches!(self.token.kind, Ident(_)));

		let path = self.parse_path()?;

		Ok(TyKind::Path(path))
	}

	// Parse [`TyKind::Reference`]
	// fn parse_ty_reference(&mut self) -> Result<TyKind> {
	// 	debug_assert_eq!(self.last_token.kind, Ampersand);

	// 	let ty = Box::new(self.parse_ty()?);

	// 	Ok(TyKind::Reference(ty))
	// }

	/// Parse [`TyKind::Pointer`]
	fn parse_ty_pointer(&mut self) -> Result<TyKind> {
		debug_assert_eq!(self.last_token.kind, Star);

		let ty = Box::new(self.parse_ty()?);

		Ok(TyKind::Pointer(ty))
	}
}

/// Statements
impl Parser<'_> {
	fn parse_stmt(&mut self) -> Result<Stmt> {
		let lo = self.token.span;
		let kind = match self.token.kind {
			// Keyword(For) => self.parse_stmt_for()?,
			Semi => {
				self.expect(Semi)?;
				StmtKind::Empty
			}

			Kw(kw::Let) => self.parse_stmt_let()?,

			Eof => {
				let report = Report::build(ReportKind::Error, self.token.span)
					.with_message("expected more input")
					.with_label(Label::new(self.token.span).with_message("here"));
				return Err(Diagnostic::new(report));
			}
			_ => {
				let expr = Box::new(self.parse_expr()?);
				if self.eat(Semi) {
					StmtKind::Expr(expr)
				} else {
					// TODO: enforce parsing for expr ret
					StmtKind::ExprRet(expr)
				}
			}
		};
		Ok(Stmt {
			kind,
			span: self.close_span(lo),
			id: self.make_node_id(),
		})
	}

	fn parse_expr_loop(&mut self) -> Result<ExprKind> {
		self.expect(Kw(kw::Loop))?;
		let body = Box::new(self.parse_block()?);
		Ok(ExprKind::Loop { body })
	}

	fn parse_stmt_let(&mut self) -> Result<StmtKind> {
		self.expect(Kw(kw::Let))?;

		let mutable = self.eat_kw(kw::Mut);

		let name = self.expect_ident()?;

		// definition with optional ty
		let ty = if self.eat(Colon) {
			Some(Box::new(self.parse_ty()?))
		} else {
			None
		};

		let value = if self.eat(Semi) {
			None
		} else {
			self.expect(Eq)?;
			let value = Box::new(self.parse_expr()?);
			self.expect(Semi)?;
			Some(value)
		};

		Ok(StmtKind::Let {
			name,
			ty,
			value,
			mutable,
		})
	}
}

impl Parser<'_> {
	fn parse_block(&mut self) -> Result<Block> {
		let lo = self.token.span;
		self.expect(OpenBrace)?;
		let stmts = self.parse_until(CloseBrace, Self::parse_stmt)?;

		Ok(Block {
			stmts,
			span: self.close_span(lo),
			id: self.make_node_id(),
		})
	}

	fn parse_attrs(&mut self, kind: AttrKind) -> Result<Vec<Attr>> {
		let peek = match kind {
			AttrKind::Parent => PoundPound,
			AttrKind::Next => Pound,
		};
		self.parse_while(peek, |p| p.parse_attr(kind))
	}

	fn parse_attr(&mut self, kind: AttrKind) -> Result<Attr> {
		match kind {
			AttrKind::Parent => debug_assert!(matches!(self.last_token.kind, PoundPound)),
			AttrKind::Next => debug_assert!(matches!(self.last_token.kind, Pound)),
		}

		let lo = self.last_token.span;

		let path = self.parse_attr_path()?;

		let meta = if self.eat(OpenParen) {
			let exprs = self.parse_seq_rest(OpenParen, CloseParen, Comma, Parser::parse_expr)?;
			AttrMeta::Tuple(exprs)
		} else if self.eat(OpenBrace) {
			let exprs = self.parse_seq_rest(OpenBrace, CloseBrace, Comma, Parser::parse_expr)?;
			AttrMeta::Map(exprs)
		} else if self.eat(OpenBracket) {
			let exprs =
				self.parse_seq_rest(OpenBracket, CloseBracket, Comma, Parser::parse_expr)?;
			AttrMeta::List(exprs)
		} else {
			AttrMeta::None
		};

		Ok(Attr {
			path,
			meta,
			kind,
			span: self.close_span(lo),
			id: self.make_node_id(),
		})
	}
}

mod errors {
	use std::path::Path;

	use ariadne::{Label, ReportKind};

	use crate::{
		lexer::{Token, TokenKind},
		session::{Report, ReportBuilder, Span},
	};

	pub fn expected_token_kind(expected: TokenKind, actual: Token) -> ReportBuilder {
		Report::build(ReportKind::Error, actual.span)
			.with_message(format!("expected {expected}"))
			.with_label(
				Label::new(actual.span)
					.with_message(format!("found {} that was unexpected", actual.kind)),
			)
	}

	/// Construct should fit in the sentence "expected {}"
	pub fn expected_construct_no_match(construct: &str, token: Token) -> ReportBuilder {
		Report::build(ReportKind::Error, token.span)
			.with_message(format!("expected {construct}"))
			.with_label(
				Label::new(token.span)
					.with_message(format!("found {} that was unexpected", token.kind)),
			)
	}

	pub fn module_multiple_candidates(
		name_span: Span,
		child_path: &Path,
		sibling_path: &Path,
	) -> ReportBuilder {
		Report::build(ReportKind::Error, name_span)
			.with_message(format!(
				"found both {} and {} as possible candidates",
				child_path.display(),
				sibling_path.display()
			))
			.with_label(Label::new(name_span).with_message("while trying to load this module"))
	}

	pub fn module_no_candidates(
		name_span: Span,
		child_path: &Path,
		sibling_path: &Path,
	) -> ReportBuilder {
		Report::build(ReportKind::Error, name_span)
			.with_message(format!(
				"searched for {} and {} as possible candidates, but found none",
				child_path.display(),
				sibling_path.display()
			))
			.with_label(Label::new(name_span).with_message("while trying to load this module"))
	}
}
