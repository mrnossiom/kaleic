//! Tokens to AST parsing logic
//!
//! Contains the recursive decent parser of the language.
//!
//! Entrypoint to parsing is [`parse_root`].

use std::{fmt, mem, ops::ControlFlow};

use ariadne::{Label, Report, ReportKind};

#[expect(
	clippy::enum_glob_use,
	reason = "single glob usage here, reduces code size"
)]
use crate::lexer::TokenKind::*;
use crate::{
	ast::{
		Attr, AttrMeta, BinaryOp, Block, Expr, ExprKind, FieldDef, FnDecl, Function, Generics,
		Ident as Id, Item, ItemKind, NodeId, Param, Path, Root, ShortCircuitOp, Spanned, Stmt,
		StmtKind, Ty, TyKind, TypeAlias, UnaryOp, Variant, VariantKind,
	},
	errors,
	lexer::{Keyword as Kw, Lexer, Token, TokenKind},
	session::{Diagnostic, SessionCtx, SourceFile, Span},
};

pub fn parse_root(scx: &SessionCtx, source: &SourceFile) -> Root {
	let mut p = Parser::new(scx, source);
	match Root::parse(&mut p) {
		Ok(ast) => ast,
		Err(diag) => scx.dcx().emit_fatal(&diag),
	}
}

trait Parse: Sized + fmt::Debug {
	fn parse(p: &mut Parser) -> Result<Self, Diagnostic>;
}

type PResult<T> = std::result::Result<T, Diagnostic>;

#[derive(Debug)]
enum AssocOp {
	Binary(BinaryOp),
	ShortCircuit(ShortCircuitOp),
	Assign,
}

impl AssocOp {
	fn from_token_kind(kind: TokenKind) -> Option<Self> {
		let kind = match kind {
			TokenKind::Plus => Self::Binary(BinaryOp::Plus),
			TokenKind::Dash => Self::Binary(BinaryOp::Minus),
			TokenKind::Star => Self::Binary(BinaryOp::Mul),
			TokenKind::Div => Self::Binary(BinaryOp::Div),
			TokenKind::Mod => Self::Binary(BinaryOp::Mod),
			TokenKind::BitwiseAnd => Self::Binary(BinaryOp::And),
			TokenKind::BitwiseOr => Self::Binary(BinaryOp::Or),
			TokenKind::Xor => Self::Binary(BinaryOp::Xor),
			TokenKind::Shl => Self::Binary(BinaryOp::Shl),
			TokenKind::Shr => Self::Binary(BinaryOp::Shr),
			TokenKind::Gt => Self::Binary(BinaryOp::Gt),
			TokenKind::Ge => Self::Binary(BinaryOp::Ge),
			TokenKind::Lt => Self::Binary(BinaryOp::Lt),
			TokenKind::Le => Self::Binary(BinaryOp::Le),
			TokenKind::EqEq => Self::Binary(BinaryOp::EqEq),
			TokenKind::Ne => Self::Binary(BinaryOp::Ne),
			TokenKind::Eq => Self::Assign,
			TokenKind::Keyword(Kw::And) => Self::ShortCircuit(ShortCircuitOp::And),
			TokenKind::Keyword(Kw::Or) => Self::ShortCircuit(ShortCircuitOp::Or),
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

	next_node_id: u32,
}

impl<'scx> Parser<'scx> {
	fn new(scx: &'scx SessionCtx, file: &'scx SourceFile) -> Self {
		let SourceFile {
			name: _,
			content,
			offset,
		} = &file;

		let mut parser = Self {
			scx,

			lexer: Lexer::new(scx, content, *offset),

			token: Token::DUMMY,
			last_token: Token::DUMMY,

			next_node_id: 0,
		};

		// init the first token
		parser.bump();

		parser
	}

	fn make_node_id(&mut self) -> crate::ast::NodeId {
		let next_node_id = self.next_node_id;
		self.next_node_id += 1;
		NodeId(next_node_id)
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

	#[track_caller]
	fn expect(&mut self, expected_kind: TokenKind) -> PResult<Token> {
		if self.check(expected_kind) {
			self.bump();
			Ok(self.token)
		} else {
			let report = errors::parser::expected_token_kind(expected_kind, self.token);
			Err(Diagnostic::new(report))
		}
	}

	fn eat_ident(&mut self) -> Option<Id> {
		self.token.as_ident().inspect(|_| {
			self.bump();
		})
	}

	fn expect_ident(&mut self) -> PResult<Id> {
		self.eat_ident().ok_or_else(|| {
			let placeholder = self.scx.symbols.intern("_");
			let report = errors::parser::expected_token_kind(Ident(placeholder), self.token);
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
		mut parse: impl FnMut(&mut Self) -> PResult<T>,
	) -> PResult<Vec<T>> {
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

	fn parse_until<T: Parse>(&mut self, end: TokenKind) -> PResult<Vec<T>> {
		let mut many = Vec::new();
		while !self.eat(end) {
			many.push(T::parse(self)?);
		}
		Ok(many)
	}

	fn parse_until_func<T>(
		&mut self,
		end: TokenKind,
		mut parse: impl FnMut(&mut Self) -> PResult<T>,
	) -> PResult<Vec<T>> {
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
		mut parse: impl FnMut(&mut Self) -> PResult<T>,
	) -> PResult<Vec<T>> {
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

/// Expressions
impl Parser<'_> {
	/// Parse an expression
	fn parse_expr(&mut self) -> PResult<Expr> {
		let lhs = self.parse_expr_single_and_postfix()?;
		self.parse_expr_assoc_rest(None, lhs)
	}

	/// Parse an expression right-hand side by eating association operators
	/// (e.g. binary operators or assignment equal) while their precedence is higher.
	fn parse_expr_assoc_rest(&mut self, precedence: Option<u32>, mut lhs: Expr) -> PResult<Expr> {
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
				kind: new_kind,
				span: self.close_span(lo),
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
	fn parse_expr_single_and_postfix(&mut self) -> PResult<Expr> {
		let mut expr = self.parse_expr_single()?;
		loop {
			match self.parse_expr_postfix(expr)? {
				ControlFlow::Continue(next_expr) => expr = next_expr,
				ControlFlow::Break(next_expr) => break Ok(next_expr),
			}
		}
	}

	// check for postfix constructs
	fn parse_expr_postfix(&mut self, expr: Expr) -> PResult<ControlFlow<Expr, Expr>> {
		let lo = expr.span.start();
		let kind = if self.eat(Dot) {
			if matches!(self.token.kind, Ident(_)) {
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
			} else if self.eat(Keyword(Kw::Match)) {
				// `<expr> . match { <arms> }`
				let arms = self.parse_expr_match();
				ExprKind::Match {
					expr: Box::new(expr),
					arms: todo!("parse match expression"),
				}
			} else {
				let report =
					errors::parser::expected_construct_no_match("a postfix construct", self.token);
				return Err(Diagnostic::new(report));
			}
		} else if self.check(OpenParen) {
			// `<expr> ()`
			self.parse_fn_call(expr)?
		} else {
			return Ok(ControlFlow::Break(expr));
		};
		Ok(ControlFlow::Continue(Expr {
			kind,
			span: self.close_span(lo),
		}))
	}

	/// Parse a single expression without eating binary operators
	///
	/// See [`Self::parse_expr`] for full expression parsing including binary operations
	fn parse_expr_single(&mut self) -> PResult<Expr> {
		let lo = self.token.span;

		let kind = if self.eat(Keyword(Kw::Not)) {
			self.parse_expr_not()?
		} else if self.eat(Dash) {
			self.parse_expr_neg()?
		} else if matches!(self.token.kind, Ident(_)) {
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
		} else if self.eat(Keyword(Kw::If)) {
			self.parse_expr_if()?
		} else if self.eat(Keyword(Kw::While)) {
			self.parse_expr_while()?
		} else if self.eat(Keyword(Kw::Loop)) {
			self.parse_expr_loop()?
		} else if self.eat(Keyword(Kw::Return)) {
			self.parse_expr_return()?
		} else if self.eat(Keyword(Kw::Break)) {
			self.parse_expr_break()?
		} else if self.eat(Keyword(Kw::Continue)) {
			self.parse_expr_continue()?
		} else {
			let report = errors::parser::expected_construct_no_match("an expression", self.token);
			return Err(Diagnostic::new(report));
		};

		Ok(Expr {
			kind,
			span: self.close_span(lo),
		})
	}

	/// Parse [`ExprKind::Unary`] for [`UnaryOp::Not`]
	fn parse_expr_not(&mut self) -> PResult<ExprKind> {
		debug_assert_eq!(self.last_token.kind, Bang);

		let expr = Box::new(self.parse_expr()?);

		let op = Spanned::new(UnaryOp::Not, self.last_token.span);
		Ok(ExprKind::Unary { op, expr })
	}

	/// Parse [`ExprKind::Unary`] for [`UnaryOp::Minus`]
	fn parse_expr_neg(&mut self) -> PResult<ExprKind> {
		debug_assert_eq!(self.last_token.kind, Dash);

		let expr = Box::new(self.parse_expr()?);

		let op = Spanned::new(UnaryOp::Minus, self.last_token.span);
		Ok(ExprKind::Unary { op, expr })
	}

	/// Parse [`ExprKind::Access`]
	fn parse_expr_access(&mut self) -> PResult<ExprKind> {
		let path = self.parse_path()?;

		Ok(ExprKind::Access { path })
	}

	/// Parse [`ExprKind::Paren`]
	fn parse_expr_paren(&mut self) -> PResult<ExprKind> {
		debug_assert_eq!(self.last_token.kind, OpenParen);

		let expr = Box::new(self.parse_expr()?);
		self.expect(CloseParen)?;

		Ok(ExprKind::Paren { expr })
	}

	/// Parse [`ExprKind::If`]
	fn parse_expr_if(&mut self) -> PResult<ExprKind> {
		debug_assert_eq!(self.last_token.kind, Keyword(Kw::If));

		let cond = Box::new(self.parse_expr()?);
		let conseq = Box::new(self.parse_block()?);
		let altern = if self.eat(Keyword(Kw::Else)) {
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
	fn parse_expr_while(&mut self) -> PResult<ExprKind> {
		debug_assert_eq!(self.last_token.kind, Keyword(Kw::While));

		let check = Box::new(self.parse_expr()?);
		let body = Box::new(self.parse_block()?);

		Ok(ExprKind::WhileLoop { check, body })
	}

	/// Parse [`ExprKind::Return`]
	fn parse_expr_return(&mut self) -> PResult<ExprKind> {
		debug_assert_eq!(self.last_token.kind, Keyword(Kw::Return));

		// TODO: bad for recovery
		let expr = self.parse_expr().ok().map(Box::new);

		Ok(ExprKind::Return { expr })
	}

	/// Parse [`ExprKind::Break`]
	fn parse_expr_break(&mut self) -> PResult<ExprKind> {
		debug_assert_eq!(self.last_token.kind, Keyword(Kw::Break));

		let label_span = self.token.span;
		let label = if self.eat(TokenKind::Apostrophe) {
			let label = self.expect_ident()?;
			Some(Spanned::new(label, self.close_span(label_span)))
		} else {
			None
		};

		let expr = self.parse_expr().ok().map(Box::new);

		Ok(ExprKind::Break { expr, label })
	}

	/// Parse [`ExprKind::Continue`]
	fn parse_expr_continue(&mut self) -> PResult<ExprKind> {
		debug_assert_eq!(self.last_token.kind, Keyword(Kw::Continue));

		let label_span = self.token.span;
		let label = if self.eat(TokenKind::Apostrophe) {
			let label = self.expect_ident()?;
			Some(Spanned::new(label, self.close_span(label_span)))
		} else {
			None
		};

		Ok(ExprKind::Continue { label })
	}
}

impl Parse for Root {
	fn parse(p: &mut Parser) -> Result<Self, Diagnostic> {
		let attrs = p.parse_while(TokenKind::PoundPound, |p| p.parse_attr(&AttrKind::Parent))?;

		let items = p.parse_until::<Item>(Eof)?;

		Ok(Self { attrs, items })
	}
}

impl Parse for Item {
	fn parse(p: &mut Parser) -> Result<Self, Diagnostic> {
		let lo = p.token.span;

		let attrs = p.parse_while(TokenKind::Pound, |p| p.parse_attr(&AttrKind::Next))?;

		let kind = if p.eat(Keyword(Kw::Fn)) {
			ItemKind::Function(Parse::parse(p)?)
		} else if p.eat(Keyword(Kw::Unsafe)) {
			p.parse_item_unsafe_extern()?
		} else if p.eat(Keyword(Kw::Struct)) {
			p.parse_item_struct()?
		} else if p.eat(Keyword(Kw::Enum)) {
			p.parse_item_enum()?
		} else if p.eat(Keyword(Kw::Trait)) {
			p.parse_item_trait()?
		} else if p.eat(Keyword(Kw::For)) {
			p.parse_item_trait_impl()?
		} else if p.eat(Keyword(Kw::Type)) {
			ItemKind::TypeAlias(TypeAlias::parse(p)?)
		} else if p.eat(Keyword(Kw::Extern)) {
			todo!("recover to unsafe extern block");
		} else {
			let report = errors::parser::expected_construct_no_match("an item", p.token);
			return Err(Diagnostic::new(report));
		};

		Ok(Self {
			kind,
			attrs,
			span: p.close_span(lo),
			id: p.make_node_id(),
		})
	}
}

impl Parse for Function {
	fn parse(p: &mut Parser) -> Result<Self, Diagnostic> {
		debug_assert_eq!(p.last_token.kind, Keyword(Kw::Fn));

		let (name, decl) = p.parse_fn_decl()?;
		let body = if p.check(OpenBrace) {
			Some(Box::new(p.parse_block()?))
		} else if p.eat(Semi) {
			None
		} else {
			let report = errors::parser::expected_construct_no_match(
				"a function body or a semicolon",
				p.token,
			);
			return Err(Diagnostic::new(report));
		};

		Ok(Self { name, decl, body })
	}
}

impl Parse for TypeAlias {
	fn parse(p: &mut Parser) -> Result<Self, Diagnostic> {
		debug_assert_eq!(p.last_token.kind, Keyword(Kw::Type));

		let name = p.expect_ident()?;
		let alias = if p.eat(Eq) {
			let ty = Some(Box::new(p.parse_ty()?));
			p.expect(Semi)?;
			ty
		} else if p.eat(Semi) {
			None
		} else {
			let report = errors::parser::expected_construct_no_match("a type alias body", p.token);
			return Err(Diagnostic::new(report));
		};

		Ok(Self { name, alias })
	}
}

/// Items
impl Parser<'_> {
	/// Parse [`Extern`] block of items
	fn parse_item_unsafe_extern(&mut self) -> PResult<ItemKind> {
		debug_assert_eq!(self.last_token.kind, Keyword(Kw::Unsafe));

		self.expect(Keyword(Kw::Extern))?;

		self.expect(OpenBrace)?;
		let items = self.parse_until_func(CloseBrace, |p| Item::parse(p))?;

		Ok(ItemKind::ForeignMod { items })
	}

	/// Parse [`ItemKind::Struct`]
	fn parse_item_struct(&mut self) -> PResult<ItemKind> {
		debug_assert_eq!(self.last_token.kind, Keyword(Kw::Struct));

		let name = self.expect_ident()?;
		let generics = self.parse_generics_def()?;
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
			let report =
				errors::parser::expected_construct_no_match("a struct definition", self.token);
			return Err(Diagnostic::new(report));
		};

		Ok(ItemKind::Struct {
			name,
			generics,
			fields,
		})
	}

	/// Parse [`ItemKind::Enum`]
	fn parse_item_enum(&mut self) -> PResult<ItemKind> {
		debug_assert_eq!(self.last_token.kind, Keyword(Kw::Enum));

		let name = self.expect_ident()?;
		let generics = self.parse_generics_def()?;
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
	fn parse_item_trait(&mut self) -> PResult<ItemKind> {
		debug_assert_eq!(self.last_token.kind, Keyword(Kw::Trait));

		let name = self.expect_ident()?;
		let generics = self.parse_generics_def()?;

		self.expect(OpenBrace)?;
		let members = self.parse_until_func(CloseBrace, Item::parse)?;

		Ok(ItemKind::Trait {
			name,
			generics,
			members,
		})
	}

	/// Parse [`ItemKind::TraitImpl`]
	fn parse_item_trait_impl(&mut self) -> PResult<ItemKind> {
		debug_assert_eq!(self.last_token.kind, Keyword(Kw::For));

		let type_ = self.parse_path()?;
		self.expect(Keyword(Kw::Impl))?;
		let trait_ = self.parse_path()?;
		self.expect(OpenBrace)?;
		let members = self.parse_until_func(CloseBrace, Item::parse)?;

		Ok(ItemKind::TraitImpl {
			type_,
			trait_,
			members,
		})
	}

	/// Parse [`FieldDef`]
	fn parse_field_def(&mut self) -> PResult<FieldDef> {
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
	fn parse_variant_def(&mut self) -> PResult<Variant> {
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

	fn parse_fn_decl(&mut self) -> PResult<(Id, FnDecl)> {
		let name = self.expect_ident()?;
		let args_lo = self.token.span;
		let generics = self.parse_generics_def()?;
		self.expect(OpenParen)?;
		let params = self.parse_seq_rest(OpenParen, CloseParen, Comma, Parser::parse_param)?;
		let ret = if !self.check(OpenBrace) && !self.check(Semi) {
			Some(self.parse_ty()?)
		} else {
			None
		};

		let fn_decl = FnDecl {
			params,
			ret,
			span: self.close_span(args_lo),
		};
		Ok((name, fn_decl))
	}

	fn parse_param(&mut self) -> PResult<Param> {
		let name = self.expect_ident()?;
		self.expect(Colon)?;
		let ty = self.parse_ty()?;
		Ok(Param { name, ty })
	}

	fn parse_path(&mut self) -> PResult<Path> {
		let mut segments = Vec::new();
		segments.push(self.expect_ident()?);
		segments.extend(self.parse_while(ColonColon, Self::expect_ident)?);

		let generics = if self.check(Lt) {
			self.parse_ty_generics()?
		} else {
			Vec::new()
		};

		Ok(Path { segments, generics })
	}

	fn parse_generics_def(&mut self) -> PResult<Generics> {
		if !self.check(Lt) {
			return Ok(Generics(vec![]));
		}

		// TODO: this is a modified expansion of
		// let (generics, span) = self.parse_seq(Angled, Comma, Self::expect_ident)?;

		let mut finished = false;
		let mut generics = Vec::new();

		self.expect(Lt)?;
		while !self.eat(Gt) && !finished {
			generics.push(self.expect_ident()?);

			// no comma means no item left
			finished = !self.eat(Comma);
		}

		Ok(Generics(generics))
	}

	/// Parse [`ExprKind::FnCall`]
	fn parse_fn_call(&mut self, expr: Expr) -> PResult<ExprKind> {
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
}

/// Types
impl Parser<'_> {
	fn parse_ty(&mut self) -> PResult<Ty> {
		let lo = self.token.span;

		let kind = if matches!(self.token.kind, Ident(_)) {
			self.parse_ty_path()?
		} else if self.eat(Star) {
			self.parse_ty_pointer()?
		} else if self.eat(Ampersand) {
			self.parse_ty_reference()?
		} else {
			let report = errors::parser::expected_construct_no_match("a type", self.token);
			return Err(Diagnostic::new(report));
		};

		Ok(Ty {
			kind,
			span: self.close_span(lo),
		})
	}

	fn parse_ty_path(&mut self) -> PResult<TyKind> {
		debug_assert!(matches!(self.token.kind, Ident(_)));

		let path = self.parse_path()?;

		Ok(TyKind::Path(path))
	}

	/// Parse [`TyKind::Reference`]
	fn parse_ty_reference(&mut self) -> PResult<TyKind> {
		debug_assert_eq!(self.last_token.kind, Ampersand);

		let ty = Box::new(self.parse_ty()?);

		Ok(TyKind::Reference(ty))
	}

	/// Parse [`TyKind::Pointer`]
	fn parse_ty_pointer(&mut self) -> PResult<TyKind> {
		debug_assert_eq!(self.last_token.kind, Star);

		let ty = Box::new(self.parse_ty()?);

		Ok(TyKind::Pointer(ty))
	}

	/// Parse `"<" <ty> ">"`
	fn parse_ty_generics(&mut self) -> PResult<Vec<Ty>> {
		let mut finished = false;

		let mut seq = Vec::new();

		self.expect(Lt)?;
		while !self.eat(Gt) && !finished {
			seq.push(self.parse_ty()?);

			// no comma means no item left
			finished = !self.eat(Comma);
		}
		Ok(seq)
	}
}

/// Statements
impl Parser<'_> {
	fn parse_stmt(&mut self) -> PResult<Stmt> {
		let lo = self.token.span;
		let kind = match self.token.kind {
			// Keyword(For) => self.parse_stmt_for()?,
			Semi => {
				self.expect(Semi)?;
				StmtKind::Empty
			}

			Keyword(Kw::Let) => self.parse_stmt_let()?,

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
		})
	}

	fn parse_expr_loop(&mut self) -> PResult<ExprKind> {
		self.expect(Keyword(Kw::Loop))?;
		let body = Box::new(self.parse_block()?);
		Ok(ExprKind::Loop { body })
	}

	fn parse_stmt_let(&mut self) -> PResult<StmtKind> {
		self.expect(Keyword(Kw::Let))?;

		let mutable = self.eat(Keyword(Kw::Mut));

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
	fn parse_block(&mut self) -> PResult<Block> {
		let lo = self.token.span;
		self.expect(OpenBrace)?;
		let stmts = self.parse_until_func(CloseBrace, Self::parse_stmt)?;

		Ok(Block {
			stmts,
			span: self.close_span(lo),
		})
	}

	fn parse_attr(&mut self, kind: &AttrKind) -> PResult<Attr> {
		match kind {
			AttrKind::Parent => debug_assert!(matches!(self.last_token.kind, PoundPound)),
			AttrKind::Next => debug_assert!(matches!(self.last_token.kind, Pound)),
		}

		let lo = self.last_token.span;

		let path = self.parse_path()?;

		let meta = if self.eat(OpenParen) {
			let exprs = self.parse_seq_rest(OpenParen, CloseParen, Comma, Parser::parse_expr)?;
			AttrMeta::Tuple(exprs)
		} else if self.eat(OpenBrace) {
			let exprs = self.parse_seq_rest(OpenBrace, CloseBrace, Comma, Parser::parse_expr)?;
			AttrMeta::Map(todo!())
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
			span: self.close_span(lo),
		})
	}
}

/// What should the attr attach to
enum AttrKind {
	/// `##path`
	Parent,
	/// `#path`
	Next,
}
