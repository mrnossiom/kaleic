//! Tokens to AST parsing logic
//!
//! Contains the recursive decent parser of the language.
//!
//! Entrypoint to parsing is [`Parser::parse_root`].

use std::{fmt, mem, ops::ControlFlow};

use ariadne::{Label, Report, ReportKind};

#[allow(clippy::enum_glob_use)]
use crate::lexer::{Keyword::*, LiteralKind::*, TokenKind::*};
use crate::{
	ast::{
		BinaryOp, Block, Expr, ExprKind, FieldDef, FnDecl, Function, Ident, Item, ItemKind, NodeId,
		Param, Path, Root, Spanned, Stmt, StmtKind, TraitItem, TraitItemKind, Ty, TyKind,
		TypeAlias, UnaryOp, Variant, VariantKind,
	},
	bug, errors,
	lexer::{Lexer, Token, TokenKind},
	session::{Diagnostic, SessionCtx, SourceFile, Span},
};

macro_rules! fn_name {
	() => {{
		const fn f() {}
		fn type_name_of<T>(_: T) -> &'static str {
			std::any::type_name::<T>()
		}
		let name = type_name_of(f);
		let mut segments = name.rsplit("::");
		// skip `f`
		_ = segments.next();
		segments.next().unwrap()
	}};
}

macro_rules! debug_parser {
	($self:expr) => {
		tracing::trace!(tkn = ?$self.token, "{:<30}", fn_name!());
	}
}

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
		let mut parser = Self {
			scx,

			lexer: Lexer::new(scx, &file.content, file.offset),

			token: Token::DUMMY,
			last_token: Token::DUMMY,

			next_node_id: 0,
		};

		// init the first token
		parser.bump();

		parser
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

	fn eat_ident(&mut self) -> Option<Ident> {
		self.token.as_ident().inspect(|_| {
			self.bump();
		})
	}

	fn expect_ident(&mut self) -> PResult<Ident> {
		self.eat_ident().ok_or_else(|| {
			let placeholder = self.scx.symbols.intern("_");
			let report =
				errors::parser::expected_token_kind(TokenKind::Ident(placeholder), self.token);
			Diagnostic::new(report)
		})
	}

	fn close_span(&self, start: Span) -> Span {
		start.to(self.last_token.span)
	}

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

	fn parse_while<T>(
		&mut self,
		sep: TokenKind,
		mut parse: impl FnMut(&mut Self) -> PResult<T>,
	) -> PResult<Vec<T>> {
		let mut many = Vec::new();
		while self.eat(sep) {
			many.push(parse(self)?);
		}
		Ok(many)
	}

	/// Looks one token ahead
	#[expect(dead_code)]
	fn look_ahead(&self) -> TokenKind {
		self.lexer.clone().next().map_or(Eof, |tkn| tkn.kind)
	}

	fn make_node_id(&mut self) -> NodeId {
		let node_id = NodeId(self.next_node_id);
		self.next_node_id = self
			.next_node_id
			.checked_add(1)
			.unwrap_or_else(|| bug!("tried to construct too much `parser::NodeId`s"));
		node_id
	}
}

/// Expressions
impl Parser<'_> {
	/// Parse an expression
	fn parse_expr(&mut self) -> PResult<Expr> {
		debug_parser!(self);

		let lhs = self.parse_expr_single_and_postfix()?;
		self.parse_expr_assoc_rest(None, lhs)
	}

	/// Parse an expression right-hand side by eating association operators
	/// (e.g. binary operators or assignment equal) while their precedence is higher.
	fn parse_expr_assoc_rest(&mut self, precedence: Option<u32>, mut lhs: Expr) -> PResult<Expr> {
		debug_parser!(self);

		let lo = self.token.span;

		while let Some(assoc_op) = self.eat_assoc_token_with_precedence(precedence) {
			let left = Box::new(lhs);
			let right = Box::new(self.parse_expr()?);

			let new_kind = match assoc_op.bit {
				AssocOp::Binary(bin_op) => {
					let op = Spanned::new(bin_op, assoc_op.span);
					ExprKind::Binary { op, left, right }
				}
				AssocOp::Assign => ExprKind::Assign {
					target: left,
					value: right,
				},
			};

			lhs = Expr {
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
		debug_parser!(self);

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
		debug_parser!(self);

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
					let args =
						self.parse_seq_rest(OpenParen, CloseParen, Comma, Parser::parse_expr)?;
					ExprKind::Method(Box::new(expr), field, args)
				} else {
					ExprKind::Field(Box::new(expr), field)
				}
			} else if self.eat(Star) {
				// `<expr> . *`
				ExprKind::Deref(Box::new(expr))
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
			id: self.make_node_id(),
		}))
	}

	/// Parse a single expression without eating binary operators
	///
	/// See [`Self::parse_expr`] for full expression parsing including binary operations
	fn parse_expr_single(&mut self) -> PResult<Expr> {
		debug_parser!(self);

		let lo = self.token.span;

		let kind = if self.eat(Not) {
			self.parse_expr_not()?
		} else if self.eat(Dash) {
			self.parse_expr_neg()?
		} else if matches!(self.token.kind, TokenKind::Ident(_)) {
			self.parse_expr_access()?
		} else if matches!(self.token.kind, Literal(_, _)) {
			self.parse_expr_literal()
		} else if self.eat(OpenParen) {
			self.parse_expr_paren()?
		} else if self.eat(Keyword(If)) {
			self.parse_expr_if()?
		}
		// TODO: make loops be expressions
		// else if self.eat(Keyword(While) {
		// 	self.parse_expr_while()?
		// }
		else if self.eat(Keyword(Return)) {
			self.parse_expr_return()?
		} else if self.eat(Keyword(Break)) {
			self.parse_expr_break()?
		} else if self.eat(Keyword(Continue)) {
			self.parse_expr_continue()?
		} else {
			let report = errors::parser::expected_construct_no_match("an expression", self.token);
			return Err(Diagnostic::new(report));
		};

		Ok(Expr {
			kind,
			span: self.close_span(lo),
			id: self.make_node_id(),
		})
	}

	/// Parse [`ExprKind::Unary`] for [`UnaryOp::Not`]
	fn parse_expr_not(&mut self) -> PResult<ExprKind> {
		debug_parser!(self);
		debug_assert_eq!(self.last_token.kind, Not);

		let expr = Box::new(self.parse_expr()?);

		let op = Spanned::new(UnaryOp::Not, self.last_token.span);
		Ok(ExprKind::Unary { op, expr })
	}

	/// Parse [`ExprKind::Unary`] for [`UnaryOp::Minus`]
	fn parse_expr_neg(&mut self) -> PResult<ExprKind> {
		debug_parser!(self);
		debug_assert_eq!(self.last_token.kind, Dash);

		let expr = Box::new(self.parse_expr()?);

		let op = Spanned::new(UnaryOp::Minus, self.last_token.span);
		Ok(ExprKind::Unary { op, expr })
	}

	/// Parse [`ExprKind::Literal`]
	fn parse_expr_literal(&mut self) -> ExprKind {
		debug_parser!(self);
		assert!(matches!(self.token.kind, TokenKind::Literal(_, _)));

		let TokenKind::Literal(kind, sym) = self.token.kind else {
			bug!("should be called when on a literal");
		};
		self.bump();

		match kind {
			Integer => ExprKind::Literal(Integer, sym),
			Float => ExprKind::Literal(Float, sym),
			// handle prefixed strings (e.g. c"content")
			Str => ExprKind::Literal(Str, sym),
		}
	}

	/// Parse [`ExprKind::Access`]
	fn parse_expr_access(&mut self) -> PResult<ExprKind> {
		debug_parser!(self);

		let path = self.parse_path()?;

		Ok(ExprKind::Access(path))
	}

	/// Parse [`ExprKind::Paren`]
	fn parse_expr_paren(&mut self) -> PResult<ExprKind> {
		debug_parser!(self);
		debug_assert_eq!(self.last_token.kind, OpenParen);

		let expr = Box::new(self.parse_expr()?);
		self.expect(CloseParen)?;

		Ok(ExprKind::Paren(expr))
	}

	/// Parse [`ExprKind::If`]
	fn parse_expr_if(&mut self) -> PResult<ExprKind> {
		debug_parser!(self);
		debug_assert_eq!(self.last_token.kind, Keyword(If));

		let cond = Box::new(self.parse_expr()?);
		let conseq = Box::new(self.parse_block()?);
		let altern = if self.eat(Keyword(Else)) {
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

	/// Parse [`ExprKind::Return`]
	fn parse_expr_return(&mut self) -> PResult<ExprKind> {
		debug_parser!(self);
		debug_assert_eq!(self.last_token.kind, Keyword(Return));

		// TODO: bad for recovery
		let expr = self.parse_expr().ok().map(Box::new);

		Ok(ExprKind::Return(expr))
	}

	/// Parse [`ExprKind::Break`]
	fn parse_expr_break(&mut self) -> PResult<ExprKind> {
		debug_parser!(self);
		debug_assert_eq!(self.last_token.kind, Keyword(Break));

		let expr = self.parse_expr().ok().map(Box::new);

		Ok(ExprKind::Break(expr))
	}

	/// Parse [`ExprKind::Continue`]
	fn parse_expr_continue(&mut self) -> PResult<ExprKind> {
		debug_parser!(self);
		debug_assert_eq!(self.last_token.kind, Keyword(Continue));

		// TODO: parse label

		Ok(ExprKind::Continue)
	}
}

impl Parse for Root {
	fn parse(p: &mut Parser) -> Result<Self, Diagnostic> {
		debug_parser!(p);

		let items = p.parse_until::<Item>(Eof)?;

		Ok(Self { items })
	}
}

impl Parse for Item {
	fn parse(p: &mut Parser) -> Result<Self, Diagnostic> {
		debug_parser!(p);

		let lo = p.token.span;

		let kind = if p.eat(Keyword(Fn)) {
			ItemKind::Function(Parse::parse(p)?)
		} else if p.eat(Keyword(Extern)) {
			p.parse_item_extern()?
		} else if p.eat(Keyword(Struct)) {
			p.parse_item_struct()?
		} else if p.eat(Keyword(Enum)) {
			p.parse_item_enum()?
		} else if p.eat(Keyword(Trait)) {
			p.parse_item_trait()?
		} else if p.eat(Keyword(For)) {
			p.parse_item_trait_impl()?
		} else if p.eat(Keyword(Type)) {
			ItemKind::TypeAlias(TypeAlias::parse(p)?)
		} else {
			let report = errors::parser::expected_construct_no_match("an item", p.token);
			return Err(Diagnostic::new(report));
		};

		Ok(Self {
			kind,
			span: p.close_span(lo),
			id: p.make_node_id(),
		})
	}
}

impl Parse for Function {
	fn parse(p: &mut Parser) -> Result<Self, Diagnostic> {
		debug_parser!(p);
		debug_assert_eq!(p.last_token.kind, Keyword(Fn));

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

		Ok(Self {
			name,
			decl,
			body,
			abi: None,
		})
	}
}

impl Parse for TypeAlias {
	fn parse(p: &mut Parser) -> Result<Self, Diagnostic> {
		debug_parser!(p);
		debug_assert_eq!(p.last_token.kind, Keyword(Type));

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
	/// Parse [`Function`] with [`Function::externess`] set to some ABI.
	fn parse_item_extern(&mut self) -> PResult<ItemKind> {
		debug_parser!(self);
		debug_assert_eq!(self.last_token.kind, Keyword(Extern));

		let abi = self.parse_expr()?;
		self.expect(Keyword(Fn))?;
		let mut func = Function::parse(self)?;

		func.abi = Some(abi);

		Ok(ItemKind::Function(func))
	}

	/// Parse [`ItemKind::Struct`]
	fn parse_item_struct(&mut self) -> PResult<ItemKind> {
		debug_parser!(self);
		debug_assert_eq!(self.last_token.kind, Keyword(Struct));

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
					name: Ident::new(self.scx.symbols.intern(&i.to_string()), ty.span),
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
		debug_parser!(self);
		debug_assert_eq!(self.last_token.kind, Keyword(Enum));

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
		debug_parser!(self);
		debug_assert_eq!(self.last_token.kind, Keyword(Trait));

		let name = self.expect_ident()?;
		let generics = self.parse_generics_def()?;

		self.expect(OpenBrace)?;
		let members = self.parse_until_func(CloseBrace, Self::parse_trait_member)?;

		Ok(ItemKind::Trait {
			name,
			generics,
			members,
		})
	}

	/// Parse [`ItemKind::TraitImpl`]
	fn parse_item_trait_impl(&mut self) -> PResult<ItemKind> {
		debug_parser!(self);
		debug_assert_eq!(self.last_token.kind, Keyword(For));

		let type_ = self.parse_path()?;
		self.expect(Keyword(Impl))?;
		let trait_ = self.parse_path()?;
		self.expect(OpenBrace)?;
		let members = self.parse_until_func(CloseBrace, Self::parse_trait_member)?;

		Ok(ItemKind::TraitImpl {
			type_,
			trait_,
			members,
		})
	}

	/// Parse [`FieldDef`]
	fn parse_field_def(&mut self) -> PResult<FieldDef> {
		debug_parser!(self);

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
		debug_parser!(self);

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

	/// Parse [`TraitItem`]
	fn parse_trait_member(&mut self) -> PResult<TraitItem> {
		debug_parser!(self);

		let Item { kind, span, .. } = Item::parse(self)?;

		let kind = match kind {
			ItemKind::TypeAlias(type_) => TraitItemKind::Type(type_),
			ItemKind::Function(func) => TraitItemKind::Function(func),
			_ => {
				let report = errors::parser::incorrect_item_in_trait(span);
				return Err(Diagnostic::new(report));
			}
		};

		Ok(TraitItem { kind, span })
	}

	fn parse_fn_decl(&mut self) -> PResult<(Ident, FnDecl)> {
		debug_parser!(self);

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
		debug_parser!(self);

		let name = self.expect_ident()?;
		self.expect(Colon)?;
		let ty = self.parse_ty()?;
		Ok(Param { name, ty })
	}

	fn parse_path(&mut self) -> PResult<Path> {
		debug_parser!(self);

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

	fn parse_generics_def(&mut self) -> PResult<Vec<Ident>> {
		if !self.check(Lt) {
			return Ok(vec![]);
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

		Ok(generics)
	}

	/// Parse [`ExprKind::FnCall`]
	fn parse_fn_call(&mut self, expr: Expr) -> PResult<ExprKind> {
		debug_parser!(self);

		let args_lo = self.token.span;
		self.expect(OpenParen)?;
		let args = self.parse_seq_rest(OpenParen, CloseParen, Comma, Parser::parse_expr)?;

		Ok(ExprKind::FnCall {
			expr: Box::new(expr),
			args: Spanned::new(args, self.close_span(args_lo)),
		})
	}
}

/// Types
impl Parser<'_> {
	fn parse_ty(&mut self) -> PResult<Ty> {
		debug_parser!(self);

		let lo = self.token.span;

		let kind = if matches!(self.token.kind, Ident(_)) {
			self.parse_ty_path()?
		} else if self.eat(Ampersand) {
			self.parse_ty_pointer()?
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
		debug_parser!(self);
		debug_assert!(matches!(self.token.kind, Ident(_)));

		let path = self.parse_path()?;

		Ok(TyKind::Path(path))
	}

	/// Parse [`TyKind::Pointer`]
	fn parse_ty_pointer(&mut self) -> PResult<TyKind> {
		debug_parser!(self);
		debug_assert_eq!(self.last_token.kind, Ampersand);

		let ty = Box::new(self.parse_ty()?);

		Ok(TyKind::Pointer(ty))
	}

	/// Parse `"<" <ty> ">"`
	fn parse_ty_generics(&mut self) -> PResult<Vec<Ty>> {
		debug_parser!(self);

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
		debug_parser!(self);

		let lo = self.token.span;
		let kind = match self.token.kind {
			Keyword(Loop) => self.parse_stmt_loop()?,
			Keyword(While) => self.parse_stmt_while()?,
			// Keyword(For) => self.parse_stmt_for()?,
			Semi => {
				self.expect(Semi)?;
				StmtKind::Empty
			}

			Keyword(Var) => self.parse_stmt_var()?,
			Keyword(Cst) => self.parse_stmt_var()?,

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

	fn parse_stmt_loop(&mut self) -> PResult<StmtKind> {
		debug_parser!(self);

		self.expect(Keyword(Loop))?;
		let body = Box::new(self.parse_block()?);
		Ok(StmtKind::Loop { body })
	}

	fn parse_stmt_while(&mut self) -> PResult<StmtKind> {
		debug_parser!(self);

		self.expect(Keyword(While))?;
		let check = Box::new(self.parse_expr()?);
		let body = Box::new(self.parse_block()?);
		Ok(StmtKind::WhileLoop { check, body })
	}

	fn parse_stmt_var(&mut self) -> PResult<StmtKind> {
		debug_parser!(self);

		self.expect(Keyword(Var))?;
		let name = self.expect_ident()?;

		// definition with optional ty
		let ty = if self.eat(Colon) {
			Some(Box::new(self.parse_ty()?))
		} else {
			None
		};

		self.expect(Eq)?;

		let value = Box::new(self.parse_expr()?);
		self.expect(Semi)?;

		Ok(StmtKind::Let { name, ty, value })
	}
}

impl Parser<'_> {
	fn parse_block(&mut self) -> PResult<Block> {
		debug_parser!(self);

		let lo = self.token.span;
		self.expect(OpenBrace)?;
		let stmts = self.parse_until_func(CloseBrace, Self::parse_stmt)?;

		Ok(Block {
			stmts,
			span: self.close_span(lo),
			id: self.make_node_id(),
		})
	}
}
