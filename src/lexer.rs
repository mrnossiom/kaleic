//! Source code to tokens lexing logic

use core::fmt;
use std::str::Chars;

use crate::ast::Ident;
use crate::session::{BytePos, SessionCtx, Span, Symbol};

#[allow(clippy::enum_glob_use)]
use crate::lexer::{Keyword::*, LiteralKind::*, TokenKind::*};

#[derive(Debug, PartialEq, Eq)]
pub enum Spacing {
	Alone,
	Joint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
	pub kind: TokenKind,
	pub span: Span,
}

impl Token {
	pub const DUMMY: Self = Self::new(Eof, Span::DUMMY);

	#[must_use]
	pub const fn new(kind: TokenKind, span: Span) -> Self {
		Self { kind, span }
	}

	fn maybe_glue_joint(&self, next: &Self) -> Option<Self> {
		let glued_kind = match (self.kind, next.kind) {
			(Eq, Eq) => EqEq,
			(Not, Eq) => Ne,

			(Gt, Eq) => Ge,
			(Lt, Eq) => Le,

			(Lt, Lt) => Shl,
			(Gt, Gt) => Shr,

			(Colon, Colon) => ColonColon,

			(Ampersand, Ampersand) => todo!("for recovery, see `and` kw"),
			(Or, Or) => todo!("for recovery, see `or` kw"),

			(_, _) => return None,
		};

		Some(Self::new(glued_kind, self.span.to(next.span)))
	}

	#[must_use]
	pub const fn as_ident(self) -> Option<Ident> {
		match self.kind {
			Ident(sym) => Some(Ident::new(sym, self.span)),
			_ => None,
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
	Ident(Symbol),
	Keyword(Keyword),
	Literal(LiteralKind, Symbol),

	OpenParen,
	CloseParen,
	OpenBracket,
	CloseBracket,
	OpenBrace,
	CloseBrace,

	/// `!`
	Not,
	/// `+`
	Plus,
	/// `-`
	Dash,
	/// `*`
	Star,
	/// `/`
	Div,
	/// `%`
	Mod,
	/// `&`
	And,
	/// `|`
	Or,
	/// `^`
	Xor,
	/// `<<`
	Shl,
	/// `>>`
	Shr,
	/// `>`
	Gt,
	/// `>=`
	Ge,
	/// `<`
	Lt,
	/// `<=`
	Le,
	/// `==`
	EqEq,
	/// `!=`
	Ne,
	/// `,`
	Comma,
	/// `:`
	Colon,
	/// `;`
	Semi,
	/// `.`
	Dot,
	/// `&`
	Ampersand,
	/// `=`
	Eq,
	/// `::`
	ColonColon,

	/// Fallback token for unrecognized lexeme
	Unknown,
	/// Used to reduce `Option` boilerplate
	Eof,
}

impl fmt::Display for TokenKind {
	/// Should fit in the sentence "found {}"
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Ident(_) => write!(f, "an identifier"),
			Keyword(_) => write!(f, "a keyword"),
			Literal(kind, _) => write!(f, "a {kind} literal"),

			OpenParen => write!(f, "an opening parenthesis"),
			CloseParen => write!(f, "a closing parenthesis"),
			OpenBracket => write!(f, "an opening bracket"),
			CloseBracket => write!(f, "a closing bracket"),
			OpenBrace => write!(f, "an opening brace"),
			CloseBrace => write!(f, "a closing brace"),

			Not => write!(f, "a logical negation"),
			Plus => write!(f, "a plus operator"),
			Dash => write!(f, "a minus operator"),
			Star => write!(f, "a multiplication operator"),
			Div => write!(f, "a division operator"),
			Mod => write!(f, "a modulo operator"),

			And => write!(f, "an and operator"),
			Or => write!(f, "an or operator"),
			Xor => write!(f, "a xor operator"),

			Shl => write!(f, "a shift left operator"),
			Shr => write!(f, "a shift right operator"),

			Gt => write!(f, "a greater than comparator"),
			Ge => write!(f, "a greater or equal comparator"),
			Lt => write!(f, "a lesser than comparator"),
			Le => write!(f, "a lesser or equal comparator"),

			EqEq => write!(f, "a equal comparator"),
			Ne => write!(f, "a different comparator"),

			Comma => write!(f, "a comma"),
			Colon => write!(f, "a colon"),
			Semi => write!(f, "a semicolon"),
			Dot => write!(f, "a dot"),
			Ampersand => write!(f, "an ampersand"),

			Eq => write!(f, "an assign sign"),

			ColonColon => write!(f, "a path separator"),

			Unknown => write!(f, "an unknown token"),
			Eof => write!(f, "the end of the file"),
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiteralKind {
	Integer,
	Float,
	Str,
}

impl fmt::Display for LiteralKind {
	/// Should fit in the sentence "a {} literal"
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Integer => write!(f, "integer"),
			Float => write!(f, "float"),
			Str => write!(f, "string"),
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
	Fn,
	Type,
	Extern,
	Struct,
	Enum,
	Trait,
	For,
	Impl,

	Var,
	Cst,

	If,
	Else,
	Is,

	Loop,
	While,

	Return,
	Break,
	Continue,
}

const EOF_CHAR: char = '\0';

#[derive(Debug, Clone)]
pub struct Lexer<'scx, 'src> {
	scx: &'scx SessionCtx,

	source: &'src str,
	chars: Chars<'src>,
	token: Option<char>,
	offset: BytePos,

	next_glued: Option<Token>,
}

impl<'scx, 'src> Lexer<'scx, 'src> {
	#[must_use]
	pub fn new(scx: &'scx SessionCtx, source: &'src str, offset: BytePos) -> Self {
		let chars = source.chars();
		Self {
			scx,
			source,
			chars,
			token: None,
			offset,

			next_glued: None,
		}
	}

	fn bump(&mut self) -> Option<char> {
		self.token = self.chars.next();
		self.offset = self.offset + BytePos::from_usize(self.token.map_or(0, char::len_utf8));
		self.token
	}

	fn first(&self) -> char {
		// TODO: is the clone cheap? or should we have extra logic like peekable
		self.chars.clone().next().unwrap_or(EOF_CHAR)
	}

	fn second(&self) -> char {
		self.chars.clone().nth(1).unwrap_or(EOF_CHAR)
	}

	fn bump_while(&mut self, mut cond: impl FnMut(char) -> bool) {
		while cond(self.first()) && !self.is_eof() {
			self.bump();
		}
	}

	fn str_from_to(&self, start: BytePos, end: BytePos) -> &str {
		&self.source[start.to_usize()..end.to_usize()]
	}

	fn str_from(&self, start: BytePos) -> &str {
		&self.source[start.to_usize()..self.offset.to_usize()]
	}

	fn is_eof(&self) -> bool {
		self.chars.as_str().is_empty()
	}
}

impl Lexer<'_, '_> {
	pub fn next_token(&mut self) -> Option<(Token, Spacing)> {
		let mut spacing = Spacing::Joint;

		loop {
			let start = self.offset;

			let kind = match self.bump()? {
				c if is_ident_start(c) => {
					self.bump_while(is_ident_continue);
					// TODO: make kw an symbol wrapper with preinterned value
					match self.str_from(start) {
						"fn" => Keyword(Fn),
						"type" => Keyword(Type),
						"extern" => Keyword(Extern),
						"struct" => Keyword(Struct),
						"enum" => Keyword(Enum),
						"trait" => Keyword(Trait),
						"for" => Keyword(For),
						"impl" => Keyword(Impl),

						"var" => Keyword(Var),
						"cst" => Keyword(Cst),

						"if" => Keyword(If),
						"else" => Keyword(Else),
						"is" => Keyword(Is),

						"loop" => Keyword(Loop),
						"while" => Keyword(While),

						"return" => Keyword(Return),
						"break" => Keyword(Break),
						"continue" => Keyword(Continue),

						ident => TokenKind::Ident(self.scx.symbols.intern(ident)),
					}
				}

				// Int or Float
				c if c.is_ascii_digit() => {
					self.bump_while(|c| char::is_ascii_digit(&c));
					// avoid to eat the dot if this is a mac call after
					let kind = if self.first() == '.' && !is_ident_start(self.second()) {
						self.bump();
						// TODO: ensure that the float indeed has a digit after the dot
						assert!(self.token.is_some_and(|c| char::is_ascii_digit(&c)));
						self.bump_while(|c| char::is_ascii_digit(&c));
						Float
					} else {
						Integer
					};
					Literal(kind, self.scx.symbols.intern(self.str_from(start)))
				}

				'"' => {
					while let Some(c) = self.bump() {
						match c {
							'\\' if self.first() == '\\' || self.first() == '"' => {
								// skip escaped character
								self.bump();
							}
							'"' => break,
							_ => {}
						}
					}

					// strip quotes
					let symbol = self.str_from_to(
						start + BytePos::from_u32(1),
						self.offset - BytePos::from_u32(1),
					);
					Literal(Str, self.scx.symbols.intern(symbol))
				}

				// Non-significative whitespace
				c if c.is_ascii_whitespace() => {
					spacing = Spacing::Alone;
					continue;
				}

				// Delimiters
				'(' => OpenParen,
				')' => CloseParen,
				'[' => OpenBracket,
				']' => CloseBracket,
				'{' => OpenBrace,
				'}' => CloseBrace,

				'+' => Plus,
				'-' => Dash,
				'*' => Star,
				'/' => match self.first() {
					'/' => {
						// eat the whole line
						self.bump_while(|c| c != '\n');
						spacing = Spacing::Alone;
						continue;
					}
					'*' => {
						// eat the star
						self.bump();
						self.skip_block_comment();
						spacing = Spacing::Alone;
						continue;
					}
					_ => Div,
				},
				'%' => Mod,

				'>' => Gt,
				'<' => Lt,
				'=' => Eq,

				'!' => Not,

				',' => Comma,
				'.' => Dot,
				':' => Colon,
				';' => Semi,

				'&' => Ampersand,

				_ => Unknown,
			};

			let span = Span::new(start, self.offset);
			let token = Token { kind, span };
			return Some((token, spacing));
		}
	}

	fn next_token_glued(&mut self) -> Option<Token> {
		let mut token = self
			.next_glued
			.take()
			.or_else(|| self.next_token().map(|(tkn, _spacing)| tkn))?;

		loop {
			// maybe glue joint token if applicable
			if let Some((next, spacing)) = self.next_token() {
				if spacing == Spacing::Joint
					&& let Some(glued_token) = token.maybe_glue_joint(&next)
				{
					token = glued_token;
				} else {
					// save token for next iteration
					self.next_glued = Some(next);
					return Some(token);
				}
			} else {
				return Some(token);
			}
		}
	}

	fn skip_block_comment(&mut self) {
		let mut count = 0;

		// handle nested block comments
		while let Some(c) = self.bump() {
			match c {
				'/' if self.first() == '*' => count += 1,
				'*' if self.first() == '/' && count == 0 => {
					// eat the trailing slash
					self.bump();
					break;
				}
				'*' if self.first() == '/' => count -= 1,
				_ => {}
			}
		}
	}
}

impl Iterator for Lexer<'_, '_> {
	type Item = Token;
	fn next(&mut self) -> Option<Self::Item> {
		self.next_token_glued()
	}
}

const fn is_ident_start(c: char) -> bool {
	c.is_ascii_alphabetic() || c == '_'
}

const fn is_ident_continue(c: char) -> bool {
	c.is_ascii_alphanumeric() || c == '_'
}
