//! Source code to tokens lexing logic

use core::fmt;
use std::str::Chars;

use crate::ast::Ident;
use crate::session::{BytePos, SessionCtx, Span, Symbol};

#[allow(clippy::enum_glob_use)]
use crate::lexer::{Keyword::*, TokenKind::*};

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
			(Bang, Eq) => Ne,

			(Gt, Eq) => Ge,
			(Lt, Eq) => Le,

			(Lt, Lt) => Shl,
			(Gt, Gt) => Shr,

			(Colon, Colon) => ColonColon,
			(Pound, Pound) => PoundPound,

			(Ampersand, Ampersand) => todo!("for recovery, see `and` kw"),
			(BitwiseOr, BitwiseOr) => todo!("for recovery, see `or` kw"),

			(TokenKind::Ident(ident), LiteralStr(sym)) => todo!("resolve cstr and custom strings"),

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
	LiteralStr(Symbol),
	LiteralInt(Symbol),
	LiteralFloat(Symbol),

	OpenParen,
	CloseParen,
	OpenBracket,
	CloseBracket,
	OpenBrace,
	CloseBrace,

	/// `!`
	Bang,
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
	BitwiseAnd,
	/// `|`
	BitwiseOr,
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
	/// `#`
	Pound,
	/// `=`
	Eq,
	/// `::`
	ColonColon,
	/// `'`
	Apostrophe,
	/// `##`
	PoundPound,

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
			LiteralStr(_) => write!(f, "a string literal"),
			LiteralInt(_) => write!(f, "a integer literal"),
			LiteralFloat(_) => write!(f, "a float literal"),

			OpenParen => write!(f, "an opening parenthesis"),
			CloseParen => write!(f, "a closing parenthesis"),
			OpenBracket => write!(f, "an opening bracket"),
			CloseBracket => write!(f, "a closing bracket"),
			OpenBrace => write!(f, "an opening brace"),
			CloseBrace => write!(f, "a closing brace"),

			Bang => write!(f, "a logical negation"),
			Plus => write!(f, "a plus operator"),
			Dash => write!(f, "a minus operator"),
			Star => write!(f, "a multiplication operator"),
			Div => write!(f, "a division operator"),
			Mod => write!(f, "a modulo operator"),

			BitwiseAnd => write!(f, "an and operator"),
			BitwiseOr => write!(f, "an or operator"),
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
			Apostrophe => write!(f, "an apostrophe"),
			Pound => write!(f, "a pound sign"),
			PoundPound => write!(f, "a double pound sign"),

			Unknown => write!(f, "an unknown token"),
			Eof => write!(f, "the end of the file"),
		}
	}
}

// TODO: should keywords be a verb? def (function), let (variables), ??? (type)
// or should we keep names like fn, trait, var

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
	Fn,
	// Decl,
	//
	Unsafe,
	Extern,
	Type,
	Struct,
	Enum,

	Impl,
	Trait,
	For,
	Loop,
	While,
	If,
	Else,
	Is,
	Match,

	Let,
	Mut,

	And,
	Or,
	Not,

	Return,
	Break,
	Continue,
}

const EOF_CHAR: char = '\0';

#[derive(Debug, Clone)]
pub struct Lexer<'scx, 'src> {
	scx: &'scx SessionCtx,

	source: &'src str,
	start_pos: BytePos,
	chars: Chars<'src>,
	token: Option<char>,
	offset: BytePos,

	next_glued: Option<Token>,
}

impl<'scx, 'src> Lexer<'scx, 'src> {
	#[must_use]
	pub fn new(scx: &'scx SessionCtx, source: &'src str, start_pos: BytePos) -> Self {
		let chars = source.chars();
		Self {
			scx,
			source,
			start_pos,
			chars,
			token: None,
			offset: start_pos,

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

	fn pos_to_source_idx(&self, pos: BytePos) -> usize {
		(pos - self.start_pos).to_usize()
	}

	fn str_from_to(&self, start: BytePos, end: BytePos) -> &str {
		&self.source[self.pos_to_source_idx(start)..self.pos_to_source_idx(end)]
	}

	fn str_from(&self, start: BytePos) -> &str {
		self.str_from_to(start, self.offset)
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
						// "decl" => Keyword(Decl),
						"unsafe" => Keyword(Unsafe),
						"extern" => Keyword(Extern),
						"type" => Keyword(Type),
						"struct" => Keyword(Struct),
						"enum" => Keyword(Enum),

						"impl" => Keyword(Impl),
						"trait" => Keyword(Trait),
						"for" => Keyword(For),
						"loop" => Keyword(Loop),
						"while" => Keyword(While),
						"if" => Keyword(If),
						"else" => Keyword(Else),
						"is" => Keyword(Is),
						"match" => Keyword(Match),

						"let" => Keyword(Let),
						"mut" => Keyword(Mut),

						"and" => Keyword(And),
						"or" => Keyword(Or),
						"not" => Keyword(Not),

						"return" => Keyword(Return),
						"break" => Keyword(Break),
						"continue" => Keyword(Continue),

						ident => Ident(self.scx.symbols.intern(ident)),
					}
				}

				// Int or Float
				c if c.is_ascii_digit() => {
					self.bump_while(|c| char::is_ascii_digit(&c));
					// avoid to eat the dot if this is a mac call after
					if self.first() == '.' && !is_ident_start(self.second()) {
						self.bump();

						// TODO: ensure that the float indeed has a digit after the dot
						assert!(char::is_ascii_digit(&self.first()));
						self.bump_while(|c| char::is_ascii_digit(&c));
						LiteralFloat(self.scx.symbols.intern(self.str_from(start)))
					} else {
						LiteralInt(self.scx.symbols.intern(self.str_from(start)))
					}
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
					LiteralStr(self.scx.symbols.intern(symbol))
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

				'!' => Bang,

				',' => Comma,
				'.' => Dot,
				':' => Colon,
				';' => Semi,

				'&' => Ampersand,
				'#' => Pound,

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
