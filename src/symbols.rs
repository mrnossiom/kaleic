use core::fmt;
use std::num::NonZero;

use parking_lot::RwLock;
use string_interner::Symbol as _;
use string_interner::{StringInterner, backend::StringBackend};

use crate::bug;

macro_rules! define_symbols {
    {
        keywords { $($kw_name:ident: $kw_str:literal),* , },
        symbols { $($sym_name:ident $(: $sym_str:literal)?),* , }
    } => {
    	define_symbols!(@complete
			keywords { $($kw_name: $kw_str),* },
			symbols { $($sym_name: define_symbols!(@or_stringify $sym_name $($sym_str)?)),* }
    	);
    };
    (@or_stringify $name:ident) => { stringify!($name) };
    (@or_stringify $name:ident $str:literal) => { $str };
    {
    	@complete
        keywords { $($kw_name:ident: $kw_str:expr),* },
        symbols { $($sym_name:ident: $sym_str:expr),* }
	} => {
        pub(crate) static SYMBOLS: &[&str] = &[
            $($kw_str),*,
            $($sym_str),*
        ];

        #[allow(non_upper_case_globals)]
        pub(crate) mod kw {
        	use super::{Id, Symbol};
            $(pub const $kw_name: Symbol = Symbol::new(Id::$kw_name as u32).unwrap();)*
        }
        #[allow(non_upper_case_globals)]
        pub(crate) mod sym {
        	use super::{Id, Symbol};
            $(pub const $sym_name: Symbol = Symbol::new(Id::$sym_name as u32).unwrap();)*
        }

        // this enum is used to assign unique identifiers
        #[allow(non_camel_case_types)]
        #[repr(u32)]
        enum Id {
            $($kw_name),*,
            $($sym_name),*
        }
	};
}

define_symbols! {
	keywords {
		// keep in sync with `is_keyword`
		And: "and",
		Break: "break",
		Continue: "continue",
		Else: "else",
		Enum: "enum",
		Extern: "extern",
		Fn: "fn",
		For: "for",
		If: "if",
		Impl: "impl",
		Is: "is",
		Let: "let",
		Loop: "loop",
		Match: "match",
		Mut: "mut",
		Not: "not",
		Or: "or",
		Return: "return",
		Struct: "struct",
		Trait: "trait",
		Type: "type",
		Unsafe: "unsafe",
		While: "while",

		Underscore: "_",
	},
	symbols {
		// values
		argc,
		argv,
		main,

		true_: "true",
		false_: "false",

		// types
		bool,
		float,
		never,
		sint,
		str,
		uint,

		// lang items
		AddTrait,
		AddAssignTrait,
		SubTrait,
		SubAssignTrait,

		never_ty,
		bool_ty,
		uint_ty,
		sint_ty,
		float_ty,

		// attrs
		lang_item,
		link,
	}
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Symbol(NonZero<u32>);

impl Symbol {
	pub(crate) const fn new(index: u32) -> Option<Self> {
		if let Some(value) = NonZero::new((index).wrapping_add(1)) {
			Some(Self(value))
		} else {
			None
		}
	}

	pub(crate) const fn get(self) -> u32 {
		self.0.get().strict_sub(1)
	}

	pub(crate) const fn is_keyword(self) -> bool {
		self.get() <= kw::While.get()
	}
}

impl string_interner::Symbol for Symbol {
	#[inline]
	fn try_from_usize(index: usize) -> Option<Self> {
		Self::new(u32::try_from(index).unwrap())
	}

	#[inline]
	fn to_usize(self) -> usize {
		self.get() as usize
	}
}

impl fmt::Debug for Symbol {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		#[cfg(feature = "debug")]
		let interned = INTERNER.with(|i| {
			i.get().map_or(Ok(false), |i| {
				i.read().resolve(*self).map_or(Ok(false), |str| {
					if f.alternate() {
						write!(f, "{str}").map(|()| true)
					} else {
						write!(f, "`{str}`#{}", self.get()).map(|()| true)
					}
				})
			})
		})?;
		#[cfg(not(feature = "debug"))]
		let interned = false;

		if !interned {
			write!(f, "sym#{:?}", self.to_usize())?;
		}

		Ok(())
	}
}

#[cfg(feature = "debug")]
thread_local! {
	static INTERNER: std::sync::OnceLock<std::sync::Arc<RwLock<StringInterner<StringBackend<Symbol>>>>> = std::sync::OnceLock::default();
}

pub(crate) struct SymbolInterner {
	#[cfg(feature = "debug")]
	inner: std::sync::Arc<RwLock<StringInterner<StringBackend<Symbol>>>>,
	#[cfg(not(feature = "debug"))]
	inner: RwLock<StringInterner<StringBackend<Symbol>>>,
}

impl fmt::Debug for SymbolInterner {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("SymbolInterner").finish_non_exhaustive()
	}
}

impl Default for SymbolInterner {
	fn default() -> Self {
		let interner = StringInterner::from_iter(SYMBOLS);
		let inner = RwLock::new(interner);
		#[cfg(feature = "debug")]
		let inner = {
			let inner = std::sync::Arc::new(inner);
			_ = INTERNER.with(|i| i.set(inner.clone()));
			inner
		};
		Self { inner }
	}
}

impl SymbolInterner {
	#[must_use]
	pub(crate) fn intern(&self, symbol: &str) -> Symbol {
		self.inner.write().get_or_intern(symbol)
	}

	#[must_use]
	pub(crate) fn resolve(&self, symbol: Symbol) -> String {
		match self.inner.read().resolve(symbol) {
			Some(s) => s.to_owned(),
			None => bug!("there is a single symbol interner, thus all symbol are valid"),
		}
	}
}
