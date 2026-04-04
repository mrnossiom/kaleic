use std::path::PathBuf;

use crate::{
	ast::{Attr, AttrMeta, AttrPath, ExprKind, Ident},
	resolve::LangItem,
	session::{Diagnostic, SessionCtx},
	symbols::{Symbol, sym},
};

type Result<T> = std::result::Result<T, Diagnostic>;

pub(crate) fn is_match(target: &[Ident], pattern: &[Symbol]) -> bool {
	target.len() == pattern.len()
		&& target
			.iter()
			.zip(pattern)
			.all(|(sample, target)| sample.sym == *target)
}

pub(crate) trait AttrParse: Sized {
	fn match_path(path: &AttrPath) -> bool;
	fn parse(scx: &SessionCtx, attr: &Attr) -> Result<Self>;
}

pub(crate) fn try_parse_attr<A: AttrParse>(scx: &SessionCtx, attr: &Attr) -> Option<Result<A>> {
	if A::match_path(&attr.path) {
		Some(A::parse(scx, attr))
	} else {
		None
	}
}

pub(crate) struct ModPath {
	pub(crate) path: PathBuf,
}

impl AttrParse for ModPath {
	fn match_path(path: &AttrPath) -> bool {
		is_match(&path.segments, &[sym::path])
	}
	fn parse(scx: &SessionCtx, attr: &Attr) -> Result<Self> {
		if let AttrMeta::Tuple(exprs) = &attr.meta
			&& let [expr] = exprs.as_slice()
			&& let ExprKind::LiteralStr { sym } = expr.kind
		{
			let path = PathBuf::from(&*scx.symbols.resolve(sym));
			Ok(Self { path })
		} else {
			todo!("wrong syntax for mod path")
		}
	}
}

pub(crate) struct RegisterLangItem {
	pub(crate) lang_item: LangItem,
}

impl AttrParse for RegisterLangItem {
	fn match_path(path: &AttrPath) -> bool {
		is_match(&path.segments, &[sym::lang_item])
	}

	fn parse(scx: &SessionCtx, attr: &Attr) -> Result<Self> {
		if let AttrMeta::Tuple(exprs) = &attr.meta
			&& let [expr] = exprs.as_slice()
			&& let ExprKind::LiteralStr { sym } = expr.kind
		{
			let Some(lang_item) = LangItem::parse(sym) else {
				let report = errors::invalid_lang_item(expr.span);
				return Err(Diagnostic::new(report));
			};

			Ok(Self { lang_item })
		} else {
			let report = todo!("report wrong syntax for `lang_item` attr");
			Err(Diagnostic::new(report))
		}
	}
}

pub(crate) struct NoStd {}

impl AttrParse for NoStd {
	fn match_path(path: &AttrPath) -> bool {
		is_match(&path.segments, &[sym::no_std])
	}

	fn parse(scx: &SessionCtx, attr: &Attr) -> Result<Self> {
		if matches!(&attr.meta, AttrMeta::None) {
			Ok(Self {})
		} else {
			todo!()
		}
	}
}

pub(crate) struct NoCore {}

impl AttrParse for NoCore {
	fn match_path(path: &AttrPath) -> bool {
		is_match(&path.segments, &[sym::no_core])
	}

	fn parse(scx: &SessionCtx, attr: &Attr) -> Result<Self> {
		if matches!(&attr.meta, AttrMeta::None) {
			Ok(Self {})
		} else {
			todo!()
		}
	}
}

mod errors {
	use ariadne::{Label, ReportKind};

	use crate::session::{Report, Span};

	pub fn invalid_lang_item(lang_item_span: Span) -> ariadne::ReportBuilder<Span, ReportKind> {
		Report::build(ReportKind::Error, lang_item_span)
			.with_message("language item does not exist")
			.with_label(Label::new(lang_item_span).with_message("here"))
	}
}
