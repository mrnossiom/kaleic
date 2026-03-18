use ariadne::{Label, ReportKind};

use crate::session::{Report, ReportBuilder, Span};

pub mod parser {
	use crate::lexer::{Token, TokenKind};

	use super::*;

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
}

pub mod lowerer {
	use super::*;

	pub fn no_semicolon_mid_block(expr_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, expr_span)
			.with_message("expression is missing a semicolon but is not at the end")
			.with_label(Label::new(expr_span.end()).with_message("here"))
			.with_message("you may need to add a semicolon at the end of the expression")
	}

	pub(crate) fn incorrect_item_in_trait(item_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, item_span)
			.with_message("invalid item in trait definition".to_string())
			.with_label(Label::new(item_span).with_message("found an item that was unexpected"))
			.with_help("only type definitions and functions are allowed")
	}
}

pub mod ty {
	use ariadne::Color;

	use super::*;
	use crate::{
		resolve::Namespace,
		ty::{Infer, InferKind, TyKind},
	};

	pub fn report_unconstrained(ty_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, ty_span)
			.with_message("expression's type is unconstrained, need type annotations")
			.with_label(Label::new(ty_span))
	}

	pub fn function_cannot_infer_signature(io_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, io_span)
			.with_message("function cannot infer its signature")
			.with_label(Label::new(io_span).with_message("specify a concrete type"))
	}

	pub fn type_unknown(path_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, path_span)
			.with_message("type is invalid")
			.with_label(Label::new(path_span).with_message("type is not in scope"))
	}

	pub fn variable_not_in_scope(ident_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, ident_span)
			.with_message("variable is not in scope")
			.with_label(Label::new(ident_span).with_message("unknown variable"))
	}

	pub fn function_nb_args_mismatch(
		call_span: Span,
		expected_nb: usize,
		actual_nb: usize,
		// def_span: Span,
	) -> ReportBuilder {
		Report::build(ReportKind::Error, call_span)
			.with_message("wrong number of arguments to this function")
			.with_label(Label::new(call_span).with_message(format!(
				"expect {expected_nb} arguments but got {actual_nb}"
			)))
		// TODO: show definition of the original function
		// .with_label(Label::new(def_span).with_message("here is the original definition"))
	}

	pub fn tried_to_call_non_function(
		expr_span: Span,
		call_span: Span,
		actual_ty: &TyKind<Infer>,
	) -> ReportBuilder {
		Report::build(ReportKind::Error, expr_span)
			.with_message("tried to call an expression that is not a function")
			.with_label(Label::new(expr_span).with_message(format!(
				"this is expected to be a function, but is {actual_ty}"
			)))
			.with_label(Label::new(call_span).with_message("this is the call"))
	}

	pub fn unification_mismatch(expected: &TyKind<Infer>, actual: &TyKind<Infer>) -> ReportBuilder {
		todo!("ty mismatch `{expected:?}` vs. `{actual:?}`");
	}

	pub fn infer_kind_unification_mismatch(
		infer: InferKind,
		infer_span: Span,
		actual_infer: InferKind,
		actual_infer_span: Span,
	) -> ReportBuilder {
		Report::build(ReportKind::Error, infer_span)
			.with_message("mismatched types")
			.with_label(
				Label::new(actual_infer_span)
					.with_message(format!("expected {infer}, found {actual_infer}"))
					.with_color(Color::Red),
			)
			.with_label(
				Label::new(infer_span)
					.with_message("expected because of this expression")
					.with_color(Color::Blue),
			)
	}

	pub fn infer_ty_unification_mismatch(
		infer: InferKind,
		infer_span: Span,
		ty: &TyKind<Infer>,
		ty_span: Span,
	) -> ReportBuilder {
		Report::build(ReportKind::Error, infer_span)
			.with_message("mismatched types")
			.with_label(Label::new(infer_span).with_message(format!("expected {infer}")))
			.with_label(Label::new(ty_span).with_message(format!("found {ty}")))
	}

	pub fn item_name_conflict(
		original: Span,
		conflicted: Span,
		namespace: &Namespace,
	) -> ReportBuilder {
		Report::build(ReportKind::Error, original)
			.with_message(format!(
				"distinct {namespace} items have a conflicting name"
			))
			.with_label(Label::new(original).with_message("this is the first item encountered"))
			.with_label(Label::new(conflicted).with_message("this item has the same name"))
	}
}
