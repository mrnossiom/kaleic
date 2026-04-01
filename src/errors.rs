use ariadne::{Label, ReportKind};

use crate::session::{Report, ReportBuilder, Span};

pub(crate) mod parser {
	use std::path::Path;

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

pub(crate) mod resolve {
	use super::*;

	pub fn type_not_in_scope(path_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, path_span)
			.with_message("type is invalid")
			.with_label(Label::new(path_span).with_message("type is not in scope"))
	}

	pub fn value_not_in_scope(path_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, path_span)
			.with_message("value is invalid")
			.with_label(Label::new(path_span).with_message("value is not in scope"))
	}

	pub fn invalid_lang_item(lang_item_span: Span) -> ariadne::ReportBuilder<Span, ReportKind> {
		Report::build(ReportKind::Error, lang_item_span)
			.with_message("language item does not exist")
			.with_label(Label::new(lang_item_span).with_message("here"))
	}
}

pub(crate) mod lowerer {
	use super::*;

	pub fn no_semicolon_mid_block(expr_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, expr_span)
			.with_message("expression is missing a semicolon but is not at the end")
			.with_label(Label::new(expr_span.end()).with_message("here"))
			.with_message("you may need to add a semicolon at the end of the expression")
	}

	pub fn incorrect_item_in_trait(item_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, item_span)
			.with_message("invalid item in trait definition".to_string())
			.with_label(Label::new(item_span).with_message("found an item that was unexpected"))
			.with_help("only type definitions and functions are allowed")
	}

	pub fn generic_in_attr_path(generics: Span) -> ariadne::ReportBuilder<Span, ReportKind> {
		Report::build(ReportKind::Error, generics)
			.with_message("attribute paths cannot contain generics".to_string())
			.with_label(Label::new(generics).with_message("remove these generics"))
	}
}

pub(crate) mod ty {
	use ariadne::Color;

	use super::*;
	use crate::{
		resolve::Namespace,
		ty::{InferExprTy, InferKind},
	};

	pub fn report_unconstrained(ty_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, ty_span)
			.with_message("expression's type is unconstrained, need type annotations")
			.with_label(Label::new(ty_span).with_message("here"))
	}

	pub fn function_cannot_infer_signature(io_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, io_span)
			.with_message("function cannot infer its signature")
			.with_label(Label::new(io_span).with_message("specify a concrete type"))
	}

	pub fn type_alias_empty(item_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, item_span)
			.with_message("type alias have to be defined outside trait definitions")
			.with_label(Label::new(item_span).with_message("define this type alias"))
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
		actual_ty: &InferExprTy,
	) -> ReportBuilder {
		Report::build(ReportKind::Error, expr_span)
			.with_message("tried to call an expression that is not a function")
			.with_label(Label::new(expr_span).with_message(format!(
				"this is expected to be a function, but is {actual_ty}"
			)))
			.with_label(Label::new(call_span).with_message("this is the call"))
	}

	pub fn unification_mismatch(expected: &InferExprTy, actual: &InferExprTy) -> ReportBuilder {
		todo!("ty mismatch `{expected}` vs. `{actual}`");
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
		ty: &InferExprTy,
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
		namespace: Namespace,
	) -> ReportBuilder {
		Report::build(ReportKind::Error, original)
			.with_message(format!(
				"distinct {namespace} items have a conflicting name"
			))
			.with_label(Label::new(original).with_message("this is the first item encountered"))
			.with_label(Label::new(conflicted).with_message("this item has the same name"))
	}

	pub fn no_main_function() -> ReportBuilder {
		Report::build(ReportKind::Error, Span::DUMMY).with_message("no main function")
	}

	pub fn main_function_wrong_signature(fn_span: Span) -> ReportBuilder {
		Report::build(ReportKind::Error, fn_span)
			.with_message("main function doesn't match the expected signature")
			.with_label(Label::new(fn_span).with_message("here"))
	}
}
