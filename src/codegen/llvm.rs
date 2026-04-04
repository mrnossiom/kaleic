use std::{fmt::Write as _, ops::Deref, path::Path, rc::Rc};

use inkwell::{
	AddressSpace, IntPredicate, OptimizationLevel,
	basic_block::BasicBlock,
	builder::Builder,
	context::Context,
	execution_engine::ExecutionEngine,
	module::Module,
	passes::PassBuilderOptions,
	targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine},
	types::{BasicType, BasicTypeEnum, FunctionType},
	values::{AnyValue, BasicValue, BasicValueEnum, FunctionValue, PointerValue},
};
use rustc_hash::FxHashMap;

use crate::{
	ast, bug,
	codegen::{Backend, CodeGenBackend, JitBackend, ObjectBackend},
	collect::DefId,
	hir::{self, Enum, ExprId, Function, Struct},
	session::{ArtefactKind, ScxHandle, SessionCtx},
	symbols::Symbol,
	ty::{self, LateTy, TyCtx, TyKind},
};

type Result<T> = std::result::Result<T, &'static str>;

#[must_use]
#[derive(Debug, Clone)]
enum MaybeValue<'ctx> {
	Value(BasicValueEnum<'ctx>),
	Never,
}

impl MaybeValue<'_> {
	const fn is_never(&self) -> bool {
		matches!(self, Self::Never)
	}
}

pub struct Generator<'tcx, 'ctx> {
	tcx: &'tcx TyCtx<'tcx>,

	ctx: &'ctx Context,
	builder: Builder<'ctx>,
	module: Module<'ctx>,
	jit: ExecutionEngine<'ctx>,

	function_ids: FxHashMap<DefId, FunctionValue<'ctx>>,

	empty_ty: BasicTypeEnum<'ctx>,
}

impl<'tcx> Generator<'tcx, '_> {
	pub(crate) fn new_jit(tcx: &'tcx TyCtx<'_>) -> Self {
		Self::new(tcx)
	}

	pub(crate) fn new_object(tcx: &'tcx TyCtx<'_>) -> Self {
		Self::new(tcx)
	}
}

impl<'tcx, 'ctx> Generator<'tcx, 'ctx> {
	pub(crate) fn new(tcx: &'tcx TyCtx) -> Self {
		// TODO: do not leak
		let ctx = Box::leak(Box::new(Context::create()));

		let module = ctx.create_module("repl");

		// TODO: mode to `new_jit` function
		let opt_level = if tcx.scx().options.opt {
			OptimizationLevel::Default
		} else {
			OptimizationLevel::None
		};
		let jit = module.create_jit_execution_engine(opt_level).unwrap();

		let empty_ty = ctx.struct_type(&[], false).as_basic_type_enum();

		Self {
			tcx,

			ctx,
			builder: ctx.create_builder(),
			module,
			jit,

			function_ids: FxHashMap::default(),

			empty_ty,
		}
	}

	fn to_llvm_type(&self, ty: &LateTy) -> Option<BasicTypeEnum<'ctx>> {
		match ty.clone() {
			ty::TyKind::Primitive(kind) => match kind {
				ty::PrimitiveKind::Unit => Some(self.empty_ty),
				ty::PrimitiveKind::Never => None,
				ty::PrimitiveKind::Bool => Some(self.ctx.bool_type().into()),
				ty::PrimitiveKind::UnsignedInt | ty::PrimitiveKind::SignedInt => {
					Some(self.ctx.i32_type().into())
				}
				ty::PrimitiveKind::Float => Some(self.ctx.f32_type().into()),
				ty::PrimitiveKind::Str => todo!(),
			},
			ty::TyKind::Pointer(..) | ty::TyKind::Fn(..) => {
				Some(self.ctx.ptr_type(AddressSpace::default()).into())
			}
			ty::TyKind::Struct(enum_) => todo!(),
			ty::TyKind::Enum(struct_) => todo!(),
			ty::TyKind::Error => {
				bug!("error type kind is a placeholder and should not reach codegen")
			}
		}
	}

	fn define_func(
		&self,
		func_val: FunctionValue<'ctx>,
		def_id: DefId,
		decl: &ty::FnDecl<LateTy>,
		body: &hir::Block,
	) -> Result<()> {
		let type_env = &self.tcx.type_env.borrow();
		let typeck_results = &self.tcx.typeck_results.borrow_key(&def_id);

		let mut generator = FunctionGenerator {
			scx: self.tcx.scx(),

			typeck_results,
			function_ids: &self.function_ids,

			ctx: self.ctx,
			module: &self.module,
			builder: &self.builder,
			function: func_val,

			variables: FxHashMap::default(),
			loop_stack: Vec::default(),

			empty_ty: self.empty_ty,
		};

		generator.codegen_body(decl, body)?;

		self.tcx.scx().register_artefact(
			&ArtefactKind::BackendIr(def_id, Backend::Llvm),
			|artefact| write!(artefact, "{}", func_val.print_to_string().to_string_lossy()),
		);

		if !func_val.verify(true) {
			let name = func_val.get_name().to_string_lossy().into_owned();
			bug!("function `{name}` is invalid")

			// TODO SAFETY: do not keep any reference to this function, e.g. the fn map
			// unsafe { func_val.delete() }
		}

		Ok(())
	}
}

impl CodeGenBackend for Generator<'_, '_> {
	fn codegen_root(&mut self, hir: &hir::Root) {
		let type_env = self.tcx.type_env.borrow();

		for item in &hir.items {
			match &item.kind {
				hir::ItemKind::Function(Function { name, decl, body }) => {
					let type_env = self.tcx.type_env.borrow();
					let TyKind::Fn(decl) = &*type_env[&item.def_id] else {
						todo!()
					};

					let func_id = self.declare_func(name.sym, decl).unwrap();
					self.function_ids.insert(item.def_id, func_id);
				}
				hir::ItemKind::ForeignMod { items } => {
					for item in items {
						match &item.kind {
							hir::ForeignItemKind::Function(Function { name, decl, body }) => {
								let TyKind::Fn(decl) = &*type_env[&item.def_id] else {
									todo!()
								};

								let func_id = self.declare_func(name.sym, decl).unwrap();
								self.function_ids.insert(item.def_id, func_id);
							}
						}
					}
				}

				hir::ItemKind::Struct(Struct { .. }) | hir::ItemKind::Enum(Enum { .. }) => {
					// TODO: codegen constructors here?
				}
				hir::ItemKind::TypeAlias(_) | hir::ItemKind::Trait { .. } => {}
				hir::ItemKind::TraitImpl { .. } => {
					// TODO: codegen methods
				}
			}
		}
		for item in &hir.items {
			match &item.kind {
				hir::ItemKind::Function(Function { name, decl, body }) => {
					let TyKind::Fn(decl) = &*type_env[&item.def_id] else {
						todo!()
					};

					let Some(func_id) = self.function_ids.get(&item.def_id) else {
						println!("assuming fn `{:#?}` is external", name.sym);
						continue;
					};

					let body = body.as_ref().unwrap();
					self.define_func(*func_id, item.def_id, decl, body).unwrap();
				}

				hir::ItemKind::TraitImpl { .. } => {
					// TODO
				}

				hir::ItemKind::Struct(Struct { .. }) | hir::ItemKind::Enum(Enum { .. }) => {
					// TODO
				}

				hir::ItemKind::ForeignMod { .. }
				| hir::ItemKind::TypeAlias(_)
				| hir::ItemKind::Trait { .. } => {}
			}
		}
	}
}

impl JitBackend for Generator<'_, '_> {
	fn finalize(&mut self) {}

	fn call_main(&self) {
		#[expect(unsafe_code)]
		let ret = unsafe { self.jit.get_function::<unsafe extern "C" fn()>("main") }.unwrap();

		#[expect(unsafe_code)]
		unsafe {
			ret.call();
		}
	}
}

impl ObjectBackend for Generator<'_, '_> {
	fn write_object(self: Box<Self>, path: &Path) {
		Target::initialize_all(&InitializationConfig::default());

		let target_triple = TargetMachine::get_default_triple();
		let target = Target::from_triple(&target_triple).unwrap();
		let target_machine = target
			.create_target_machine(
				&target_triple,
				"generic",
				"",
				if self.tcx.scx().options.opt {
					OptimizationLevel::Default
				} else {
					OptimizationLevel::None
				},
				RelocMode::PIC,
				CodeModel::Default,
			)
			.unwrap();

		let passes: &[&str] = &[
			"mem2reg",
			// "reassociate",
			// "instcombine",
			// "gvn",
			// "simplifycfg",
			// "basic-aa",
		];

		self.module
			.run_passes(
				&passes.join(","),
				&target_machine,
				PassBuilderOptions::create(),
			)
			.unwrap();

		target_machine
			.write_to_file(&self.module, FileType::Object, path)
			.unwrap();
	}
}

impl<'ctx> Generator<'_, 'ctx> {
	pub(crate) fn lower_signature(&self, decl: &ty::FnDecl<LateTy>) -> FunctionType<'ctx> {
		let mut args_ty = Vec::new();
		for ty::Param { name: _, ty, id } in &decl.inputs {
			let type_ = self.to_llvm_type(ty).unwrap();
			args_ty.push(type_.into());
		}

		if let Some(ret_ty) = self.to_llvm_type(&decl.output) {
			ret_ty.fn_type(&args_ty, false)
		} else {
			self.ctx.void_type().fn_type(&args_ty, false)
		}
	}

	pub(crate) fn declare_func(
		&self,
		name: Symbol,
		decl: &ty::FnDecl<LateTy>,
	) -> Result<FunctionValue<'ctx>> {
		let fn_ty = self.lower_signature(decl);

		let fn_val = self
			.module
			.add_function(&self.tcx.scx().symbols.resolve(name), fn_ty, None);

		// set arguments name
		fn_val
			.get_param_iter()
			.zip(&decl.inputs)
			.for_each(|(arg, ty::Param { name, ty, id })| {
				arg.into_int_value()
					.set_name(&self.tcx.scx().symbols.resolve(name.sym));
			});

		Ok(fn_val)
	}
}

struct FunctionGenerator<'scx, 'bld, 'ctx> {
	scx: &'scx SessionCtx,

	typeck_results: &'scx FxHashMap<ExprId, Rc<LateTy>>,
	function_ids: &'bld FxHashMap<DefId, FunctionValue<'ctx>>,

	ctx: &'ctx Context,
	module: &'bld Module<'ctx>,
	builder: &'bld Builder<'ctx>,
	function: FunctionValue<'ctx>,

	variables: FxHashMap<hir::NodeId, PointerValue<'ctx>>,
	// stack of loop and continuation blocks
	// TODO: support labels
	loop_stack: Vec<(
		BasicBlock<'ctx>,
		BasicBlock<'ctx>,
		Option<PointerValue<'ctx>>,
	)>,

	// TODO: move to a predefined types struct
	empty_ty: BasicTypeEnum<'ctx>,
}

impl<'ctx> FunctionGenerator<'_, '_, 'ctx> {
	fn empty_value(&self) -> MaybeValue<'ctx> {
		let empty = self.ctx.const_struct(&[], false);
		MaybeValue::Value(empty.into())
	}

	// TODO: remove duplicate
	fn to_llvm_type(&self, ty: &LateTy) -> Option<BasicTypeEnum<'ctx>> {
		match ty.clone() {
			ty::TyKind::Primitive(kind) => match kind {
				ty::PrimitiveKind::Unit => Some(self.empty_ty),
				ty::PrimitiveKind::Never => None,
				ty::PrimitiveKind::Bool => Some(self.ctx.i8_type().into()),
				ty::PrimitiveKind::UnsignedInt | ty::PrimitiveKind::SignedInt => {
					Some(self.ctx.i32_type().into())
				}
				ty::PrimitiveKind::Float => Some(self.ctx.f32_type().into()),
				ty::PrimitiveKind::Str => todo!(),
			},
			ty::TyKind::Pointer(_kind) => todo!(),
			ty::TyKind::Fn(_fn_decl) => Some(self.ctx.ptr_type(AddressSpace::default()).into()),
			ty::TyKind::Struct(enum_) => todo!(),
			ty::TyKind::Enum(struct_) => todo!(),
			ty::TyKind::Error => {
				bug!("error type kind is a placeholder and should not reach codegen")
			}
		}
	}

	fn codegen_body(&mut self, decl: &ty::FnDecl<LateTy>, block: &hir::Block) -> Result<()> {
		let bb = self.ctx.append_basic_block(self.function, "entry");
		self.builder.position_at_end(bb);

		for (ty::Param { name, ty, id }, value) in
			decl.inputs.iter().zip(self.function.get_param_iter())
		{
			let Some(ty) = self.to_llvm_type(ty) else {
				return Ok(());
			};
			let place = self
				.builder
				.build_alloca(ty, &format!("{:#?}", name.sym))
				.unwrap();

			self.builder.build_store(place, value).unwrap();
			self.variables.insert(*id, place);
		}

		match self.codegen_block(block)? {
			MaybeValue::Value(ret_val) => _ = self.builder.build_return(Some(&ret_val)).unwrap(),
			MaybeValue::Never => _ = self.builder.build_unreachable().unwrap(),
		}
		Ok(())
	}

	fn codegen_block(&mut self, block: &hir::Block) -> Result<MaybeValue<'ctx>> {
		for stmt in &block.stmts {
			let should_stop_block_codegen = self.codegen_stmt(stmt)?;
			if should_stop_block_codegen {
				return Ok(MaybeValue::Never);
			}
		}

		if let Some(expr) = &block.ret {
			self.codegen_expr(expr)
		} else {
			Ok(self.empty_value())
		}
	}

	fn codegen_stmt(&mut self, stmt: &hir::Stmt) -> Result<bool /* should_stop_block_codegen */> {
		match &stmt.kind {
			hir::StmtKind::Expr { expr } => Ok(self.codegen_expr(expr)?.is_never()),
			hir::StmtKind::Let {
				name,
				value,
				ty: _,
				mutable: _,
			} => {
				let ty = &self.typeck_results[&value.expr_id()];
				let ty = self.to_llvm_type(ty).unwrap();
				let place = self
					.builder
					.build_alloca(ty, &format!("{:#?}", name.sym))
					.unwrap();

				let expr_value = self.codegen_expr(value)?;
				match expr_value {
					MaybeValue::Value(value) => {
						self.builder.build_store(place, value).unwrap();
					}
					MaybeValue::Never => {}
				}

				self.variables.insert(stmt.id, place);
				Ok(expr_value.is_never())
			}
		}
	}

	fn codegen_expr(&mut self, expr: &hir::Expr) -> Result<MaybeValue<'ctx>> {
		let expr_ty = &self.typeck_results[&expr.expr_id()];

		let value = match &expr.kind {
			hir::ExprKind::LiteralStr { sym } => {
				let sym = self.scx.symbols.resolve(*sym);

				let ty = &self.typeck_results[&expr.expr_id()];
				let ty = self.to_llvm_type(ty).unwrap();

				let val = todo!();
				MaybeValue::Value(val)
			}
			hir::ExprKind::LiteralInt { sym } => {
				let number = {
					let lit = self.scx.symbols.resolve(*sym);
					lit.parse::<u64>().unwrap()
				};

				let ty = &self.typeck_results[&expr.expr_id()];
				let ty = self.to_llvm_type(ty).unwrap();

				let value = ty
					.into_int_type()
					.const_int(number, true)
					.as_basic_value_enum();

				MaybeValue::Value(value)
			}
			hir::ExprKind::LiteralFloat { sym } => {
				let number = {
					let lit = self.scx.symbols.resolve(*sym);
					lit.parse::<f64>().unwrap()
				};

				let ty = &self.typeck_results[&expr.expr_id()];
				let ty = self.to_llvm_type(ty).unwrap();

				let value = ty
					.into_float_type()
					.const_float(number)
					.as_basic_value_enum();

				MaybeValue::Value(value)
			}
			hir::ExprKind::Access { path } => {
				let hir_id = path.resolved.into_local().unwrap();
				let place = *self.variables.get(&hir_id).unwrap();

				let ty = &self.typeck_results[&expr.expr_id()];
				let ty = self.to_llvm_type(ty).unwrap();

				let value = self
					.builder
					.build_load(ty, place, "")
					.unwrap()
					.as_basic_value_enum();

				MaybeValue::Value(value)
			}
			hir::ExprKind::Assign { target, value } => {
				let hir::ExprKind::Access { path } = &target.kind else {
					todo!("invalid lvalue");
				};

				let hir_id = path.resolved.into_local().unwrap();
				let place = *self.variables.get(&hir_id).unwrap();
				let expr_value = self.codegen_expr(value)?;
				match expr_value {
					MaybeValue::Value(value) => {
						self.builder.build_store(place, value).unwrap();
					}
					MaybeValue::Never => {}
				}

				expr_value
			}
			hir::ExprKind::Binary { op, left, right } => self.codegen_bin_op(*op, left, right)?,

			hir::ExprKind::Unary { op, expr } => todo!(),
			hir::ExprKind::Method { expr, name, params } => todo!(),
			hir::ExprKind::Field { expr, name } => todo!(),
			hir::ExprKind::Deref { expr } => todo!(),

			hir::ExprKind::FnCall { expr, args } => {
				let mut argsz = Vec::new();
				for arg in &args.bit {
					// TODO
					match self.codegen_expr(arg)? {
						MaybeValue::Value(value) => argsz.push(value.into()),
						MaybeValue::Never => {}
					}
				}

				let call = if let hir::ExprKind::Access { path } = &expr.kind {
					// direct call
					let def_id = path.resolved.into_def().unwrap();
					let func = self.function_ids[&def_id];

					if args.bit.len() != func.count_params() as usize {
						return Err("fn call args count mismatch");
					}

					self.builder.build_call(func, &argsz, "").unwrap()
				} else {
					// indirect call
					let addr = match self.codegen_expr(expr)? {
						MaybeValue::Value(val) => val.into_pointer_value(),
						MaybeValue::Never => return Ok(MaybeValue::Never),
					};

					let fn_type = todo!();
					self.builder
						.build_indirect_call(fn_type, addr, &argsz, "")
						.unwrap()
				};

				match call.try_as_basic_value().basic() {
					Some(val) => MaybeValue::Value(val),
					None => {
						if matches!(&**expr_ty, TyKind::Primitive(ty::PrimitiveKind::Never)) {
							MaybeValue::Never
						} else {
							self.empty_value()
						}
					}
				}
			}

			hir::ExprKind::If {
				cond,
				conseq,
				altern,
			} => self.codegen_if(cond, conseq, altern.as_deref())?,
			hir::ExprKind::Loop { block } => {
				let loop_ = self.ctx.append_basic_block(self.function, "loop");
				let cont = self.ctx.append_basic_block(self.function, "cont");

				let block_ty = &self.typeck_results[&expr.expr_id()];
				let cont_ptr = if let Some(block_ty) = self.to_llvm_type(block_ty) {
					Some(self.builder.build_alloca(block_ty, "return").unwrap())
				} else {
					None
				};

				self.loop_stack.push((loop_, cont, cont_ptr));

				self.builder.build_unconditional_branch(loop_).unwrap();

				self.builder.position_at_end(loop_);

				match self.codegen_block(block)? {
					MaybeValue::Value(value) => {
						_ = self.builder.build_unconditional_branch(loop_).unwrap();
					}
					MaybeValue::Never => {}
				}

				self.builder.position_at_end(cont);

				self.loop_stack.pop();

				if let Some((block_ty, ptr)) = self.to_llvm_type(block_ty).zip(cont_ptr) {
					MaybeValue::Value(self.builder.build_load(block_ty, ptr, "").unwrap())
				} else {
					self.empty_value()
				}
			}

			hir::ExprKind::Unit => self.empty_value(),

			hir::ExprKind::Return { expr } => {
				match self.codegen_expr(expr)? {
					MaybeValue::Value(value) => {
						self.builder.build_return(Some(&value)).unwrap();
					}
					MaybeValue::Never => {
						self.builder.build_return(None).unwrap();
					}
				}

				MaybeValue::Never
			}
			hir::ExprKind::Break { expr, label } => {
				let (_loop_, cont, cont_ptr) = *self.loop_stack.last().unwrap();

				match self.codegen_expr(expr)? {
					MaybeValue::Value(value) => {
						_ = self.builder.build_store(cont_ptr.unwrap(), value).unwrap();
					}
					MaybeValue::Never => {}
				}

				self.builder.build_unconditional_branch(cont).unwrap();
				MaybeValue::Never
			}
			hir::ExprKind::Continue { label } => {
				let (loop_, _cont, _cont_ptr) = *self.loop_stack.last().unwrap();
				self.builder.build_unconditional_branch(loop_).unwrap();
				MaybeValue::Never
			}
		};
		Ok(value)
	}

	fn codegen_if(
		&mut self,
		cond: &hir::Expr,
		conseq: &hir::Block,
		altern: Option<&hir::Block>,
	) -> Result<MaybeValue<'ctx>> {
		let condition = match self.codegen_expr(cond)? {
			MaybeValue::Value(val) => val.into_int_value(),
			MaybeValue::Never => return Ok(MaybeValue::Never),
		};

		let then_bb = self.ctx.append_basic_block(self.function, "then");
		let else_bb = altern
			.as_ref()
			.map(|_| self.ctx.append_basic_block(self.function, "else"));
		let cont_bb = self.ctx.append_basic_block(self.function, "merge");

		// TODO: so ugly
		let ty = conseq
			.ret
			.as_ref()
			.map_or(TyKind::Primitive(ty::PrimitiveKind::Unit), |ret| {
				self.typeck_results
					.get(&ret.expr_id())
					.unwrap()
					.deref()
					.clone()
			});
		let ty = self.to_llvm_type(&ty);

		let ret_ptr = ty
			.as_ref()
			.map(|ty| self.builder.build_alloca(*ty, "").unwrap());

		self.builder
			.build_conditional_branch(condition, then_bb, else_bb.unwrap_or(cont_bb))
			.unwrap();
		self.builder.position_at_end(then_bb);

		match self.codegen_block(conseq)? {
			MaybeValue::Value(value) => {
				self.builder.build_store(ret_ptr.unwrap(), value).unwrap();
				self.builder.build_unconditional_branch(cont_bb).unwrap();
			}
			MaybeValue::Never => {}
		}

		if let Some(altern) = altern {
			let else_bb = else_bb.unwrap();

			self.builder.position_at_end(else_bb);

			match self.codegen_block(altern)? {
				MaybeValue::Value(value) => {
					self.builder.build_store(ret_ptr.unwrap(), value).unwrap();
					self.builder.build_unconditional_branch(cont_bb).unwrap();
				}
				MaybeValue::Never => {}
			}
		}

		self.builder.position_at_end(cont_bb);

		let value = if let Some(ty) = ty {
			let value = self.builder.build_load(ty, ret_ptr.unwrap(), "").unwrap();
			MaybeValue::Value(value)
		} else {
			self.empty_value()
		};

		Ok(value)
	}

	fn codegen_bin_op(
		&mut self,
		op: ast::Spanned<ast::BinaryOp>,
		left: &hir::Expr,
		right: &hir::Expr,
	) -> Result<MaybeValue<'ctx>> {
		let lhs = self.codegen_expr(left)?;
		let rhs = self.codegen_expr(right)?;

		// cannot be zst
		let (lhs, rhs) = match (lhs, rhs) {
			(MaybeValue::Value(lhs), MaybeValue::Value(rhs)) => (lhs, rhs),
			(MaybeValue::Never, _) | (_, MaybeValue::Never) => return Ok(MaybeValue::Never),
		};

		let lhs = lhs.into_int_value();
		let rhs = rhs.into_int_value();

		let ins = match op.bit {
			ast::BinaryOp::Plus => self.builder.build_int_add(lhs, rhs, "").unwrap(),
			ast::BinaryOp::Minus => self.builder.build_int_sub(lhs, rhs, "").unwrap(),
			ast::BinaryOp::Mul => self.builder.build_int_mul(lhs, rhs, "").unwrap(),
			ast::BinaryOp::Div => self.builder.build_int_unsigned_div(lhs, rhs, "").unwrap(),
			ast::BinaryOp::Mod => self.builder.build_int_unsigned_rem(lhs, rhs, "").unwrap(),

			ast::BinaryOp::And => self.builder.build_and(lhs, rhs, "").unwrap(),
			ast::BinaryOp::Or => self.builder.build_or(lhs, rhs, "").unwrap(),
			ast::BinaryOp::Xor => self.builder.build_xor(lhs, rhs, "").unwrap(),
			ast::BinaryOp::Shl => self.builder.build_left_shift(lhs, rhs, "").unwrap(),
			ast::BinaryOp::Shr => self.builder.build_right_shift(lhs, rhs, false, "").unwrap(),

			ast::BinaryOp::Gt => self
				.builder
				.build_int_compare(IntPredicate::SGT, lhs, rhs, "")
				.unwrap(),
			ast::BinaryOp::Ge => self
				.builder
				.build_int_compare(IntPredicate::SGE, lhs, rhs, "")
				.unwrap(),
			ast::BinaryOp::Lt => self
				.builder
				.build_int_compare(IntPredicate::SLT, lhs, rhs, "")
				.unwrap(),
			ast::BinaryOp::Le => self
				.builder
				.build_int_compare(IntPredicate::SLE, lhs, rhs, "")
				.unwrap(),
			ast::BinaryOp::EqEq => self
				.builder
				.build_int_compare(IntPredicate::EQ, lhs, rhs, "")
				.unwrap(),
			ast::BinaryOp::Ne => self
				.builder
				.build_int_compare(IntPredicate::NE, lhs, rhs, "")
				.unwrap(),
		};

		Ok(MaybeValue::Value(ins.as_basic_value_enum()))
	}
}
