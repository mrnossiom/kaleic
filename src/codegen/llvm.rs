use std::collections::HashMap;

use inkwell::{
	AddressSpace, IntPredicate, OptimizationLevel,
	basic_block::BasicBlock,
	builder::Builder,
	context::Context,
	execution_engine::ExecutionEngine,
	module::Module,
	passes::PassBuilderOptions,
	targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine},
	types::{BasicType, BasicTypeEnum, FunctionType},
	values::{BasicValue, BasicValueEnum, FunctionValue, PointerValue},
};

use crate::{
	ast, bug,
	codegen::{CodeGenBackend, JitBackend, ObjectBackend},
	hir::{self, Function},
	lexer,
	session::{PrintKind, SessionCtx, Symbol},
	ty::{self, TyCtx, TyKind},
};

type Result<T> = std::result::Result<T, &'static str>;

#[derive(Debug, Clone)]
enum MaybeValue<'ctx> {
	Value(BasicValueEnum<'ctx>),
	Zst,
	Never,
}

impl<'ctx> MaybeValue<'ctx> {
	const fn is_never(&self) -> bool {
		matches!(self, Self::Never)
	}

	fn as_value(&self) -> Option<Option<&dyn BasicValue<'ctx>>> {
		match self {
			Self::Value(val) => Some(Some(val)),
			Self::Zst => Some(None),
			Self::Never => None,
		}
	}

	fn as_value_enum(&self) -> Option<Option<BasicValueEnum<'ctx>>> {
		match self {
			Self::Value(val) => Some(Some(*val)),
			Self::Zst => Some(None),
			Self::Never => None,
		}
	}
}

pub struct Generator<'tcx, 'ctx> {
	tcx: &'tcx TyCtx<'tcx>,

	ctx: &'ctx Context,
	builder: Builder<'ctx>,
	module: Module<'ctx>,
	jit: ExecutionEngine<'ctx>,
}

impl<'tcx> Generator<'tcx, '_> {
	pub fn new_jit(tcx: &'tcx TyCtx) -> Self {
		// TODO
		let context = Box::leak(Box::new(inkwell::context::Context::create()));

		Self::new(tcx, context)
	}
}

impl<'tcx, 'ctx> Generator<'tcx, 'ctx> {
	pub fn new(tcx: &'tcx TyCtx, ctx: &'ctx Context) -> Self {
		let module = ctx.create_module("repl");
		let jit = module
			.create_jit_execution_engine(OptimizationLevel::None)
			.unwrap();

		Self {
			tcx,

			ctx,
			builder: ctx.create_builder(),
			module,
			jit,
		}
	}

	fn to_llvm_type(&self, output: &ty::TyKind) -> Option<BasicTypeEnum<'ctx>> {
		match output.clone() {
			ty::TyKind::Primitive(kind) => match kind {
				ty::PrimitiveKind::Void | ty::PrimitiveKind::Never => None,
				// (self.ctx.void_type().into()),
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
			ty::TyKind::Ref(ref_) => {
				self.to_llvm_type(&self.tcx.ty_env.borrow().as_ref().unwrap()[&ref_])
			}
			ty::TyKind::Error => {
				bug!("error type kind is a placeholder and should not reach codegen")
			}
		}
	}

	fn define_func(
		&mut self,
		func_val: FunctionValue<'ctx>,
		decl: &ty::FnDecl,
		body: &hir::Block,
	) -> Result<()> {
		let borrow = self.tcx.ty_env.borrow();
		let ty_env = borrow.as_ref().unwrap();
		let borrow = self.tcx.typeck_results.borrow();
		let typeck_results = borrow.as_ref().unwrap();

		let empty_ty = self.ctx.struct_type(&[], false).as_basic_type_enum();

		let mut generator = FunctionGenerator {
			scx: self.tcx.scx,

			ty_env,
			typeck_results,

			ctx: &*self.ctx,
			module: &self.module,
			builder: &self.builder,
			function: func_val,

			variables: HashMap::new(),
			loop_stack: Vec::new(),

			empty_ty,
		};

		generator.codegen_body(decl, body)?;

		if self.tcx.scx.options.print.contains(&PrintKind::BackendIr) {
			func_val.print_to_stderr();
		}

		if !func_val.verify(true) {
			#[allow(unsafe_code)]
			unsafe {
				func_val.delete();
			}
			return Err("function is invalid");
		}

		Ok(())
	}
}

impl<'ctx> CodeGenBackend for Generator<'_, '_> {
	fn codegen_root(&mut self, hir: &hir::Root) {
		let mut function_ids = HashMap::new();

		for item in &hir.items {
			match &item.kind {
				hir::ItemKind::Function(Function {
					name,
					decl,
					body,
					abi,
				}) => {
					let borrow = self.tcx.ty_env.borrow();
					let TyKind::Fn(decl) = borrow.as_ref().unwrap().get(&item.id).unwrap() else {
						todo!()
					};

					match abi {
						hir::Abi::Kalei => {
							let func_id = self.declare_func(name.sym, decl).unwrap();
							function_ids.insert(item.id, func_id);
						}
						hir::Abi::C => {
							let _func_id = self.declare_func(name.sym, &decl).unwrap();
						}
					}
				}
				_ => {}
			}
		}
		for item in &hir.items {
			match &item.kind {
				hir::ItemKind::Function(Function {
					name,
					decl,
					body,
					abi,
				}) => {
					let borrow = self.tcx.ty_env.borrow();
					let TyKind::Fn(decl) = borrow.as_ref().unwrap().get(&item.id).unwrap() else {
						todo!()
					};

					let Some(func_id) = function_ids.get(&item.id) else {
						println!("assuming fn {:#?} is external", name.sym);
						continue;
					};

					let body = body.as_ref().unwrap();
					self.define_func(*func_id, &decl, &body).unwrap();
				}
				_ => {}
			}
		}
	}
}

impl JitBackend for Generator<'_, '_> {
	fn call_main(&mut self) {
		#[expect(unsafe_code)]
		let ret = unsafe {
			self.jit
				.get_function::<unsafe extern "C" fn() -> i64>("main")
		}
		.unwrap();

		#[expect(unsafe_code)]
		unsafe {
			ret.call()
		};
	}
}

impl ObjectBackend for Generator<'_, '_> {
	fn get_object(self) -> cranelift_object::ObjectProduct {
		todo!()
	}
}

impl<'ctx> Generator<'_, 'ctx> {
	pub fn apply_passes(&self) {
		Target::initialize_all(&InitializationConfig::default());

		let target_triple = TargetMachine::get_default_triple();
		let target = Target::from_triple(&target_triple).unwrap();
		let target_machine = target
			.create_target_machine(
				&target_triple,
				"generic",
				"",
				OptimizationLevel::None,
				RelocMode::PIC,
				CodeModel::Default,
			)
			.unwrap();

		let passes: &[&str] = &[
			"instcombine",
			"reassociate",
			"gvn",
			"simplifycfg",
			// "basic-aa",
			"mem2reg",
		];

		self.module
			.run_passes(
				&passes.join(","),
				&target_machine,
				PassBuilderOptions::create(),
			)
			.unwrap();
	}
}

impl<'ctx> Generator<'_, 'ctx> {
	pub fn lower_signature(&self, decl: &ty::FnDecl) -> FunctionType<'ctx> {
		let mut args_ty = Vec::new();
		for ty::Param { name: _, ty } in &decl.inputs {
			let type_ = self.to_llvm_type(ty).unwrap();
			args_ty.push(type_.into());
		}

		if let Some(ret_ty) = self.to_llvm_type(&decl.output) {
			ret_ty.fn_type(&args_ty, false)
		} else {
			self.ctx.void_type().fn_type(&args_ty, false)
		}
	}

	pub fn declare_func(&mut self, name: Symbol, decl: &ty::FnDecl) -> Result<FunctionValue<'ctx>> {
		let fn_ty = self.lower_signature(decl);

		let name = self.tcx.scx.symbols.resolve(name);
		let fn_val = self.module.add_function(&name, fn_ty, None);

		// set arguments name
		fn_val
			.get_param_iter()
			.zip(&decl.inputs)
			.for_each(|(arg, ty::Param { name, ty })| {
				arg.into_int_value()
					.set_name(&self.tcx.scx.symbols.resolve(name.sym));
			});

		Ok(fn_val)
	}
}

struct FunctionGenerator<'scx, 'bld, 'ctx> {
	scx: &'scx SessionCtx,

	ty_env: &'scx HashMap<hir::NodeId, ty::TyKind>,
	typeck_results: &'scx HashMap<hir::NodeId, ty::TyKind>,

	ctx: &'ctx Context,
	module: &'bld Module<'ctx>,
	builder: &'bld Builder<'ctx>,
	function: FunctionValue<'ctx>,

	variables: HashMap<Symbol, PointerValue<'ctx>>,
	// stack of loop and continuation blocks
	// TODO: support labels
	loop_stack: Vec<(BasicBlock<'ctx>, BasicBlock<'ctx>)>,

	// TODO: move to a predefined types struct
	empty_ty: BasicTypeEnum<'bld>,
}

impl<'ctx> FunctionGenerator<'_, '_, 'ctx> {
	// TODO: remove duplicate
	fn to_llvm_type(&self, output: &ty::TyKind) -> Option<BasicTypeEnum<'ctx>> {
		match output.clone() {
			ty::TyKind::Primitive(kind) => match kind {
				ty::PrimitiveKind::Void | ty::PrimitiveKind::Never => None,
				// (self.ctx.void_type().into()),
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
			ty::TyKind::Ref(ref_) => self.to_llvm_type(&self.ty_env[&ref_]),
			ty::TyKind::Error => {
				bug!("error type kind is a placeholder and should not reach codegen")
			}
		}
	}

	fn codegen_body(&mut self, decl: &ty::FnDecl, block: &hir::Block) -> Result<()> {
		let bb = self.ctx.append_basic_block(self.function, "entry");
		self.builder.position_at_end(bb);

		for (ty::Param { name, ty }, value) in
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
			self.variables.insert(name.sym, place);
		}

		let ret_val = self.codegen_block(block)?;
		if let Some(ret_val) = ret_val.as_value() {
			self.builder.build_return(ret_val).unwrap();
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
			Ok(MaybeValue::Zst)
		}
	}

	fn codegen_stmt(&mut self, stmt: &hir::Stmt) -> Result<bool /* should_stop_block_codegen */> {
		match &stmt.kind {
			hir::StmtKind::Expr(expr) => Ok(self.codegen_expr(expr)?.is_never()),
			hir::StmtKind::Let {
				name,
				value,
				ty: _,
				mutable: _,
			} => {
				let ty = self.typeck_results.get(&value.id).unwrap();
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
					MaybeValue::Zst | MaybeValue::Never => {}
				}

				self.variables.insert(name.sym, place);
				Ok(expr_value.is_never())
			}
			hir::StmtKind::Loop(block) => {
				let loop_ = self.ctx.append_basic_block(self.function, "loop");
				let cont = self.ctx.append_basic_block(self.function, "cont");

				self.loop_stack.push((loop_, cont));

				self.builder.build_unconditional_branch(loop_).unwrap();

				self.builder.position_at_end(loop_);

				self.codegen_block(block)?;
				self.builder.build_unconditional_branch(loop_).unwrap();

				self.builder.position_at_end(cont);

				self.loop_stack.pop();

				Ok(false)
			}
		}
	}

	fn codegen_expr(&mut self, expr: &hir::Expr) -> Result<MaybeValue<'ctx>> {
		let value = match &expr.kind {
			hir::ExprKind::Literal { lit, sym } => {
				let sym = self.scx.symbols.resolve(*sym);

				let ty = self.typeck_results.get(&expr.id).unwrap();
				let ty = self.to_llvm_type(ty).unwrap();

				let val = match lit {
					lexer::LiteralKind::Integer => ty
						.into_int_type()
						.const_int(sym.parse::<u64>().unwrap(), true)
						.as_basic_value_enum(),
					lexer::LiteralKind::Float => todo!(),
					lexer::LiteralKind::Str => todo!(),
				};
				MaybeValue::Value(val)
			}
			hir::ExprKind::Access { path } => {
				let place = *self.variables.get(&path.simple().sym).unwrap();

				let ty = self.typeck_results.get(&expr.id).unwrap();
				let ty = self.to_llvm_type(ty).unwrap();

				let value = self
					.builder
					.build_load(ty, place, &self.scx.symbols.resolve(path.simple().sym))
					.unwrap()
					.as_basic_value_enum();

				MaybeValue::Value(value.clone())
			}
			hir::ExprKind::Assign { target, value } => {
				let hir::ExprKind::Access { path } = &target.kind else {
					todo!("invalid lvalue");
				};

				let place = *self.variables.get(&path.simple().sym).unwrap();
				let expr_value = self.codegen_expr(value)?;
				match expr_value {
					MaybeValue::Value(value) => {
						self.builder.build_store(place, value).unwrap();
					}
					MaybeValue::Zst | MaybeValue::Never => {}
				};

				expr_value
			}
			hir::ExprKind::Binary { op, left, right } => self.codegen_bin_op(*op, left, right)?,

			hir::ExprKind::Unary { op, expr } => todo!(),
			hir::ExprKind::Method { expr, name, params } => todo!(),
			hir::ExprKind::Field { expr, name: ident } => todo!(),
			hir::ExprKind::Deref { expr } => todo!(),

			hir::ExprKind::FnCall { expr, args } => {
				let hir::ExprKind::Access { path } = &expr.kind else {
					todo!("not a fn")
				};
				let func = self
					.module
					.get_function(&self.scx.symbols.resolve(path.simple().sym))
					.unwrap();
				if args.bit.len() != func.count_params() as usize {
					return Err("fn call args count mismatch");
				}

				let mut argsz = Vec::new();
				for arg in &args.bit {
					let val = self.codegen_expr(&arg)?;
					if let Some(Some(val)) = val.as_value_enum() {
						argsz.push(val.into());
					}
				}

				let call = self.builder.build_call(func, &argsz, "call").unwrap();
				let value = call.try_as_basic_value().basic();
				match value {
					Some(val) => MaybeValue::Value(val),
					None => MaybeValue::Zst,
				}
			}
			hir::ExprKind::If {
				cond,
				conseq,
				altern,
			} => self.codegen_if(cond, conseq, altern.as_deref())?,

			hir::ExprKind::Return { expr } => {
				if let Some(expr) = expr {
					match self.codegen_expr(expr)? {
						MaybeValue::Value(value) => {
							self.builder.build_return(Some(&value)).unwrap();
						}
						MaybeValue::Zst | MaybeValue::Never => {
							self.builder.build_return(None).unwrap();
						}
					}
				} else {
					self.builder.build_return(None).unwrap();
				};

				MaybeValue::Never
			}
			hir::ExprKind::Break { expr, label } => {
				let (_loop_, cont) = *self.loop_stack.last().unwrap();

				if let Some(expr) = expr {
					match self.codegen_expr(expr)? {
						// TODO: transfer value to callsite
						MaybeValue::Value(value) => {}
						MaybeValue::Zst | MaybeValue::Never => {}
					}
				} else {
				}
				self.builder.build_unconditional_branch(cont).unwrap();
				MaybeValue::Never
			}
			hir::ExprKind::Continue { label } => {
				let (loop_, _cont) = *self.loop_stack.last().unwrap();
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
			MaybeValue::Zst => panic!(),
			MaybeValue::Never => return Ok(MaybeValue::Never),
		};

		let then_bb = self.ctx.append_basic_block(self.function, "then");
		let else_bb = self.ctx.append_basic_block(self.function, "else");
		let merge_bb = self.ctx.append_basic_block(self.function, "merge");
		self.builder
			.build_conditional_branch(condition, then_bb, else_bb)
			.unwrap();
		self.builder.position_at_end(then_bb);
		let then_val = self.codegen_block(conseq)?;
		self.builder.build_unconditional_branch(merge_bb).unwrap();
		let then_bb = self.builder.get_insert_block().unwrap();
		self.builder.position_at_end(else_bb);
		let else_val = self.codegen_block(altern.as_ref().unwrap())?;
		self.builder.build_unconditional_branch(merge_bb).unwrap();
		let else_bb = self.builder.get_insert_block().unwrap();
		self.builder.position_at_end(merge_bb);
		let phi = self
			.builder
			.build_phi(self.ctx.i64_type(), "if_ret")
			.unwrap();
		if let Some(Some(then_val)) = then_val.as_value() {
			phi.add_incoming(&[(then_val, then_bb)]);
		}
		if let Some(Some(else_val)) = else_val.as_value() {
			phi.add_incoming(&[(else_val, else_bb)]);
		}
		Ok(MaybeValue::Value(phi.as_basic_value()))
	}

	fn codegen_bin_op(
		&mut self,
		op: ast::Spanned<ast::BinaryOp>,
		left: &hir::Expr,
		right: &hir::Expr,
	) -> Result<MaybeValue<'ctx>> {
		tracing::trace!("codegen_bin_op");

		let lhs = self.codegen_expr(left)?;
		let rhs = self.codegen_expr(right)?;

		// cannot be zst
		let (lhs, rhs) = match (lhs, rhs) {
			(MaybeValue::Value(lhs), MaybeValue::Value(rhs)) => (lhs, rhs),
			(MaybeValue::Never, _) | (_, MaybeValue::Never) => return Ok(MaybeValue::Never),
			_ => panic!(),
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
