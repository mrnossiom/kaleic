use std::{fmt::Write as _, ops::Deref, path::Path, rc::Rc, sync::Arc};

use cranelift::prelude::{isa::TargetIsa, *};
use cranelift_control::ControlPlane;
use cranelift_jit::JITModule;
use cranelift_module::{DataDescription, FuncId, Linkage, Module, default_libcall_names};
use cranelift_object::{ObjectBuilder, ObjectModule};
use rustc_hash::FxHashMap;

use crate::{
	ast::{self, BinaryOp},
	bug,
	codegen::{Backend, CodeGenBackend, JitBackend, ObjectBackend},
	collect::DefId,
	hir::{self, Enum, ExprId, Function, Struct},
	session::{ArtefactKind, ScxHandle, SessionCtx},
	symbols::Symbol,
	ty::{self, LateTy, TyCtx, TyKind},
};

type Result<T> = std::result::Result<T, &'static str>;

pub enum MaybeValue {
	Value(Value),
	/// Zero-sized value
	Zst,
	Never,
}

pub struct Generator<'tcx, M> {
	tcx: &'tcx TyCtx<'tcx>,

	module: M,
	isa: Arc<dyn TargetIsa + 'static>,
	builder_context: FunctionBuilderContext,

	functions: FxHashMap<DefId, FuncId>,
}

impl<'tcx, M: Module> Generator<'tcx, M> {
	pub(crate) fn new(tcx: &'tcx TyCtx, isa: Arc<dyn TargetIsa + 'static>, module: M) -> Self {
		Self {
			tcx,
			module,
			isa,
			builder_context: FunctionBuilderContext::new(),
			functions: FxHashMap::default(),
		}
	}

	// Return `None` on non-concrete types (e.g. zst, never)
	// TODO: remove duplicate on function generator
	fn to_cl_type(&self, ty: &LateTy) -> Option<Type> {
		match ty.clone() {
			ty::TyKind::Primitive(kind) => match kind {
				ty::PrimitiveKind::Unit | ty::PrimitiveKind::Never => None,
				ty::PrimitiveKind::Bool => Some(types::I8),
				ty::PrimitiveKind::UnsignedInt | ty::PrimitiveKind::SignedInt => Some(types::I32),
				ty::PrimitiveKind::Float => Some(types::F32),
				ty::PrimitiveKind::Str => todo!(),
			},
			ty::TyKind::Pointer(_kind) => Some(self.isa.pointer_type()),
			ty::TyKind::Fn(_fn_decl) => Some(self.isa.pointer_type()),
			ty::TyKind::Struct(enum_) => todo!(),
			ty::TyKind::Enum(struct_) => todo!(),
			ty::TyKind::Error => {
				bug!("error type kind is a placeholder and should not reach codegen")
			}
		}
	}
}

impl<'tcx> Generator<'tcx, JITModule> {
	pub(crate) fn new_jit(tcx: &'tcx TyCtx) -> Self {
		use ::cranelift::prelude::{Configurable, settings};
		use cranelift_jit::{JITBuilder, JITModule};
		use cranelift_module::default_libcall_names;

		let mut flag_builder = settings::builder();
		flag_builder.set("opt_level", "speed_and_size").unwrap();
		let isa = cranelift_native::builder()
			.unwrap()
			.finish(settings::Flags::new(flag_builder))
			.unwrap();

		let builder = JITBuilder::with_isa(isa.clone(), default_libcall_names());
		let module = JITModule::new(builder);

		Self::new(tcx, isa, module)
	}
}

impl<'tcx> Generator<'tcx, ObjectModule> {
	pub(crate) fn new_object(tcx: &'tcx TyCtx) -> Self {
		let mut flag_builder = settings::builder();
		flag_builder.set("opt_level", "speed_and_size").unwrap();

		let isa = cranelift_native::builder()
			.unwrap()
			.finish(settings::Flags::new(flag_builder))
			.unwrap();

		let builder = ObjectBuilder::new(isa.clone(), "out", default_libcall_names()).unwrap();

		// builder.per_function_section(per_function_section) what is this?

		let module = ObjectModule::new(builder);

		Self::new(tcx, isa, module)
	}
}

impl<M: Module> Generator<'_, M> {
	pub(crate) fn lower_signature(&self, decl: &ty::FnDecl<LateTy>) -> Signature {
		let mut signature = self.module.make_signature();

		for ty::Param { name: _, ty, id } in &decl.inputs {
			let Some(type_) = self.to_cl_type(ty) else {
				continue;
			};
			signature.params.push(AbiParam::new(type_));
		}
		if let Some(type_) = self.to_cl_type(&decl.output) {
			signature.returns.push(AbiParam::new(type_));
		}

		signature
	}

	pub(crate) fn declare_func(
		&mut self,
		name: Symbol,
		def_id: DefId,
		decl: &ty::FnDecl<LateTy>,
		linkage: Linkage,
	) -> Result<FuncId> {
		if self.functions.contains_key(&def_id) {
			return Err("already defined");
		}

		let signature = self.lower_signature(decl);

		let func_id = self
			.module
			.declare_function(&self.tcx.scx().symbols.resolve(name), linkage, &signature)
			.unwrap();

		self.functions.insert(def_id, func_id);
		Ok(func_id)
	}

	fn define_func(
		&mut self,
		func_id: FuncId,
		def_id: DefId,
		decl: &ty::FnDecl<LateTy>,
		body: &hir::Block,
	) -> Result<()> {
		let mut context = self.module.make_context();

		// TODO: this computes the signature a second time after declaration
		context.func.signature = self.lower_signature(decl);

		let builder = FunctionBuilder::new(&mut context.func, &mut self.builder_context);

		let typeck_results = &self.tcx.typeck_results.borrow_key(&def_id);

		let mut generator = FunctionGenerator {
			scx: self.tcx.scx(),

			typeck_results,

			builder,
			functions: &self.functions,
			module: &mut self.module,
			isa: self.isa.clone(),

			values: FxHashMap::default(),
			loop_stack: Vec::default(),
		};

		generator.codegen_body(decl, body)?;

		generator.builder.finalize();

		context
			.optimize(self.module.isa(), &mut ControlPlane::default())
			.unwrap();

		self.tcx.scx().register_artefact(
			&ArtefactKind::BackendIr(def_id, Backend::Cranelift),
			|artefact| write!(artefact, "{}", context.func.display()),
		);

		self.module.define_function(func_id, &mut context).unwrap();

		self.module.clear_context(&mut context);

		Ok(())
	}
}

impl<M: Module> CodeGenBackend for Generator<'_, M> {
	fn codegen_root(&mut self, hir: &hir::Root) {
		let mut function_ids = FxHashMap::default();

		let type_env = self.tcx.type_env.borrow();

		for item in &hir.items {
			match &item.kind {
				hir::ItemKind::Function(Function { name, decl, body }) => {
					let TyKind::Fn(decl) = &*type_env[&item.def_id] else {
						todo!()
					};

					// TODO: react to abi attribute
					// TODO: change to hidden by default
					let func_id = self
						.declare_func(name.sym, item.def_id, decl, Linkage::Hidden)
						.unwrap();
					function_ids.insert(item.def_id, func_id);
				}
				hir::ItemKind::ForeignMod { items } => {
					for item in items {
						match &item.kind {
							hir::ForeignItemKind::Function(Function { name, decl, body }) => {
								let TyKind::Fn(decl) = &*type_env[&item.def_id] else {
									todo!()
								};

								let _func_id = self
									.declare_func(name.sym, item.def_id, decl, Linkage::Import)
									.unwrap();
							}
						}
					}
				}

				hir::ItemKind::Struct(Struct { .. }) | hir::ItemKind::Enum(Enum { .. }) => {
					// TODO
					todo!("codegen constructors here?")
				}
				hir::ItemKind::TypeAlias(..) | hir::ItemKind::Trait { .. } => {}
				hir::ItemKind::TraitImpl { .. } => {
					// TODO
					todo!("codegen methods")
				}
			}
		}
		for item in &hir.items {
			match &item.kind {
				hir::ItemKind::Function(Function { name, decl, body }) => {
					let type_env = self.tcx.type_env.borrow();
					let TyKind::Fn(decl) = &*type_env[&item.def_id] else {
						todo!()
					};

					let Some(func_id) = function_ids.get(&item.def_id) else {
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
				| hir::ItemKind::TypeAlias(..)
				| hir::ItemKind::Trait { .. } => {}
			}
		}
	}
}

impl JitBackend for Generator<'_, JITModule> {
	fn finalize(&mut self) {
		self.module.finalize_definitions().unwrap();
	}

	fn call_main(&self) {
		let main_fn_id = self.tcx.main_fn_id.borrow();
		let main_id = self.functions[&main_fn_id];
		let func = self.module.get_finalized_function(main_id);

		// TODO: this is unsafe for so much reasons, but we assume that our codegen is perfect :)
		// SAFETY: main signature is enforced
		let main = unsafe { std::mem::transmute::<*const u8, fn()>(func) };

		main();
	}
}

impl ObjectBackend for Generator<'_, ObjectModule> {
	fn write_object(self: Box<Self>, path: &Path) {
		let object = self.module.finish();
		let bytes = object.emit().unwrap();
		std::fs::write(path, bytes).unwrap();
	}
}

pub struct FunctionGenerator<'scx, 'bld> {
	scx: &'scx SessionCtx,

	typeck_results: &'scx FxHashMap<ExprId, Rc<LateTy>>,

	builder: FunctionBuilder<'bld>,
	functions: &'bld FxHashMap<DefId, FuncId>,
	module: &'bld mut dyn Module,
	isa: Arc<dyn TargetIsa + 'static>,

	values: FxHashMap<hir::NodeId, Option<Variable>>,
	loop_stack: Vec<(Block, Block)>,
}

/// Codegen hir functions
impl FunctionGenerator<'_, '_> {
	// TODO: remove duplicate
	fn to_cl_type(&self, ty: &LateTy) -> Option<Type> {
		match ty.clone() {
			ty::TyKind::Primitive(kind) => match kind {
				ty::PrimitiveKind::Unit | ty::PrimitiveKind::Never => None,
				ty::PrimitiveKind::Bool => Some(types::I8),
				ty::PrimitiveKind::UnsignedInt | ty::PrimitiveKind::SignedInt => Some(types::I32),
				ty::PrimitiveKind::Float => Some(types::F32),
				ty::PrimitiveKind::Str => todo!(),
			},
			ty::TyKind::Pointer(_kind) => todo!(),
			ty::TyKind::Fn(_fn_decl) => Some(self.isa.pointer_type()),
			ty::TyKind::Struct(struct_) => todo!(),
			ty::TyKind::Enum(enum_) => todo!(),
			ty::TyKind::Error => {
				bug!("error type kind is a placeholder and should not reach codegen")
			}
		}
	}

	fn codegen_body(&mut self, decl: &ty::FnDecl<LateTy>, block: &hir::Block) -> Result<()> {
		let entry_block = self.builder.create_block();
		self.builder
			.append_block_params_for_function_params(entry_block);
		self.builder.switch_to_block(entry_block);
		self.builder.seal_block(entry_block);

		let mut block_param_idx = 0;
		for ty::Param { name, ty, id } in &decl.inputs {
			let Some(ty) = self.to_cl_type(ty) else {
				// skip ZSTs
				continue;
			};

			let value = self.builder.block_params(entry_block)[block_param_idx];

			let variable = self.builder.declare_var(ty);
			self.builder.def_var(variable, value);

			self.values.insert(*id, Some(variable));

			block_param_idx += 1;
		}

		match self.codegen_block(block)? {
			MaybeValue::Value(value) => _ = self.builder.ins().return_(&[value]),
			MaybeValue::Zst => _ = self.builder.ins().return_(&[]),
			MaybeValue::Never => {}
		}

		Ok(())
	}

	fn codegen_block(&mut self, block: &hir::Block) -> Result<MaybeValue> {
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
			hir::StmtKind::Expr { expr } => match self.codegen_expr(expr)? {
				MaybeValue::Value(_) | MaybeValue::Zst => {}
				MaybeValue::Never => return Ok(true),
			},
			hir::StmtKind::Let {
				name,
				value,
				ty: _,
				mutable: _,
			} => match self.codegen_expr(value)? {
				MaybeValue::Value(expr_value) => {
					let ty = &self.typeck_results[&value.expr_id()];
					let ty = self.to_cl_type(ty).unwrap();
					let variable = self.builder.declare_var(ty);
					self.builder.def_var(variable, expr_value);

					self.values.insert(stmt.id, Some(variable));
				}
				MaybeValue::Zst => {
					self.values.insert(stmt.id, None);
				}
				MaybeValue::Never => {}
			},
		}
		Ok(false)
	}

	fn codegen_expr(&mut self, expr: &hir::Expr) -> Result<MaybeValue> {
		let value = match &expr.kind {
			hir::ExprKind::LiteralInt { sym } => {
				let number = {
					let lit = self.scx.symbols.resolve(*sym);
					lit.parse::<i64>().unwrap()
				};

				let ty = &self.typeck_results[&expr.expr_id()];
				let int_ty = self.to_cl_type(ty).unwrap();
				let value = self.builder.ins().iconst(int_ty, number);
				MaybeValue::Value(value)
			}
			hir::ExprKind::LiteralFloat { sym } => {
				let number = {
					let lit = self.scx.symbols.resolve(*sym);
					lit.parse::<f64>().unwrap()
				};
				let ty = &self.typeck_results[&expr.expr_id()];
				let int_ty = self.to_cl_type(ty).unwrap();
				// FIXME: take ty into account
				let value = self.builder.ins().f64const(number);
				MaybeValue::Value(value)
			}
			hir::ExprKind::LiteralStr { sym } => {
				let data = {
					let lit = self.scx.symbols.resolve(*sym);
					let mut data = DataDescription::new();
					data.define(lit.to_owned().into_boxed_str().into());
					data
				};
				let data_id = self.module.declare_anonymous_data(false, false).unwrap();

				self.module.define_data(data_id, &data).unwrap();

				let global_value = self.module.declare_data_in_func(data_id, self.builder.func);
				let value = self
					.builder
					.ins()
					.global_value(self.module.isa().pointer_type(), global_value);

				MaybeValue::Value(value)
			}

			hir::ExprKind::Access { path } => {
				let hir_id = path.resolved.into_local().unwrap();
				match self.values.get(&hir_id) {
					Some(Some(var)) => MaybeValue::Value(self.builder.use_var(*var)),
					Some(None) => MaybeValue::Zst,
					None => return Err("var undefined"),
				}
			}

			hir::ExprKind::Unit => MaybeValue::Zst,

			hir::ExprKind::Unary { op, expr } => todo!("codegen unary {op:?} {expr:?}"),

			hir::ExprKind::Binary { op, left, right } => {
				MaybeValue::Value(self.codegen_bin_op(*op, left, right)?)
			}
			hir::ExprKind::FnCall { expr, args } => {
				let mut argsz = Vec::new();
				for arg in &args.bit {
					match self.codegen_expr(arg)? {
						MaybeValue::Value(expr_value) => argsz.push(expr_value),
						MaybeValue::Zst | MaybeValue::Never => {}
					}
				}

				let call = if let hir::ExprKind::Access { path } = &expr.kind {
					let item_id = path.resolved.into_def().unwrap();
					let func_id = self.functions[&item_id];

					let local_func = self.module.declare_func_in_func(func_id, self.builder.func);

					self.builder.ins().call(local_func, &argsz)
				} else {
					let callee = match self.codegen_expr(expr)? {
						MaybeValue::Value(callee) => callee,
						value @ (MaybeValue::Zst | MaybeValue::Never) => return Ok(value),
					};
					let sig = todo!();

					self.builder.ins().call_indirect(sig, callee, &argsz)
				};

				let inst_results = self.builder.inst_results(call);
				match inst_results.len() {
					0 => MaybeValue::Zst,
					1 => MaybeValue::Value(inst_results[0]),
					_ => panic!(),
				}
			}
			hir::ExprKind::If {
				cond,
				conseq,
				altern,
			} => self.codegen_if(cond, conseq, altern.as_deref())?,
			hir::ExprKind::Loop { block } => {
				let loop_ = self.builder.create_block();
				let cont = self.builder.create_block();

				let block_ty = &self.typeck_results[&expr.expr_id()];
				if let Some(block_ty) = self.to_cl_type(block_ty) {
					self.builder.append_block_param(cont, block_ty);
				}

				self.loop_stack.push((loop_, cont));

				self.builder.ins().jump(loop_, &[]);

				self.builder.switch_to_block(loop_);

				self.codegen_block(block)?;
				self.builder.ins().jump(loop_, &[]);

				self.builder.seal_block(loop_);

				self.builder.switch_to_block(cont);
				self.builder.seal_block(cont);

				self.loop_stack.pop();

				let block_params = self.builder.block_params(cont);
				match block_params.len() {
					0 => MaybeValue::Zst,
					1 => MaybeValue::Value(block_params[0]),
					_ => panic!(),
				}
			}

			hir::ExprKind::Method { expr, name, params } => todo!(),
			hir::ExprKind::Field { expr, name } => todo!(),
			hir::ExprKind::Deref { expr } => todo!(),

			hir::ExprKind::Assign { target, value } => {
				let hir::ExprKind::Access { path } = &target.kind else {
					todo!("invalid lvalue");
				};

				let hir_id = path.resolved.into_local().unwrap();
				let Some(variable) = self.values[&hir_id] else {
					// handle ZSTs
					return Ok(MaybeValue::Zst);
				};

				let maybe_value = self.codegen_expr(value)?;
				if let MaybeValue::Value(expr_value) = maybe_value {
					self.builder.def_var(variable, expr_value);
				}
				maybe_value
			}

			hir::ExprKind::Return { expr } => {
				match self.codegen_expr(expr)? {
					MaybeValue::Value(expr_value) => _ = self.builder.ins().return_(&[expr_value]),
					MaybeValue::Zst => _ = self.builder.ins().return_(&[]),
					MaybeValue::Never => {}
				}

				MaybeValue::Never
			}
			hir::ExprKind::Break { expr, label } => {
				let (_, cont) = *self.loop_stack.last().unwrap();

				match self.codegen_expr(expr)? {
					MaybeValue::Value(expr_value) => {
						_ = self.builder.ins().jump(cont, &[expr_value.into()]);
					}
					MaybeValue::Zst => _ = self.builder.ins().jump(cont, &[]),
					MaybeValue::Never => {}
				}

				MaybeValue::Never
			}
			hir::ExprKind::Continue { label } => {
				let (loop_, _) = *self.loop_stack.last().unwrap();

				self.builder.ins().jump(loop_, &[]);

				MaybeValue::Never
			}
		};
		Ok(value)
	}
}

/// Codegen bits
impl FunctionGenerator<'_, '_> {
	fn codegen_bin_op(
		&mut self,
		op: ast::Spanned<ast::BinaryOp>,
		left: &hir::Expr,
		right: &hir::Expr,
	) -> Result<Value> {
		// cannot be zst
		let lhs = self.codegen_expr(left)?;
		let rhs = self.codegen_expr(right)?;

		let (MaybeValue::Value(lhs), MaybeValue::Value(rhs)) = (lhs, rhs) else {
			panic!()
		};

		let ins = self.builder.ins();
		let value = match op.bit {
			BinaryOp::Plus => ins.iadd(lhs, rhs),
			BinaryOp::Minus => ins.isub(lhs, rhs),
			BinaryOp::Mul => ins.imul(lhs, rhs),
			BinaryOp::Div => ins.udiv(lhs, rhs),
			BinaryOp::Mod => ins.urem(lhs, rhs),

			BinaryOp::And => ins.band(lhs, rhs),
			BinaryOp::Or => ins.bor(lhs, rhs),
			BinaryOp::Xor => ins.bxor(lhs, rhs),

			BinaryOp::Shl => ins.ishl(lhs, rhs),
			BinaryOp::Shr => ins.sshr(lhs, rhs),

			BinaryOp::Gt => ins.icmp(IntCC::SignedGreaterThan, lhs, rhs),
			BinaryOp::Ge => ins.icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs),
			BinaryOp::Lt => ins.icmp(IntCC::SignedLessThan, lhs, rhs),
			BinaryOp::Le => ins.icmp(IntCC::SignedLessThanOrEqual, lhs, rhs),

			BinaryOp::EqEq => ins.icmp(IntCC::Equal, lhs, rhs),
			BinaryOp::Ne => ins.icmp(IntCC::NotEqual, lhs, rhs),
		};

		Ok(value)
	}

	fn codegen_if(
		&mut self,
		cond: &hir::Expr,
		conseq: &hir::Block,
		altern: Option<&hir::Block>,
	) -> Result<MaybeValue> {
		let then_block = self.builder.create_block();
		let else_block = altern.as_ref().map(|_| self.builder.create_block());
		let cont_block = self.builder.create_block();

		let condition = match self.codegen_expr(cond)? {
			MaybeValue::Value(val) => val,
			MaybeValue::Zst => bug!("a ZST cannot be used as a condition"),
			MaybeValue::Never => return Ok(MaybeValue::Never),
		};

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

		if let Some(ty) = self.to_cl_type(&ty) {
			self.builder.append_block_param(cont_block, ty);
		}

		self.builder.ins().brif(
			condition,
			then_block,
			&[],
			else_block.unwrap_or(cont_block),
			&[],
		);
		self.builder.switch_to_block(then_block);
		self.builder.seal_block(then_block);

		match self.codegen_block(conseq)? {
			MaybeValue::Value(then_ret) => {
				_ = self.builder.ins().jump(cont_block, &[then_ret.into()]);
			}
			MaybeValue::Zst => _ = self.builder.ins().jump(cont_block, &[]),
			MaybeValue::Never => {}
		}
		if let Some(altern) = altern {
			// TODO
			let else_block = else_block.unwrap();

			self.builder.switch_to_block(else_block);
			self.builder.seal_block(else_block);

			match self.codegen_block(altern)? {
				MaybeValue::Value(else_ret) => {
					_ = self.builder.ins().jump(cont_block, &[else_ret.into()]);
				}
				MaybeValue::Zst => _ = self.builder.ins().jump(cont_block, &[]),
				MaybeValue::Never => {}
			}
		}
		self.builder.switch_to_block(cont_block);
		self.builder.seal_block(cont_block);

		let block_params = self.builder.block_params(cont_block);
		Ok(match block_params.len() {
			0 => MaybeValue::Zst,
			1 => MaybeValue::Value(block_params[0]),
			_ => panic!(),
		})
	}
}
