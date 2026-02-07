use std::{collections::HashMap, sync::Arc};

use cranelift::prelude::{isa::TargetIsa, *};
use cranelift_control::ControlPlane;
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_object::{ObjectModule, ObjectProduct};

use crate::{
	ast::{self, BinaryOp},
	bug,
	codegen::{CodeGenBackend, JitBackend, ObjectBackend},
	hir::{self, Enum, Function, Struct},
	lexer,
	session::{PrintKind, SessionCtx, Symbol},
	ty::{self, TyCtx, TyKind},
};

type Result<T> = std::result::Result<T, &'static str>;

pub enum MaybeValue {
	Value(Value),
	/// Zero-sized value
	Zst,
	Never,
}

impl MaybeValue {
	fn with_slice(&self, func: impl FnOnce(&[Value])) {
		match self {
			Self::Value(val) => func(&[*val]),
			Self::Zst => func(&[]),
			Self::Never => {}
		}
	}
}

pub struct Generator<'tcx, M: Module> {
	tcx: &'tcx TyCtx<'tcx>,

	module: M,
	isa: Arc<dyn TargetIsa + 'static>,
	builder_context: FunctionBuilderContext,

	functions: HashMap<Symbol, FuncId>,
}

impl<'tcx, M: Module> Generator<'tcx, M> {
	pub fn new(tcx: &'tcx TyCtx, isa: Arc<dyn TargetIsa + 'static>, module: M) -> Self {
		Self {
			tcx,
			module,
			isa,
			builder_context: FunctionBuilderContext::new(),
			functions: HashMap::new(),
		}
	}

	// Return `None` on non-concrete types (e.g. zst, never)
	// TODO: remove duplicate on function generator
	fn to_cl_type(&self, output: &ty::TyKind) -> Option<Type> {
		match output.clone() {
			ty::TyKind::Primitive(kind) => match kind {
				ty::PrimitiveKind::Void | ty::PrimitiveKind::Never => None,
				ty::PrimitiveKind::Bool => Some(types::I8),
				ty::PrimitiveKind::UnsignedInt | ty::PrimitiveKind::SignedInt => Some(types::I32),
				ty::PrimitiveKind::Float => Some(types::F32),
				ty::PrimitiveKind::Str => todo!(),
			},
			ty::TyKind::Pointer(_kind) => todo!(),
			ty::TyKind::Fn(_fn_decl) => Some(self.isa.pointer_type()),
			ty::TyKind::Struct(enum_) => todo!(),
			ty::TyKind::Enum(struct_) => todo!(),
			ty::TyKind::Ref(ref_) => {
				self.to_cl_type(&self.tcx.ty_env.borrow().as_ref().unwrap()[&ref_])
			}
			ty::TyKind::Error => {
				bug!("error type kind is a placeholder and should not reach codegen")
			}
		}
	}
}

impl<'tcx> Generator<'tcx, JITModule> {
	pub fn new_jit(tcx: &'tcx TyCtx) -> Self {
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
	pub fn new_object(tcx: &'tcx TyCtx) -> Self {
		use ::cranelift::prelude::{Configurable, settings};
		use cranelift_module::default_libcall_names;
		use cranelift_object::{ObjectBuilder, ObjectModule};

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
	pub fn lower_signature(&mut self, decl: &ty::FnDecl) -> Signature {
		let mut signature = self.module.make_signature();

		for ty::Param { name: _, ty } in &decl.inputs {
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

	pub fn declare_func(
		&mut self,
		name: Symbol,
		decl: &ty::FnDecl,
		linkage: Linkage,
	) -> Result<FuncId> {
		if self.functions.contains_key(&name) {
			return Err("already defined");
		}

		let signature = self.lower_signature(decl);

		let func_name = self.tcx.scx.symbols.resolve(name);
		let func_id = self
			.module
			.declare_function(&func_name, linkage, &signature)
			.unwrap();

		self.functions.insert(name, func_id);
		Ok(func_id)
	}

	fn define_function(
		&mut self,
		func_id: FuncId,
		decl: &ty::FnDecl,
		body: &hir::Block,
	) -> Result<()> {
		let mut context = self.module.make_context();

		// TODO: this computes the signature a second time after declaration
		context.func.signature = self.lower_signature(decl);

		let params = decl
			.inputs
			.iter()
			// skip zst
			.filter_map(|param| self.to_cl_type(&param.ty).map(|ty| (param.name, ty)))
			.collect::<Vec<_>>();

		let mut builder = FunctionBuilder::new(&mut context.func, &mut self.builder_context);

		let entry_block = builder.create_block();
		builder.append_block_params_for_function_params(entry_block);
		builder.switch_to_block(entry_block);
		builder.seal_block(entry_block);

		let mut values = HashMap::new();
		for (i, (ident, ty)) in params.into_iter().enumerate() {
			let value = builder.block_params(entry_block)[i];

			let variable = builder.declare_var(ty);
			builder.def_var(variable, value);

			values.insert(ident.sym, Some(variable));
		}

		let borrow = self.tcx.ty_env.borrow();
		let ty_env = borrow.as_ref().unwrap();
		let borrow = self.tcx.typeck_results.borrow();
		let typeck_results = borrow.as_ref().unwrap();

		let mut generator = FunctionGenerator {
			scx: self.tcx.scx,

			ty_env: &ty_env,
			typeck_results: &typeck_results,

			builder,
			functions: &self.functions,
			module: &mut self.module,
			isa: self.isa.clone(),
			values,

			loops: Vec::default(),
		};

		let return_value = generator.codegen_block(body)?;
		return_value.with_slice(|vals| {
			generator.builder.ins().return_(vals);
		});

		generator.builder.finalize();

		context
			.optimize(self.module.isa(), &mut ControlPlane::default())
			.unwrap();

		if self.tcx.scx.options.print.contains(&PrintKind::BackendIr) {
			print!("{}", context.func.display());
		}

		self.module.define_function(func_id, &mut context).unwrap();

		self.module.clear_context(&mut context);

		Ok(())
	}
}

impl<M: Module> CodeGenBackend for Generator<'_, M> {
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
							let func_id =
							// TODO: change to hidden by default
								self.declare_func(name.sym, &decl, Linkage::Hidden).unwrap();
							function_ids.insert(item.id, func_id);
						}
						hir::Abi::C => {
							let _func_id =
								self.declare_func(name.sym, &decl, Linkage::Import).unwrap();
						}
					}
				}

				hir::ItemKind::Struct(Struct { .. }) | hir::ItemKind::Enum(Enum { .. }) => {
					todo!("codegen constructors here?")
				}
				hir::ItemKind::TypeAlias(_) | hir::ItemKind::Trait { .. } => {}
				hir::ItemKind::TraitImpl { .. } => todo!("codegen methods"),
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
						println!("assuming fn {:?} is external", name.sym);
						continue;
					};

					let body = body.as_ref().unwrap();
					self.define_function(*func_id, &decl, &body).unwrap();
				}

				hir::ItemKind::Struct(Struct { .. }) | hir::ItemKind::Enum(Enum { .. }) => todo!(),
				hir::ItemKind::TypeAlias(_) | hir::ItemKind::Trait { .. } => {}
				hir::ItemKind::TraitImpl { .. } => todo!(),
			}
		}
	}
}

impl JitBackend for Generator<'_, JITModule> {
	fn call_main(&mut self) {
		self.module.finalize_definitions().unwrap();

		let main = self.tcx.scx.symbols.intern("main");
		let main_id = self.functions.get(&main).unwrap();
		let func = self.module.get_finalized_function(*main_id);
		// TODO: this is unsafe as some functions ask for arguments, and lot a more reasons
		#[allow(unsafe_code)]
		let main = unsafe { std::mem::transmute::<*const u8, fn()>(func) };

		main();
	}
}

impl ObjectBackend for Generator<'_, ObjectModule> {
	fn get_object(self) -> ObjectProduct {
		self.module.finish()
	}
}

pub struct FunctionGenerator<'scx, 'bld> {
	scx: &'scx SessionCtx,

	ty_env: &'scx HashMap<hir::NodeId, ty::TyKind>,
	typeck_results: &'scx HashMap<hir::NodeId, ty::TyKind>,

	builder: FunctionBuilder<'bld>,
	functions: &'bld HashMap<Symbol, FuncId>,
	module: &'bld mut dyn Module,
	isa: Arc<dyn TargetIsa + 'static>,
	values: HashMap<Symbol, Option<Variable>>,

	loops: Vec<(Block, Block)>,
}

/// Codegen hir functions
impl FunctionGenerator<'_, '_> {
	// TODO: remove duplicate
	fn to_cl_type(&self, output: &ty::TyKind) -> Option<Type> {
		match output.clone() {
			ty::TyKind::Primitive(kind) => match kind {
				ty::PrimitiveKind::Void | ty::PrimitiveKind::Never => None,
				ty::PrimitiveKind::Bool => Some(types::I8),
				ty::PrimitiveKind::UnsignedInt | ty::PrimitiveKind::SignedInt => Some(types::I32),
				ty::PrimitiveKind::Float => Some(types::F32),
				ty::PrimitiveKind::Str => todo!(),
			},
			ty::TyKind::Pointer(_kind) => todo!(),
			ty::TyKind::Fn(_fn_decl) => Some(self.isa.pointer_type()),
			ty::TyKind::Struct(struct_) => todo!(),
			ty::TyKind::Enum(enum_) => todo!(),
			ty::TyKind::Ref(ref_) => self.to_cl_type(&self.ty_env[&ref_]),
			ty::TyKind::Error => {
				bug!("error type kind is a placeholder and should not reach codegen")
			}
		}
	}

	fn codegen_block(&mut self, block: &hir::Block) -> Result<MaybeValue> {
		tracing::trace!(id = ?block.id, "codegen_block");

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
		tracing::trace!(id = ?stmt.id, "codegen_stmt");

		match &stmt.kind {
			hir::StmtKind::Expr(expr) => match self.codegen_expr(expr)? {
				MaybeValue::Value(_) | MaybeValue::Zst => {}
				MaybeValue::Never => return Ok(true),
			},
			hir::StmtKind::Let { ident, value, .. } => match self.codegen_expr(value)? {
				MaybeValue::Value(expr_value) => {
					let ty = self.typeck_results.get(&value.id).unwrap();
					let ty = self.to_cl_type(ty).unwrap();
					let variable = self.builder.declare_var(ty);
					self.builder.def_var(variable, expr_value);

					self.values.insert(ident.sym, Some(variable));
				}
				MaybeValue::Zst => {
					self.values.insert(ident.sym, None);
				}
				MaybeValue::Never => {}
			},
			hir::StmtKind::Loop(block) => {
				let loop_ = self.builder.create_block();
				let cont = self.builder.create_block();

				self.loops.push((loop_, cont));

				self.builder.ins().jump(loop_, &[]);

				self.builder.switch_to_block(loop_);

				self.codegen_block(block)?;
				self.builder.ins().jump(loop_, &[]);

				self.builder.seal_block(loop_);

				self.builder.switch_to_block(cont);
				self.builder.seal_block(cont);

				self.loops.pop();
			}
		}
		Ok(false)
	}

	fn codegen_expr(&mut self, expr: &hir::Expr) -> Result<MaybeValue> {
		tracing::trace!(id = ?expr.id, "codegen_expr");
		let value = match &expr.kind {
			hir::ExprKind::Literal(lit, sym) => {
				let sym = self.scx.symbols.resolve(*sym);
				let value = match lit {
					lexer::LiteralKind::Integer => {
						let ty = self.typeck_results.get(&expr.id).unwrap();
						let int_ty = self.to_cl_type(ty).unwrap();
						self.builder
							.ins()
							.iconst(int_ty, sym.parse::<i64>().unwrap())
					}
					lexer::LiteralKind::Float => {
						self.builder.ins().f64const(sym.parse::<f64>().unwrap())
					}
					lexer::LiteralKind::Str => todo!(),
				};
				MaybeValue::Value(value)
			}
			hir::ExprKind::Access(path) => {
				let path = path.segments[0];
				match self.values.get(&path.sym) {
					Some(Some(var)) => MaybeValue::Value(self.builder.use_var(*var)),
					Some(None) => MaybeValue::Zst,
					None => return Err("var undefined"),
				}
			}

			hir::ExprKind::Unary(op, expr) => todo!("codegen unary {op:?} {expr:?}"),

			hir::ExprKind::Binary(op, left, right) => {
				MaybeValue::Value(self.codegen_bin_op(*op, left, right)?)
			}
			hir::ExprKind::FnCall { expr, args } => {
				// TODO: allow indirect calls
				let hir::ExprKind::Access(path) = &expr.kind else {
					todo!("not a fn")
				};
				let path = path.segments[0];
				let Some(func_id) = self.functions.get(&path.sym) else {
					return Err("invalid fn call");
				};

				let local_func = self
					.module
					.declare_func_in_func(*func_id, self.builder.func);

				let mut argsz = Vec::new();
				for arg in &args.bit {
					match self.codegen_expr(arg)? {
						MaybeValue::Value(expr_value) => {
							argsz.push(expr_value);
						}
						MaybeValue::Zst | MaybeValue::Never => {}
					}
				}

				let call = self.builder.ins().call(local_func, &argsz);

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
			hir::ExprKind::Method(expr, name, params) => todo!(),
			hir::ExprKind::Deref(expr) => todo!(),
			hir::ExprKind::Field(expr, ident) => todo!(),
			hir::ExprKind::Assign { target, value } => {
				let hir::ExprKind::Access(target) = &target.kind else {
					todo!("invalid lvalue");
				};
				let Some(variable) = *self.values.get(&target.simple().sym).unwrap() else {
					// handle zst
					return Ok(MaybeValue::Zst);
				};

				let maybe_value = self.codegen_expr(value)?;
				match maybe_value {
					MaybeValue::Value(expr_value) => {
						self.builder.def_var(variable, expr_value);
					}
					MaybeValue::Zst | MaybeValue::Never => {}
				}
				maybe_value
			}

			hir::ExprKind::Return(expr) => {
				if let Some(expr) = expr {
					match self.codegen_expr(expr)? {
						MaybeValue::Value(expr_value) => {
							self.builder.ins().return_(&[expr_value]);
						}
						MaybeValue::Zst => {
							self.builder.ins().return_(&[]);
						}
						MaybeValue::Never => {}
					}
				} else {
					self.builder.ins().return_(&[]);
				}

				MaybeValue::Never
			}
			hir::ExprKind::Break(expr) => {
				let (_, cont) = *self.loops.last().unwrap();

				if let Some(expr) = expr {
					match self.codegen_expr(expr)? {
						MaybeValue::Value(expr_value) => {
							self.builder.ins().jump(cont, &[expr_value.into()]);
						}
						MaybeValue::Zst => {
							self.builder.ins().jump(cont, &[]);
						}
						MaybeValue::Never => {}
					}
				} else {
					self.builder.ins().jump(cont, &[]);
				}

				MaybeValue::Never
			}
			hir::ExprKind::Continue => {
				let (loop_, _) = *self.loops.last().unwrap();

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
		tracing::trace!("codegen_bin_op");
		// cannot be zst
		let lhs = self.codegen_expr(left)?;
		let rhs = self.codegen_expr(right)?;

		let (lhs, rhs) = match (lhs, rhs) {
			(MaybeValue::Value(lhs), MaybeValue::Value(rhs)) => (lhs, rhs),
			_ => panic!(),
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
		tracing::trace!(?then_block, ?else_block, ?cont_block, "codegen_if");

		let condition = match self.codegen_expr(cond)? {
			MaybeValue::Value(val) => val,
			MaybeValue::Zst | MaybeValue::Never => panic!(),
		};

		// TODO: so ugly
		let ty = conseq
			.ret
			.as_ref()
			.map(|ret| self.typeck_results.get(&ret.id).unwrap().clone())
			.unwrap_or(TyKind::Primitive(ty::PrimitiveKind::Void));

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
				self.builder.ins().jump(cont_block, &[then_ret.into()]);
			}
			MaybeValue::Zst => {
				self.builder.ins().jump(cont_block, &[]);
			}
			MaybeValue::Never => {}
		}
		if let Some(altern) = altern {
			// TODO
			let else_block = else_block.unwrap();

			self.builder.switch_to_block(else_block);
			self.builder.seal_block(else_block);

			match self.codegen_block(altern)? {
				MaybeValue::Value(else_ret) => {
					self.builder.ins().jump(cont_block, &[else_ret.into()]);
				}
				MaybeValue::Zst => {
					self.builder.ins().jump(cont_block, &[]);
				}
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
