//! Common data for front related operations

use std::{
	cell::Cell,
	cmp,
	collections::HashSet,
	fmt, fs,
	io::{self, Write as _},
	ops::{self, Sub},
	path::{Path, PathBuf},
	process,
	rc::Rc,
	sync::atomic::{AtomicBool, Ordering},
};

use ariadne::{Config, IndexType, ReportKind};
use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::{
	ast, bug,
	codegen::{Backend, Linker},
	collect::{DefId, NameEnvironment},
	hir,
	resolve::{EarlyResolution, LangItem},
	symbols::SymbolInterner,
	ty::Put,
};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Span {
	pub(crate) start: BytePos,
	pub(crate) end: BytePos,
}

impl Span {
	pub(crate) const DUMMY: Self = Self::new(BytePos(u32::MAX), BytePos(u32::MAX));

	#[must_use]
	pub(crate) const fn new(start: BytePos, end: BytePos) -> Self {
		Self { start, end }
	}

	#[must_use]
	pub(crate) fn to(self, span: Self) -> Self {
		Self {
			start: cmp::min(self.start, span.start),
			end: cmp::max(self.end, span.end),
		}
	}

	#[must_use]
	pub(crate) const fn start(self) -> Self {
		Self {
			start: self.start,
			end: self.start,
		}
	}

	#[must_use]
	pub(crate) const fn end(self) -> Self {
		Self {
			start: self.end,
			end: self.end,
		}
	}
}

// FIXME: ariadne api doesn't allow a source_map api with lightweight span
// this relies on internal behaviour and should be replace with a custom implementation
thread_local! {
	static OFFSET_HACK: Cell<usize> = const { Cell::new(0) };
}

impl ariadne::Span for Span {
	type SourceId = BytePos;
	fn source(&self) -> &Self::SourceId {
		&self.start
	}
	fn start(&self) -> usize {
		self.start.to_usize() - OFFSET_HACK.get()
	}
	fn end(&self) -> usize {
		self.end.to_usize() - OFFSET_HACK.get()
	}
}

impl Sub<BytePos> for Span {
	type Output = Self;
	fn sub(self, rhs: BytePos) -> Self::Output {
		Self {
			start: self.start - rhs,
			end: self.end - rhs,
		}
	}
}

impl fmt::Debug for Span {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "sp#{}..{}", self.start.to_u32(), self.end.to_u32())
	}
}

#[derive(Debug)]
pub struct SessionCtx {
	dcx: DiagnosticCtx,

	pub options: Options,
	pub(crate) symbols: SymbolInterner,
	pub(crate) source_map: Rc<RwLock<SourceMap>>,

	// collect
	pub(crate) name_env: Put<NameEnvironment>,
	pub(crate) lang_items: Put<FxHashMap<LangItem, DefId>>,
	pub(crate) node_id_to_def_id: Put<FxHashMap<ast::NodeId, DefId>>,
	// resolve
	pub(crate) resolution_map: Put<FxHashMap<ast::NodeId, EarlyResolution>>,
	// lower
	pub(crate) node_id_to_hir_id: Put<FxHashMap<ast::NodeId, hir::NodeId>>,
}

impl SessionCtx {
	pub(crate) fn new() -> Self {
		let source_map = Rc::new(RwLock::new(SourceMap::default()));
		let dcx = DiagnosticCtx::new(source_map.clone());
		Self {
			options: Options::default(),
			symbols: SymbolInterner::default(),
			source_map,

			dcx,

			name_env: Put::default(),
			lang_items: Put::default(),
			node_id_to_def_id: Put::default(),
			resolution_map: Put::default(),
			node_id_to_hir_id: Put::default(),
		}
	}
}

pub(crate) trait ScxHandle {
	fn scx(&self) -> &SessionCtx;
}

impl ScxHandle for SessionCtx {
	fn scx(&self) -> &SessionCtx {
		self
	}
}

pub(crate) trait DcxHandle {
	fn dcx(&self) -> &DiagnosticCtx;
}

impl DcxHandle for DiagnosticCtx {
	fn dcx(&self) -> &DiagnosticCtx {
		self
	}
}

impl<T: ScxHandle> DcxHandle for T {
	fn dcx(&self) -> &DiagnosticCtx {
		&self.scx().dcx
	}
}

pub(crate) struct ArtefactWriter(fs::File);

impl fmt::Write for ArtefactWriter {
	fn write_str(&mut self, s: &str) -> fmt::Result {
		write!(self.0, "{s}").map_err(|_| fmt::Error)
	}
}

impl SessionCtx {
	pub(crate) fn register_artefact(
		&self,
		meta: &ArtefactKind,
		f: impl FnOnce(&mut ArtefactWriter) -> fmt::Result,
	) {
		if self.options.print.contains(&meta.kind()) {
			let file = fs::File::create(self.options.debug_output.join(meta.filename())).unwrap();
			f(&mut ArtefactWriter(file)).unwrap();
		}
	}
}

impl Default for SessionCtx {
	fn default() -> Self {
		Self::new()
	}
}

#[derive(Debug)]
pub(crate) struct DiagnosticCtx {
	degraded: AtomicBool,

	source_map: Rc<RwLock<SourceMap>>,
}

impl DiagnosticCtx {
	fn new(source_map: Rc<RwLock<SourceMap>>) -> Self {
		Self {
			degraded: AtomicBool::default(),
			source_map,
		}
	}

	#[track_caller]
	pub(crate) fn emit_build(&self, report: ReportBuilder) {
		self.emit(&Diagnostic::new(report));
	}

	pub(crate) fn emit(&self, diag: &Diagnostic) {
		if diag.report.kind == ReportKind::Error {
			self.degraded.store(true, Ordering::Relaxed);
		}

		{
			let cache = self.source_map.read();
			if let Err(err) = diag.report.write(&*cache, io::stderr()) {
				eprintln!("could not print diagnostic: {err:?}");
			}
		}

		#[cfg(feature = "debug")]
		eprintln!("error was emitted here: {}", diag.loc);
	}

	pub(crate) fn emit_fatal(&self, diagnostic: &Diagnostic) -> ! {
		self.emit(diagnostic);
		process::exit(1);
	}

	pub(crate) fn check_sane_or_exit(&self) {
		if self.degraded.load(Ordering::Relaxed) {
			println!("Emitted at least one error!");
			process::exit(1);
		}
	}
}

type TubeId = ();

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PrintKind {
	// IRs
	Ast,
	AstPretty,
	HigherIr,
	HigherIrPretty,
	BackendIr,

	NameEnv,
	TypeEnv,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ArtefactKind {
	// IRs
	Ast(TubeId),
	AstPretty(TubeId),
	HigherIr(TubeId),
	HigherIrPretty(TubeId),
	BackendIr(DefId, Backend),

	NameEnv(TubeId),
	TypeEnv(TubeId),
}

impl ArtefactKind {
	pub(crate) fn kind(&self) -> PrintKind {
		match self {
			Self::Ast(..) => PrintKind::Ast,
			Self::AstPretty(..) => PrintKind::AstPretty,
			Self::HigherIr(..) => PrintKind::HigherIr,
			Self::HigherIrPretty(..) => PrintKind::HigherIrPretty,
			Self::BackendIr(..) => PrintKind::BackendIr,
			Self::NameEnv(..) => PrintKind::NameEnv,
			Self::TypeEnv(..) => PrintKind::TypeEnv,
		}
	}

	pub(crate) fn filename(&self) -> &'static str {
		match self {
			Self::Ast(tube_id) => "ast.txt",
			Self::AstPretty(tube_id) => "ast-pretty.kl",
			Self::HigherIr(tube_id) => "hir.txt",
			Self::HigherIrPretty(tube_id) => "hir-pretty.kl",
			Self::BackendIr(def_id, Backend::Cranelift) => "bir.ll",
			Self::BackendIr(def_id, Backend::Llvm) => "bir.clif",
			Self::BackendIr(_, Backend::NoBackend) => unreachable!(),
			Self::NameEnv(tube_id) => "name-env.txt",
			Self::TypeEnv(tube_id) => "type-env.txt",
		}
	}
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub enum OutputKind {
	#[default]
	Jit,
	Object(PathBuf),
}

#[derive(Debug)]
pub struct Options {
	pub input: Option<PathBuf>,
	pub output: PathBuf,

	pub backend: Backend,
	pub jit: bool,
	pub opt: bool,
	pub linker: Linker,

	pub debug_output: PathBuf,
	pub print: HashSet<PrintKind>,
}

impl Default for Options {
	fn default() -> Self {
		Self {
			input: None,
			jit: true,
			opt: false,
			backend: Backend::default(),
			linker: Linker::default(),

			output: PathBuf::from(".cache/kaleic"),
			debug_output: PathBuf::from(".cache/kaleic/debug"),
			print: HashSet::default(),
		}
	}
}

#[derive(Debug)]
pub(crate) struct Diagnostic {
	report: Box<Report>,
	#[cfg(feature = "debug")]
	loc: &'static std::panic::Location<'static>,
}

impl Diagnostic {
	#[must_use]
	#[track_caller]
	pub(crate) fn new(report: ReportBuilder) -> Self {
		let config = Config::new().with_index_type(IndexType::Byte);
		Self {
			report: Box::new(report.with_config(config).finish()),
			#[cfg(feature = "debug")]
			loc: std::panic::Location::caller(),
		}
	}
}

#[derive(Debug, Clone)]
pub(crate) struct SourceFile {
	/// Canonicalized filename used to guess modules path
	pub(crate) path: PathBuf,
	pub(crate) content: String,
	pub(crate) offset: BytePos,
}

#[derive(Debug, Default)]
pub(crate) struct SourceMap {
	sources: Vec<Rc<SourceFile>>,
	diagnostic_sources: Vec<ariadne::Source>,
	offset: BytePos,
}

impl SourceMap {
	pub(crate) fn load_source_from_file(&mut self, path: &Path) -> io::Result<Rc<SourceFile>> {
		let path = fs::canonicalize(path)?;

		let src = std::fs::read_to_string(&path)?;
		Ok(self.load_source(path, src))
	}

	pub(crate) fn load_source(&mut self, path: PathBuf, src: String) -> Rc<SourceFile> {
		let src_len = BytePos::from_usize(src.len());

		let src_file = Rc::new(SourceFile {
			path,
			content: src.clone(),
			offset: self.offset,
		});

		let diagnostic_src = ariadne::Source::from(src);

		self.sources.push(src_file.clone());
		self.diagnostic_sources.push(diagnostic_src);
		self.offset = self.offset + src_len;

		src_file
	}

	pub(crate) fn lookup_source_file_idx(&self, pos: BytePos) -> FileIdx {
		let file_idx = self
			.sources
			.binary_search_by_key(&pos.to_u32(), |f| f.offset.to_u32())
			.unwrap_or_else(|p| {
				p.checked_sub(1).unwrap_or_else(|| {
					bug!("bytepos are handed only if there is at least a source in the file")
				})
			});
		FileIdx::new(file_idx)
	}

	#[must_use]
	pub(crate) fn fetch_span(&self, span: Span) -> &str {
		let file_idx = self.lookup_source_file_idx(span.start);
		let file = &self.sources[file_idx.to_usize()];

		let local_span = span - file.offset;

		&file.content[local_span.start.to_usize()..local_span.end.to_usize()]
	}
}

impl ariadne::Cache<BytePos> for &SourceMap {
	type Storage = String;
	fn fetch(&mut self, id: &BytePos) -> Result<&ariadne::Source<Self::Storage>, impl fmt::Debug> {
		let file_idx = self.lookup_source_file_idx(*id);
		let source = &self.sources[file_idx.to_usize()];
		OFFSET_HACK.set(source.offset.to_usize());
		let source = &self.diagnostic_sources[file_idx.to_usize()];
		Result::<_, &'static str>::Ok(source)
	}

	fn display<'a>(&self, id: &'a BytePos) -> Option<impl fmt::Display + 'a> {
		let file_idx = self.lookup_source_file_idx(*id);
		let path = self.sources[file_idx.to_usize()]
			.path
			.to_string_lossy()
			.into_owned();
		Some(path)
	}
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FileIdx(usize);

impl FileIdx {
	const fn new(idx: usize) -> Self {
		Self(idx)
	}

	const fn to_usize(self) -> usize {
		self.0
	}
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BytePos(u32);

impl BytePos {
	pub(crate) const fn from_u32(pos: u32) -> Self {
		Self(pos)
	}

	pub(crate) fn from_usize(pos: usize) -> Self {
		match u32::try_from(pos) {
			Ok(pos) => Self(pos),
			Err(_) => bug!("tried to construct a `BytePos` out of valid values"),
		}
	}

	pub(crate) const fn to_u32(self) -> u32 {
		self.0
	}

	pub(crate) const fn to_usize(self) -> usize {
		self.0 as usize
	}
}

impl ops::Add for BytePos {
	type Output = Self;
	fn add(self, rhs: Self) -> Self::Output {
		Self(self.0 + rhs.0)
	}
}

impl ops::Sub for BytePos {
	type Output = Self;
	fn sub(self, rhs: Self) -> Self::Output {
		Self(self.0 - rhs.0)
	}
}

pub(crate) type Report = ariadne::Report<Span, ReportKind>;
pub(crate) type ReportBuilder = ariadne::ReportBuilder<Span, ReportKind>;
