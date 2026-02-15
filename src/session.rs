//! Common data for front related operations

use std::{
	cmp,
	collections::HashSet,
	fmt, io,
	ops::{self, Sub},
	path::{Path, PathBuf},
	process,
	rc::Rc,
	sync::atomic::{AtomicBool, Ordering},
};

use ariadne::{Config, IndexType, ReportKind};
use parking_lot::RwLock;
use string_interner::{StringInterner, Symbol as _, backend::StringBackend, symbol::SymbolU32};

use crate::{bug, codegen::Backend};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
	pub start: BytePos,
	pub end: BytePos,
}

impl Span {
	pub(crate) const DUMMY: Self = Self::new(BytePos(u32::MAX), BytePos(u32::MAX));

	#[must_use]
	pub const fn new(start: BytePos, end: BytePos) -> Self {
		Self { start, end }
	}

	#[must_use]
	pub fn to(self, span: Self) -> Self {
		Self {
			start: cmp::min(self.start, span.start),
			end: cmp::max(self.end, span.end),
		}
	}

	#[must_use]
	pub const fn start(self) -> Self {
		Self {
			start: self.start,
			end: self.start,
		}
	}

	#[must_use]
	pub const fn end(self) -> Self {
		Self {
			start: self.end,
			end: self.end,
		}
	}
}

impl ariadne::Span for Span {
	type SourceId = BytePos;
	fn source(&self) -> &Self::SourceId {
		&self.start
	}
	fn start(&self) -> usize {
		self.start.to_usize()
	}
	fn end(&self) -> usize {
		self.end.to_usize()
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

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol(SymbolU32);

impl fmt::Debug for Symbol {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		#[cfg(feature = "debug")]
		let interned = INTERNER.with(|i| {
			i.get().map_or(Ok(false), |i| {
				i.read().resolve(self.0).map_or(Ok(false), |str| {
					if f.alternate() {
						write!(f, "{str}").map(|()| true)
					} else {
						write!(f, "`{str}`#{}", self.0.to_usize()).map(|()| true)
					}
				})
			})
		})?;
		#[cfg(not(feature = "debug"))]
		let interned = false;

		if !interned {
			write!(f, "sym#{:?}", self.0.to_usize())?;
		}

		Ok(())
	}
}

#[cfg(feature = "debug")]
thread_local! {
	static INTERNER: std::sync::OnceLock<std::sync::Arc<RwLock<StringInterner<StringBackend>>>> = std::sync::OnceLock::default();
}

pub struct SymbolInterner {
	#[cfg(feature = "debug")]
	inner: std::sync::Arc<RwLock<StringInterner<StringBackend>>>,
	#[cfg(not(feature = "debug"))]
	inner: RwLock<StringInterner<StringBackend>>,
}

impl fmt::Debug for SymbolInterner {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("SymbolInterner").finish_non_exhaustive()
	}
}

impl Default for SymbolInterner {
	fn default() -> Self {
		let inner = RwLock::default();
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
	pub fn intern(&self, symbol: &str) -> Symbol {
		Symbol(self.inner.write().get_or_intern(symbol))
	}

	#[must_use]
	pub fn resolve(&self, symbol: Symbol) -> String {
		match self.inner.read().resolve(symbol.0) {
			Some(s) => s.to_owned(),
			None => bug!("there is a single symbol interner, thus all symbol are valid"),
		}
	}
}

#[derive(Debug)]
pub struct SessionCtx {
	pub options: Options,
	diag_cx: DiagnosticCtx,

	pub symbols: SymbolInterner,
	pub source_map: Rc<RwLock<SourceMap>>,
}

impl SessionCtx {
	pub fn new() -> Self {
		let source_map = Rc::new(RwLock::new(SourceMap::default()));
		let diag_cx = DiagnosticCtx::new(source_map.clone());
		Self {
			options: Options::default(),
			diag_cx,
			symbols: SymbolInterner::default(),
			source_map,
		}
	}

	pub const fn dcx(&self) -> &DiagnosticCtx {
		&self.diag_cx
	}
}

impl Default for SessionCtx {
	fn default() -> Self {
		Self::new()
	}
}

#[derive(Debug)]
pub struct DiagnosticCtx {
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
	pub fn emit_build(&self, report: ReportBuilder) {
		self.emit(&Diagnostic::new(report));
	}

	pub fn emit(&self, diag: &Diagnostic) {
		if diag.report.kind == ReportKind::Error {
			self.degraded.store(true, Ordering::Relaxed);
		}

		let cache = self.source_map.read();
		if let Err(err) = diag.report.write(&*cache, io::stderr()) {
			tracing::error!(?err, "could not print diagnostic");
		}

		#[cfg(feature = "debug")]
		eprintln!("error was emitted here: {}", diag.loc);
	}

	pub fn emit_fatal(&self, diagnostic: &Diagnostic) -> ! {
		self.emit(diagnostic);
		process::exit(1);
	}

	pub fn check_sane_or_exit(&self) {
		if self.degraded.load(Ordering::Relaxed) {
			println!("Emitted at least one error!");
			process::exit(1);
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PrintKind {
	// IRs
	Ast,
	AstPretty,
	HigherIr,
	HigherIrPretty,
	BackendIr,

	CollectedItems,
	TypeEnvironment,
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

	pub jit: bool,
	pub output: PathBuf,

	pub backend: Backend,

	// TODO: replace with an enum
	pub print: HashSet<PrintKind>,
}

impl Default for Options {
	fn default() -> Self {
		Self {
			input: Default::default(),
			jit: true,
			output: PathBuf::from("build"),
			backend: Default::default(),
			print: Default::default(),
		}
	}
}

#[derive(Debug)]
pub struct Diagnostic {
	report: Box<Report>,
	#[cfg(feature = "debug")]
	loc: &'static std::panic::Location<'static>,
}

impl Diagnostic {
	#[must_use]
	#[track_caller]
	pub fn new(report: ReportBuilder) -> Self {
		let config = Config::new().with_index_type(IndexType::Byte);
		Self {
			report: Box::new(report.with_config(config).finish()),
			#[cfg(feature = "debug")]
			loc: std::panic::Location::caller(),
		}
	}
}

#[derive(Debug, Clone)]
pub struct SourceFile {
	pub name: String,
	pub content: String,
	pub offset: BytePos,
}

#[derive(Debug, Default)]
pub struct SourceMap {
	sources: Vec<Rc<SourceFile>>,
	diagnostic_sources: Vec<ariadne::Source>,
	offset: BytePos,
}

impl SourceMap {
	pub fn load_source_from_file(&mut self, path: &Path) -> io::Result<Rc<SourceFile>> {
		let filename = path
			.file_name()
			.ok_or_else(|| io::Error::other("expected a source file"))?;
		let src = std::fs::read_to_string(path)?;

		let filename = filename.to_string_lossy();

		let ext = filename.split('.').next_back().unwrap_or("");
		#[expect(clippy::manual_assert, reason = "to be replaced")]
		if ext != "kl" {
			// TODO: use diagnostic ctx
			panic!("file extension is not `kl`")
		}

		let source = self.load_source(&filename, src);
		Ok(source)
	}

	pub fn load_source(&mut self, name: &str, src: String) -> Rc<SourceFile> {
		let src_len = BytePos::from_usize(src.len());

		let src_file = Rc::new(SourceFile {
			name: name.to_owned(),
			content: src.clone(),
			offset: self.offset,
		});

		let diagnostic_src = ariadne::Source::from(src);

		self.sources.push(src_file.clone());
		self.diagnostic_sources.push(diagnostic_src);
		self.offset = self.offset + src_len;

		src_file
	}

	pub fn lookup_source_file_idx(&self, pos: BytePos) -> FileIdx {
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
	pub fn fetch_span(&self, span: Span) -> &str {
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
		Result::<_, &'static str>::Ok(&self.diagnostic_sources[file_idx.to_usize()])
	}
	fn display<'a>(&self, id: &'a BytePos) -> Option<impl fmt::Display + 'a> {
		let file_idx = self.lookup_source_file_idx(*id);
		Some(self.sources[file_idx.to_usize()].name.clone())
	}
}

#[derive(Debug, Clone, Copy)]
pub struct FileIdx(usize);

impl FileIdx {
	const fn new(idx: usize) -> Self {
		Self(idx)
	}

	const fn to_usize(self) -> usize {
		self.0
	}
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BytePos(u32);

impl BytePos {
	pub const fn from_u32(pos: u32) -> Self {
		Self(pos)
	}

	pub fn from_usize(pos: usize) -> Self {
		match u32::try_from(pos) {
			Ok(pos) => Self(pos),
			Err(_) => bug!("tried to construct a `BytePos` out of valid values"),
		}
	}

	pub const fn to_u32(self) -> u32 {
		self.0
	}

	pub const fn to_usize(self) -> usize {
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

pub type Report = ariadne::Report<Span, ReportKind>;
pub type ReportBuilder = ariadne::ReportBuilder<Span, ReportKind>;
