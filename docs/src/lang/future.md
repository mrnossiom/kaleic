# Future

All the syntax here are thoughts on language design that may or may not be interesting to implement in a language.
Most of the syntax here came from a quick though, is bad and needs to be bikeshed.

Modern programming languages should optimize for readability.

## Reading

- [Left to Right Programming (Programs Should Be Valid as They Are Typed)](https://graic.net/p/left-to-right-programming)

  When writing a fragment of code,
  the sooner you give context information, the sooner the language server is able to help you back.
  That is by giving type information, methods and functions discovery, etc.

## Syntax Samples

Many of the tests shown here are usual constructs in programming but tried in a left-to-right syntax.
I would like to explore the LTR design space.

### `is` operator

- Motivation: LTR programming

  Would work especially well with path inference.

Equiv. (Rust) `if let <pattern> = <expr>`

```kalei
let option = Some(42);
if option is Some(num) { }
if option.is Some(num) { }
```

- [Rust RFC (Accepted): If Let (GitHub)](https://rust-lang.github.io/rfcs/0160-if-let.html)

### `not` postfix operator

- Motivation: LTR programming

Would replace the current `not` prefix operator.

```kalei
if not vec.is_empty() { }
// becomes
if vec.is_empty().not { }
if vec.is_empty().not() { } // keep as a boolean method?
```

### postfix deref operator

- Motivation: LTR programming

```kalei
let num = 10;
let myref = &num;
let mynum = myref.*;

bar.mac_call().*;
bar.field.*;
```

### LTR Loop statements

```kalei
let stmts = [];

let stmts_iter = stmts.iter();
while stmts_iter.next() is Some(stmt) { }

in stmts for stmt { }
in stmts the stmt { }

for stmts each stmt { }
for stmts be stmt { }

all stmts be stmt { }
loop stmts for stmt { }

stmts.for |stmt| { }
stmts.iter().enumerate().for |i, stmt| { }
```

### LTR Trait implementation

```kalei
for Struct impl Trait { }
Struct::impl Trait { }
Struct.impl Trait { }
Struct.impl { }
```

### Postfix try control-flow operator

```kalei
let res = Ok(98);
let num = res.?;
```

### Type ascription

- Motivation: better type inference ergonomics

```kalei
// annotate functions as such
let myexpr = f() : uint;

let myvec = (
  (1..=100)
  .into_iter()
  .collect() : Vec<_>
)
.into_iter()
.map((|x| x+1):fn(uint) -> uint);
```

### Generics

Which syntax to adopt?

- `< >` collide with comparison operators and need special handling (e.g. Rust's turbofish)

```kalei
fn forget[T](v: T);
fn substring['a](s: &'a str) &'a str;
fn substring(s: &str) &str;

fn foo<T>(arg: T) T {
	arg
}
```

### Path

Which syntax to adopt?

```kalei
use std.arith.Plus; // can be confused with field access
use std/arith/Plus; // no
use std::arith::Plus;
```

### Alternative keywords

My rationale for using verbs instead of nouns for keywords is that they are less likely to conflict with user variable names.
On the other hand, it's difficult to have a wide range of verbs with similar semantic or just a verb for too many things.
Using one or the other would be more coherent.

- Verb keywords:
  - `def` for values (functions, constants)
  - `decl`/`dcl` for type-level constructs (structs, enums, etc.)
  - `let` for local values (variables)
- Noun keywords:
  - `fn`/`func`
  - `type`/`struct`/`enum`
  - `var`/`val`
  - `cst`/`const`

```kalei
def func() { }
def constant = 42;

trait Add {
  // dcl → declare instead of `fn`
  dcl Output;
  // def → define instead of `type/struct/enum`
  def add(self, other: Self) Self::Output;
}
```

### Callsite code

- Motivation: Avoid excessive monomorphization

```kalei
fn as_ref_str(s: impl AsRef<str>) {
  @callsite let s = s.as_ref();
  // snip
}

// alternative    vvv?
fn into_string(s: can Into<String>) {
  let s = s.into();
  // snip
}

fn callsite() {
  let my_str = "...";

  as_ref_str(my_str);
  into_string(my_str);
  // instead of the following to avoid generics
  as_ref_str(my_str.as_ref());
  as_ref_str(&my_str);
  into_string(my_str.into());
}
```

### Arbitrary prefixed string literals

This would could integrate well with a good comptime system. Is it worth?

```kalei
comptime fn regex(lit: &str) { }
let regex = regex""
// could lower to
let regex = comptime { regex("") }

// e.g. c"im a cstr"
```

### Enforce purity and capabilities

Enforce purity through effects, doc_effects, capability system or another system that suits.

Track
- writes to stdio
- calls to unsafe
- hidden interaction with randomness (e.g. Rust HashMap)
- allocations

This is one of the largest goals here

### Require value binding

Can be simply implemented as a lint. Acts like if everything was (Rust) `#[must_use]`

### Rational number type

Motivation: experiment with floats as a last resort construct and work with rational numbers

- [Numbers](https://eev.ee/blog/2015/02/28/sylph-the-programming-language-i-want/#numbers)
- [Rational — Julia](https://docs.julialang.org/en/v1/base/numbers/#Base.Rational)

### no semicolon in statements, functional syntax?

- OCaml syntax?

```
fn foo(arg: Ty) ->
  a = make_foo();
  b = make_bar(a);
  ()
```

Concern: how to know to return or not last value
- depend on the type system? consider it a return if type matches, seems rebust to errors

```
fn boo(arg: Ty) {
  _ = make_blah()
}
```

### Hot-reloading

Would be fun to play with that. Kind of like `dotnet watch`.

### Path inference

```kalei
enum InferKind {
  Infer,
  NoInfer,
}

fn foo(qrcode: InferKind);

foo(.Infer);

match InferKind.Infer {
  .Infer -> {}
  .NoInfer -> {}
}
```

- [Rust RFC (Open): Path Inference - GitHub](https://github.com/rust-lang/rfcs/pull/3444)

### Pipe operator/construct

Motivation: avoid nesting calls or `let` repetition

Question: how to tell where are going the arguments

Related: See [Uniform function call syntax](ufcs-wiki)

[ucfs-wiki]: https://en.wikipedia.org/wiki/Uniform_function_call_syntax

```kalei
let value =
  compute_value(my_looooooooog_value_name)
  |> transform_value1()
  |> transform_value2()
  |> transform_value3();

// reuse closure syntax?
let value =
  compute_value(my_looooooooog_value_name)
  |> transform_value1
  |> |v| transform_value2(v)
  |> |v| transform_value3(v, 2);

// semantically equivalent to
let value = transform_value3(transform_value2(transform_value1(compute_value(my_looooooooog_value_name))), 3);
// which may format as
let value = transform_value3(
  transform_value2(transform_value1(compute_value(
  	my_looooooooog_value_name,
  ))),
  3,
);

// or chaining the same name to avoid nesting and keeping clarity
let value = compute_value(my_looooooooog_value_name);
let value = transform_value1(value);
let value = transform_value2(value);
let value = transform_value3(value, 3);
```

### Operators

```
@ pattern
| pattern, bit-or
& pattern, bit-and
_ pattern
~ pattern?, bit-not?

# attributes

% bitop? do not use for rem
* bitop
= bitop
+ bitop
- bitop
/ bitop

? unary try
! unary not

\ escape

'' char, lifetime?
"" string
`` ?

() delim value param
[] delim index
{} delim stmts, group
<> delim type param
, punct param
: type
; stmt end

. field

^ ???
$ raw ident?

struct Fields {
	foo: uint,
	$let: uint,
}

fields.$let
```

### Exhaustive patterns in `match`

See <https://arxiv.org/abs/2504.18920v3>

### Zig style formatter

Last comma in struct style constructs defines if we should align it

```kalei
{ a , b , c }
// becomes (why would struct have whitespace when literally every other construct doesn't)
{a, b, c}

{ a , b , c , }
// becomes
{
  a,
  b,
  c,
}

{ loooooooooooooooong_names, that_make_the_construct_go_wayyyyyyyyyyyyy, over_the_limit_zzzzzzzzzzzzzzzzzzz }
// is forced to be formatted flattened
{
  loooooooooooooooong_names,
  that_make_the_construct_go_wayyyyyyyyyyyyy,
  over_the_limit_zzzzzzzzzzzzzzzzzzz,
}
```
