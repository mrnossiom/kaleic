# Future

## Sandbox Sample

Many of the tests shown here are usual constructs in programming but tried in a left-to-right syntax.
I would like to explore the LTR space

### `is` Operator

```kalei
var option = Some(42);
if option is Some(num) { }
if option.is Some(num) { }
```

### `not` Operator

```kalei
var cond = true;
while cond.! // or cond.not() (e.g. `vec.is_empty().not()`)
  and cond is true
{
  // snip
}
```

### Loop Constructs

```kalei
var stmts = [];

var stmts_iter = stmts.iter();
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

### Left-to-Right Trait implementation

```kalei
for Struct impl Trait { }
Struct::impl Trait { }
Struct.impl Trait { }
Struct.impl { }
```

### Postfix deref operator

```kalei
cst og_num = 10;
var myref = &og_num;
var mynum = myref.*;

fn deref(bar: T) {
	bar.mac_call().*;
	bar.field.*;

	bar.field.mac_call().*;
	bar.mac_call().field.*;
}
```

### Postfix try control-flow operator

```kalei
var res = Ok(98);
var num = res.?;
```

### Type ascription

```kalei
// annotate functions as such
var myexpr = f() : uint;

var myvec = (
  (1..=100)
  .into_iter()
  .collect() : Vec<_>
)
.into_iter()
.map((|x| x+1):fn(uint) -> uint);
```

### Generics

```kalei
fn forget[T](v: T) void;
fn substring['a](s: &'a str) &'a str;
fn substring(s: &str) &str;

fn foo<T>(arg: T) T {
	arg
}
```

### Alternative keywords

My rationale for using verbs instead of nouns for keywords is that they are less likely to conflict with user variable names.

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
