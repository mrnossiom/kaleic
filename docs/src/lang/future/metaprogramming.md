# Metaprogramming

## Reading

- [The Metaprogramming Dilemma - gingerBill](https://www.gingerbill.org/article/2016/12/01/the-metaprogramming-dilemma/) (Odin)

  Defines a list of metaprogramming categories:

  > - Introspection (and Reflection for OOP languages)
  > - Compile Time Execution (CTE)
  > - Template Programming
  > - Macros (Textual and Syntactic)
  > - Parametric Polymorphism (“Generics”)

## Samples

How to balance between user power and good ergonomics.

### Shape intrinsic

- Motivation: do not depend on text/token manipulation macros

```kalei
let shape = Shape::<Directions>::new().variants
// equals: let shape = &[Brush, Line, Rectangle]
```

### Inline statements

- Motivation: do not depend on text/token manipulation macros

```kalei
struct Foo { bla: Boo, bar: Baz }

fn deserialize(s: &str) -> T {
  let result = Foo::default();
  inline for field in Shape::new(result).fields {
    field.value = deserialize(s);
  }
  result
}
```

```kalei
inline if std.cfg.host.is_linux() {
  
} else {
  
}
```

- [Inline Loops - Zig](https://zig.guide/language-basics/inline-loops/)

### Assignment reflection

> [..] I very frequently find myself wanting to know the name I’m “about to be assigned to”.
>
> [Evelyn Wood, about language design](https://eev.ee/blog/2015/02/28/sylph-the-programming-language-i-want/#:~:text=I%20very%20frequently%20find%20myself%20wanting%20to%20know%20the%20name%20I%E2%80%99m%20%E2%80%9Cabout%20to%20be%20assigned%20to%E2%80%9D.)

- Motivation: in languages where you specify a lot of data inline (e.g. Nix) you tend often have `"name" = function("name", { options })`

- Concern: value changes with an LSP rename. Maybe be the source of confusion
- Concern: makes the expression dependent on context → you cannot extract that expression somewhere else

```kalei
let kalei = Source::new(`name);
// becomes
let kalei = Source::new("kalei");
```
