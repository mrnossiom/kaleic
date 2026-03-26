# Metaprogramming

## Reading

- [The Metaprogramming Dilemma - gingerBill](https://www.gingerbill.org/article/2016/12/01/the-metaprogramming-dilemma/) (Odin)

  Defines a list of metaprogramming categories:

  > - Introspection (and Reflection for OOP languages)
  > - Compile Time Execution (CTE)
  > - Template Programming
  > - Macros (Textual and Syntactic)
  > - Parametric Polymorphism (“Generics”)

  My goal for generics is to have
  generics obviously,
  some kind of checked template programming that uses introspection,
  vm/interpreted compile time execution that is sandboxed and is proven deterministic but can still access things like the heap (Rust const time is way too restrictive, too much hacks)

- [Parametricity, or Comptime is Bonkers](https://noelwelsh.com/posts/comptime-is-bonkers/)

  Though I have not used Zig much,
  I agree that their comptime are is much free form,
  and although all abstractions are leaky, it makes it too easy to depend on some construction

  Maybe never allow compiler intrinsic to output information without a trait (except `TypeId`).

  ```kalei
  // has to be the identity because T has no observable properties
  fn foo<T>(foo: T) T {}

  // as opposed to
  fn foo<T: Any>(value: &T) _ {} // we only care about opaque value identity, is used for type maps context in Rust
  fn foo<T: Shape>() _ {} // e.g. shows that it uses T for reflection
  fn foo<T: Layout>() _ {} // layout only gives information for size/alignement
  ```

- [facet - GitHub](https://github.com/facet-rs/facet)

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
