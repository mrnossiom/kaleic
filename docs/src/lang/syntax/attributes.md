# Attributes

## Design

- Which character to introduce attribute? `#`, `@`

  `@` clashes with bindings in `match` arms.

```kalei
#attr
#attr("foo")
#attr{name = "c"}
fn item() {
  #rustfmt::skip
  let matrix = [
    1, 2, 3,
    4, 5, 6,
  ];

  #allow(clippy::placeholder_name)
  let foo = ();
}
```

- Concern: Bare attributes (i.e. no `[]` to delimit) on expression can conflict

  ```
  #foo (1)
  #foo() 1

  #foo [1, 2, 3]
  #foo[] 1
  ````

  Currently, attributes have precedence.

  - Do we not have bare attributes? `#foo` is `#foo()`
  - Do we delimit the attribute?
