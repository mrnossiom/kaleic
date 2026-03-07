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
