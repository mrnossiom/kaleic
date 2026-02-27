# Attributes

## Design

Which character to introduce attribute?

- `#`

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
- `@`

  ```kalei
  @attr
  @attr("foo")
  @attr{name = "c"}
  fn item() {
    @rustfmt::skip
    let matrix = [
      1, 2, 3,
      4, 5, 6,
    ];

    @allow(clippy::placeholder_name)
    let foo = ();
  }
  ```
