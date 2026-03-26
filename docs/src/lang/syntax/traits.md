# Traits

Rust-like traits.

## Design

### Definition

```kalei
trait Foo {
  fn bar() uint;

  fn baz() {
    // snip
  }
}
```

### Implementation

```kalei
struct MyType;

impl Foo for MyType {
  // notice how there is no type in impl block
  //
  // concern: against code locality
  fn bar() {
    // chosen by a fair roll of dice
    4
  }
}
```
