# Match

## Design

### Invocation

#### (find name)

```kalei
match variable { <arms> }
match foo.method() { <arms> }
```

#### Postfix

```kalei
let variable = Some(42);

variable.match { <arms> }
foo.method().match { <arms> }

foo.method().match { <arms> }.continue()
```

- [Rust RFC (Open): Postfix Match - GitHub](https://github.com/rust-lang/rfcs/pull/3295)

### Arms

Small arrow syntax (languages)

```
<match> {
  123 -> {}
  else -> {}
}
```

Rust large arrow syntax

```
<match> {
  123 => {}
  _ => {}
}
```

no arrows

```
<match> {
  123 { 456 }
  _ {
    // -snip-
  }
}
```
