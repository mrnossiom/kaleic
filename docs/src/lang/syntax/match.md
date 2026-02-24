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

- [Open Rust RFC: Postfix match - GitHub](https://github.com/rust-lang/rfcs/pull/3295)

### Arms

Small arrow syntax (languages)

```kalei
<match> {
  123 -> {}
  else -> {}
}
```

Rust large arrow syntax

```kalei
<match> {
  123 => {}
  _ => {}
}
```

no arrows

```kalei
<match> {
  123 { 456 }
  _ {
    // -snip-
  }
}
```
