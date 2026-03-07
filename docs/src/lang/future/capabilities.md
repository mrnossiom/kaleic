# Capabilities

## Reading

- ⭐ [Context Capabilities - tmandry](https://tmandry.gitlab.io/blog/posts/2021-12-21-context-capabilities)

  Discusses the addition of a context system in Rust.
  `with` keyword for functions and traits implementations.

  Matches what I had in mind for a capability system.
  It is possible to make capabilities no-cost by using ZSTs.

  Implementation passes context as extra function arguments.

- [Designing with Static Capabilities and Effects: Use, Mention, and Invariants](https://arxiv.org/abs/2005.11444)

## Samples

```kalei
struct Env; // ZST. impl uses a global type
struct BasicArena { }

#explicit capability env = Env; // capability use-case
#implicit capability arena = BasicArena; // context use-case

fn std::env::var(s: AsRef<str>) String with Env {}
fn std::env::var(s: AsRef<str>) String with Env {}

fn main() uint \ {env} {
  let arena = BasicArena.new();
  use arena as Arena;

  foo()
    // pass some capabilities explicitly
    .with(Env)
}

fn foo() uint \ {arena, env} {
  let id = arena.alloc(1);
  let var = std::env::var("foo")?;
}
```
