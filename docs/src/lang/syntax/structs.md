# Structs

## Design

```kalei
enum Option<T> {
	Some(uint),
	None,
}

enum Toto {
	Foo,
	Bar(uint),
	Baz { tata: uint }
}

struct Baz {
	content: uint,
}

// struct NamedTuple<T, U> {
// 	foo: T,
// 	bar: U,
// }

fn foo(option: Option) { }

/*

enum Option { Foo, Bar }

fn main() {
	let maybe = .Some(10);
	match maybe {
		.Some()
		.None => {}
		else => {}
	}

	let baz = Baz { content: 10 };
	let inner = baz.content;
	let _ { content } = &baz;

	foo(.Foo);
}
```
