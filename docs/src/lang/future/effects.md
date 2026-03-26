# Effects

## Reading

- [Effekt lang](https://effekt-lang.org/)

- [Ante lang](https://antelang.org/)

- (Paper) [Algebraic Effects for Functional Programming](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/algeff-tr-2016-v2.pdf)

- [Algebraic Effects For The Rest Of Us — Dan Abramov](https://overreacted.io/algebraic-effects-for-the-rest-of-us/)

- [Pondering Effects - alopex.li](https://wiki.alopex.li/PonderingEffects)

	Focuses on static effects (i.e. effects that give information at compile time, not attached to the concept of coroutine).

## Samples

### Name resolution

```
effect Resolve {
  fn resolve(name: Sym) &ref Ty;
}

fn main() {
	let items = <..>;
	let map = Map<Sym, Ty>;

	// I don't like the try/catch-like syntax for effects
	handle {
		items.for |item| {
			'resolve(item);
		}
	} with {
		// no type in effect handling
		fn resolve(name) {
			map.entry(name).match {
				// not long arrows
				.Occupied |ty| effect_continue &ty
				.Vaccant || {
					// it's not in the handle ctx anymore???
					let ty = resolve_item(name);
					map.insert(name, ty);
					effect_continue &ty;
				}
			}
		}
	}
}

fn resolve_item(item: hir::Item) Ty \ Resolve {
	// translate hir to ty
	// ...
	let ty = 'resolve(item.subtype);
	// ...
	ty
}
```
