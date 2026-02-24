# Effects

Reference:

- [Effekt lang](https://effekt-lang.org/)

- [Ante lang](https://antelang.org/)

## Sample 1

```kalei
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
