# Parsing

## Future

### Flat AST storage

- [Carbon Docs - Parsing](https://github.com/carbon-language/carbon-lang/blob/trunk/toolchain/docs/parse.md)

- [Super-Flat ASTs (HackerNews comments)](https://news.ycombinator.com/item?id=46150677)

  Raises important question of perf vs. DevEx.

Some random idea of representation

`type Foo { id: u32 }`

lexes as this stream with token length

`IDENT(4) IDENT(3) LBRA(1) IDENT(2) COLON(1) IDENT(3) RBRA(1)`

which parses as this tree stream (braces are not represented in the stream)

`{ NAME { NAME TY } FIELDS_END } ITEM_END`
