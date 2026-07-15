# OpenQASM grammar

The files in `Grammar/` come from the OpenQASM 3.1.0 release at commit
`c717508162a0eac892fa32134716fe77a284e835` of
<https://github.com/openqasm/openqasm>. They are licensed under Apache-2.0. The
only local grammar adjustment corrects the misspelled internal token name
`VERSION_IDENTIFIER` to `VERSION_IDENTIFIER`; it does not change accepted text.

The files in `Generated/` were generated from that grammar with ANTLR 4.13.2:

```console
java -jar antlr-4.13.2-complete.jar -Dlanguage=Cpp -visitor -no-listener \
  -o ../Generated qasm3Lexer.g4 qasm3Parser.g4
```

Generated parser implementation is kept in a separate library. Normal builds do
not require Java or regenerate these files. Generated and upstream grammar
sources are excluded from repository-wide typo, license-header, and C++
formatting rewrites so regeneration remains reviewable.
