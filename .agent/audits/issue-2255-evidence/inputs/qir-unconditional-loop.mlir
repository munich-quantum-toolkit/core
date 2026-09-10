module {
  llvm.func @main() attributes {passthrough = ["entry_point"]} {
    llvm.br ^loop
  ^loop:
    llvm.br ^loop
  }
}
