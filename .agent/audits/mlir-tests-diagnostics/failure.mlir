module {
  func.func @f() {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(qc-to-qco)",
      disable_threading: true,
      verify_each: true
    }
  }
#-}
