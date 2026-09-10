module {
  func.func private @first(i64)
  func.func private @second()
  func.func @test(%input: i64) {
    %a = arith.addi %input, %input : i64
    %b = arith.addi %a, %input : i64
    func.call @first(%b) : (i64) -> ()
    func.call @second() : () -> ()
    return
  }
}
