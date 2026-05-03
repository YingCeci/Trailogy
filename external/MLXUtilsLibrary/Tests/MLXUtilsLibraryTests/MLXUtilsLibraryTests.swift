import Testing
import Foundation
@testable import MLXUtilsLibrary

@Test func example() async throws {
  guard let fileURL = Bundle.module.url(forResource: "voices", withExtension: "npz", subdirectory: "Resources") else {
    Issue.record("Could not find voices.npz in test bundle")
    return
  }
  let dict = NpyzReader.read(fileFromPath: fileURL)
  #expect(dict != nil)
}
