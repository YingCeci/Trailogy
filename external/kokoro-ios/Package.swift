// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "KokoroSwift",
  platforms: [
    .iOS(.v18), .macOS(.v15)
  ],
  products: [
    .library(
      name: "KokoroSwift",
      type: .dynamic,
      targets: ["KokoroSwift"]
    ),
  ],
  dependencies: [
    // VENDORED CHANGE: relaxed from `exact: "0.30.2"` to a range so KokoroSwift
    // can coexist with mlx-swift-lm 3.x (which requires MLX 0.31+). MLX
    // 0.30 → 0.31 is a minor bump with no breaking changes for KokoroSwift's
    // use.
    .package(url: "https://github.com/ml-explore/mlx-swift", "0.30.2" ..< "1.0.0"),
    // .package(url: "https://github.com/mlalma/eSpeakNGSwift", from: "1.0.1"),
    // VENDORED CHANGE: MisakiSwift and MLXUtilsLibrary are also vendored
    // siblings (their upstream Package.swifts also have exact-0.30.2
    // pins). Use relative paths so the whole graph resolves consistently.
    .package(path: "../MisakiSwift"),
    .package(path: "../MLXUtilsLibrary")
  ],
  targets: [
    .target(
      name: "KokoroSwift",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "MLXFFT", package: "mlx-swift"),
        // .product(name: "eSpeakNGLib", package: "eSpeakNGSwift"),
        .product(name: "MisakiSwift", package: "MisakiSwift"),
        .product(name: "MLXUtilsLibrary", package: "MLXUtilsLibrary")
      ],
      resources: [
       .copy("../../Resources/")
      ]
    ),
    .testTarget(
      name: "KokoroSwiftTests",
      dependencies: ["KokoroSwift"]
    ),
  ]
)
